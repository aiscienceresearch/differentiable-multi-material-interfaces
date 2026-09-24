"""Differentiable multi-label boundary extraction in 2D.

The background grid is triangulated. On a triangle at most three labels can
meet, which makes the case analysis complete and unambiguous:

    1 distinct corner label   interior, nothing emitted
    2 distinct corner labels  exactly two mixed edges, one interface segment
    3 distinct corner labels  three mixed edges, one triple junction, three
                              segments joining each edge crossing to it

There is no marching-squares saddle ambiguity to resolve and no four-label
case to special-case, which is the reason for preferring triangles over cubes
here.

Topology follows from where the vertices live. Crossings are computed once per
*edge* of the triangulation and shared by the (at most two) incident triangles,
so an interior mixed edge contributes exactly one vertex to exactly two
triangles. Counting segment endpoints then gives, with no further work:

    interior edge crossing  degree 2
    boundary edge crossing  degree 1  (terminates on the domain boundary)
    triple junction         degree 3

so the extracted boundary network has no dangling interior ends and no holes.
`check_topology` asserts exactly this.

Differentiability is handled by the implicit function theorem rather than by
backpropagating through the root finder. Positions are located to solver
tolerance under `no_grad`, then the gradient is reattached in closed form. For
an edge crossing at parameter t* along an edge, t* solves
    g(t, theta) = f_i(x(t)) - f_j(x(t)) = 0,
so dt*/dtheta = -(dg/dtheta) / (dg/dt), which is realised by evaluating

    t = t*.detach() - g(t*.detach(), theta) / (dg/dt).detach()

whose value is t* (since g(t*) = 0 to tolerance) and whose gradient is the
implicit derivative. Triple junctions use the 2x2 vector form of the same
identity. Two consequences matter: extraction accuracy is decoupled from
gradient correctness, and the label pair (i, j) enters the residual g itself,
so gradients flow through the label assignment and not only through vertex
positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import torch
from torch import Tensor

from fields import MultiLabelField
from simplex_roots import (
    affine_coords,
    affine_point,
    barycentric_overshoot,
    locate_segment_crossing,
    locate_simplex_equal_logits,
    reattach_segment_gradient,
    reattach_simplex_gradient,
    simplex_residual_norm,
)

# Vertex kinds
CROSSING = 0
JUNCTION = 1


@dataclass
class MultiLabelMesh:
    """Extracted labelled boundary network.

    vertices     (V, 2)  differentiable w.r.t. field parameters
    segments     (S, 2)  long, indices into `vertices`
    segment_labels (S, 2) long, the ordered pair of labels the segment separates
    vertex_kind  (V,)    long, CROSSING or JUNCTION
    vertex_on_domain_boundary (V,) bool
    node_labels  (N,)    long, argmax label at background grid nodes
    """

    vertices: Tensor
    segments: Tensor
    segment_labels: Tensor
    vertex_kind: Tensor
    vertex_on_domain_boundary: Tensor
    node_labels: Tensor
    nodes: Tensor
    triangles: Tensor
    # (V, 2) sorted label pair for crossings; (-1, -1) for junctions.
    vertex_labels: Tensor | None = None
    # (num_junctions, 3) sorted label triple for each junction.
    junction_labels: Tensor | None = None
    diagnostics: dict = dc_field(default_factory=dict)

    @property
    def num_junctions(self) -> int:
        return int((self.vertex_kind == JUNCTION).sum())

    def segment_lengths(self) -> Tensor:
        v = self.vertices[self.segments]
        return (v[:, 1] - v[:, 0]).norm(dim=-1)

    def junction_points(self) -> Tensor:
        return self.vertices[self.vertex_kind == JUNCTION]


def build_triangulated_grid(resolution: int, box=(-1.0, 1.0, -1.0, 1.0),
                            dtype=None, device=None) -> tuple[Tensor, Tensor]:
    """Uniform grid of (resolution+1)^2 nodes split into 2*resolution^2 triangles.

    Both triangles of a cell share the (v00, v11) diagonal, which keeps the
    triangulation conforming.
    """
    dtype = dtype or torch.get_default_dtype()
    x0, x1, y0, y1 = box
    xs = torch.linspace(x0, x1, resolution + 1, dtype=dtype, device=device)
    ys = torch.linspace(y0, y1, resolution + 1, dtype=dtype, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    nodes = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    n = resolution + 1
    iy, ix = torch.meshgrid(
        torch.arange(resolution, device=device), torch.arange(resolution, device=device),
        indexing="ij",
    )
    v00 = (iy * n + ix).reshape(-1)
    v10 = v00 + 1
    v01 = v00 + n
    v11 = v01 + 1
    tri_a = torch.stack([v00, v10, v11], dim=-1)
    tri_b = torch.stack([v00, v11, v01], dim=-1)
    triangles = torch.cat([tri_a, tri_b], dim=0)
    return nodes, triangles


def _unique_edges(triangles: Tensor, num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Deduplicate triangle edges.

    Returns (edge_nodes (E,2), tri_edge_id (T,3), edge_tri_count (E,)) where
    local edge m of a triangle joins its corners m and (m+1) % 3.
    """
    corners = torch.stack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], dim=1
    )  # (T, 3, 2)
    flat = corners.reshape(-1, 2)
    lo = flat.min(dim=1).values
    hi = flat.max(dim=1).values
    key = lo * num_nodes + hi
    uniq, inverse = torch.unique(key, return_inverse=True)
    edge_nodes = torch.stack([uniq // num_nodes, uniq % num_nodes], dim=-1)
    tri_edge_id = inverse.reshape(triangles.shape[0], 3)
    edge_tri_count = torch.bincount(inverse, minlength=uniq.numel())
    return edge_nodes, tri_edge_id, edge_tri_count


def extract(field: MultiLabelField, resolution: int = 64, box=(-1.0, 1.0, -1.0, 1.0),
            bisection_steps: int = 60, newton_steps: int = 8, junction_newton_steps: int = 25,
            refinement_probes: int = 7, device=None, dtype=None) -> MultiLabelMesh:
    """Extract the labelled boundary network of `field` on a triangulated grid."""
    dtype = dtype or torch.get_default_dtype()
    nodes, triangles = build_triangulated_grid(resolution, box, dtype=dtype, device=device)
    num_nodes = nodes.shape[0]

    with torch.no_grad():
        node_labels = field.labels(nodes)

    edge_nodes, tri_edge_id, edge_tri_count = _unique_edges(triangles, num_nodes)
    edge_labels = node_labels[edge_nodes]
    mixed_edge = edge_labels[:, 0] != edge_labels[:, 1]

    mixed_ids = mixed_edge.nonzero(as_tuple=True)[0]
    a = nodes[edge_nodes[mixed_ids, 0]]
    b = nodes[edge_nodes[mixed_ids, 1]]
    i = edge_labels[mixed_ids, 0]
    j = edge_labels[mixed_ids, 1]

    t = locate_segment_crossing(field, a, b, i, j, bisection_steps, newton_steps)
    t_diff = reattach_segment_gradient(field, a, b, i, j, t)
    crossings = a + t_diff[:, None] * (b - a)

    # Honest diagnostic: an edge along which some third label is the argmax, or
    # which carries more than one root, is under-resolved and needs refinement.
    with torch.no_grad():
        probes = torch.linspace(0.0, 1.0, refinement_probes + 2, dtype=dtype, device=device)[1:-1]
        third_label_hits = torch.zeros(mixed_ids.numel(), dtype=torch.bool, device=device)
        for p in probes:
            lab = field.labels(a + p * (b - a))
            third_label_hits |= (lab != i) & (lab != j)
    edges_needing_refinement = int(third_label_hits.sum())

    # Map every edge to its crossing-vertex index (-1 for unmixed edges).
    edge_to_vertex = torch.full((edge_nodes.shape[0],), -1, dtype=torch.long, device=device)
    edge_to_vertex[mixed_ids] = torch.arange(mixed_ids.numel(), device=device)

    tri_labels = node_labels[triangles]  # (T, 3)
    tri_mixed = torch.stack(
        [
            tri_labels[:, 0] != tri_labels[:, 1],
            tri_labels[:, 1] != tri_labels[:, 2],
            tri_labels[:, 2] != tri_labels[:, 0],
        ],
        dim=1,
    )  # local edge m joins corners m and (m+1) % 3
    num_mixed_edges_per_tri = tri_mixed.sum(dim=1)
    two_label = num_mixed_edges_per_tri == 2
    three_label = num_mixed_edges_per_tri == 3

    segments: list[Tensor] = []
    segment_labels: list[Tensor] = []

    # --- two-label triangles: one segment joining the two edge crossings ---
    if bool(two_label.any()):
        ids = two_label.nonzero(as_tuple=True)[0]
        order = tri_mixed[ids].to(dtype).argsort(dim=1, descending=True)[:, :2]
        e0 = tri_edge_id[ids].gather(1, order[:, 0:1]).squeeze(1)
        e1 = tri_edge_id[ids].gather(1, order[:, 1:2]).squeeze(1)
        segments.append(torch.stack([edge_to_vertex[e0], edge_to_vertex[e1]], dim=-1))
        lab = tri_labels[ids]
        segment_labels.append(
            torch.stack([lab.min(dim=1).values, lab.max(dim=1).values], dim=-1)
        )

    # --- three-label triangles: junction plus three radiating segments ---
    junctions = crossings.new_zeros((0, 2))
    junction_residual = crossings.new_zeros((0,))
    junction_outside = crossings.new_zeros((0,))
    junction_labels = torch.zeros((0, 3), dtype=torch.long, device=device)
    if bool(three_label.any()):
        ids = three_label.nonzero(as_tuple=True)[0]
        e_local = tri_edge_id[ids]  # (M, 3)
        v_local = edge_to_vertex[e_local]  # crossing vertex per local edge
        x_init = crossings[v_local].mean(dim=1).detach()
        corners = nodes[triangles[ids]]
        labels3 = tri_labels[ids]  # (M, 3), one label per corner
        u = locate_simplex_equal_logits(
            field, corners, labels3,
            u_init=affine_coords(corners, x_init), newton_steps=junction_newton_steps,
        )
        u_diff = reattach_simplex_gradient(field, corners, u, labels3)
        junctions = affine_point(corners, u_diff)
        junction_labels = labels3.sort(dim=-1).values

        # Report residual and containment for the vertices actually emitted.
        u_final = u_diff.detach()
        junction_residual = simplex_residual_norm(field, corners, u_final, labels3)
        junction_outside = barycentric_overshoot(u_final)

        base = mixed_ids.numel()
        jid = base + torch.arange(ids.numel(), device=device)
        for m in range(3):
            segments.append(torch.stack([jid, v_local[:, m]], dim=-1))
            la = tri_labels[ids, m]
            lb = tri_labels[ids, (m + 1) % 3]
            segment_labels.append(
                torch.stack([torch.minimum(la, lb), torch.maximum(la, lb)], dim=-1)
            )

    vertices = torch.cat([crossings, junctions], dim=0)
    vertex_kind = torch.cat(
        [
            torch.full((crossings.shape[0],), CROSSING, dtype=torch.long, device=device),
            torch.full((junctions.shape[0],), JUNCTION, dtype=torch.long, device=device),
        ]
    )
    on_boundary = torch.cat(
        [
            edge_tri_count[mixed_ids] == 1,
            torch.zeros(junctions.shape[0], dtype=torch.bool, device=device),
        ]
    )
    vertex_labels = torch.cat(
        [
            torch.stack([torch.minimum(i, j), torch.maximum(i, j)], dim=-1),
            torch.full((junctions.shape[0], 2), -1, dtype=torch.long, device=device),
        ]
    )

    if segments:
        segments_t = torch.cat(segments, dim=0)
        segment_labels_t = torch.cat(segment_labels, dim=0)
    else:
        segments_t = torch.zeros((0, 2), dtype=torch.long, device=device)
        segment_labels_t = torch.zeros((0, 2), dtype=torch.long, device=device)

    diagnostics = {
        "resolution": resolution,
        "num_nodes": num_nodes,
        "num_triangles": int(triangles.shape[0]),
        "num_mixed_edges": int(mixed_ids.numel()),
        "num_two_label_triangles": int(two_label.sum()),
        "num_three_label_triangles": int(three_label.sum()),
        "num_junctions": int(junctions.shape[0]),
        "edges_needing_refinement": edges_needing_refinement,
        "max_junction_residual": float(junction_residual.max()) if junction_residual.numel() else 0.0,
        # Barycentric overshoot of the emitted junction beyond its own triangle.
        # 0 means contained; values approaching 1 mean the cell is too coarse.
        "max_junction_overshoot": float(junction_outside.max()) if junction_outside.numel() else 0.0,
        "junctions_outside_triangle": int((junction_outside > 0).sum()),
        "labels_present": sorted(set(node_labels.tolist())),
    }

    return MultiLabelMesh(
        vertices=vertices,
        segments=segments_t,
        segment_labels=segment_labels_t,
        vertex_kind=vertex_kind,
        vertex_on_domain_boundary=on_boundary,
        node_labels=node_labels,
        nodes=nodes,
        triangles=triangles,
        vertex_labels=vertex_labels,
        junction_labels=junction_labels,
        diagnostics=diagnostics,
    )


def junction_angles(field: MultiLabelField, mesh: MultiLabelMesh,
                    probe: float = 1e-5) -> list[list[float]]:
    """Interior angles, in degrees, between the three interfaces at each junction.

    Tangents come from the field gradient rather than from nearby vertices. The
    i|j interface is the level set of f_i - f_j, so its tangent at the junction
    is exactly perpendicular to grad(f_i - f_j) there. Estimating the tangent by
    averaging directions to crossing vertices in a finite neighbourhood instead
    introduces a curvature bias large enough to mask the 120 degree condition.

    Only the outgoing sign is ambiguous, and it is resolved by stepping a short
    distance along each candidate and keeping the one where i and j are still
    the two dominant labels.
    """
    import math

    out: list[list[float]] = []
    if mesh.junction_labels is None:
        return out
    junction_ids = (mesh.vertex_kind == JUNCTION).nonzero(as_tuple=True)[0]
    verts = mesh.vertices.detach()

    for n, vi in enumerate(junction_ids.tolist()):
        centre = verts[vi]
        a, b, c = mesh.junction_labels[n].tolist()
        grads = field.logit_grads(centre[None])[0]  # (K, 2)
        dirs = []
        for i, j, k in ((a, b, c), (b, c, a), (a, c, b)):
            normal = grads[i] - grads[j]
            if normal.norm() < 1e-12:
                continue
            tangent = torch.stack([-normal[1], normal[0]])
            tangent = tangent / tangent.norm()
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    f = field.logits((centre + sign * probe * tangent)[None])[0]
                if min(float(f[i]), float(f[j])) > float(f[k]):
                    dirs.append(sign * tangent)
                    break
        if len(dirs) != 3:
            continue
        ang = sorted(math.degrees(math.atan2(float(d[1]), float(d[0]))) % 360 for d in dirs)
        gaps = [ang[1] - ang[0], ang[2] - ang[1], 360.0 - (ang[2] - ang[0])]
        out.append(sorted(gaps))
    return out


def check_topology(mesh: MultiLabelMesh) -> dict:
    """Verify the degree invariants the construction is supposed to guarantee.

    Interior crossings must have degree 2, crossings on the domain boundary
    degree 1, and triple junctions degree 3. Any other degree means a dangling
    end or a hole, so `ok` is the watertightness certificate.
    """
    v = mesh.vertices.shape[0]
    degree = torch.bincount(mesh.segments.reshape(-1), minlength=v)

    is_crossing = mesh.vertex_kind == CROSSING
    is_junction = mesh.vertex_kind == JUNCTION
    interior_crossing = is_crossing & ~mesh.vertex_on_domain_boundary
    boundary_crossing = is_crossing & mesh.vertex_on_domain_boundary

    bad_interior = int(((degree != 2) & interior_crossing).sum())
    bad_boundary = int(((degree != 1) & boundary_crossing).sum())
    bad_junction = int(((degree != 3) & is_junction).sum())
    isolated = int((degree == 0).sum())

    # Every segment must separate two genuinely different labels.
    mislabelled = int((mesh.segment_labels[:, 0] == mesh.segment_labels[:, 1]).sum())

    return {
        "ok": bad_interior == 0 and bad_boundary == 0 and bad_junction == 0
        and isolated == 0 and mislabelled == 0,
        "interior_crossings_wrong_degree": bad_interior,
        "boundary_crossings_wrong_degree": bad_boundary,
        "junctions_wrong_degree": bad_junction,
        "isolated_vertices": isolated,
        "segments_with_equal_labels": mislabelled,
        "num_interior_crossings": int(interior_crossing.sum()),
        "num_boundary_crossings": int(boundary_crossing.sum()),
        "num_junctions": int(is_junction.sum()),
    }
