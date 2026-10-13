"""Differentiable multi-label surface extraction in 3D.

The background grid is tetrahedralised, so at most four labels meet in a cell
and the case analysis is complete and unambiguous. Marching cubes has no such
guarantee: a cube can carry more labels than its topology can represent, which
is where multi-material marching cubes acquires its ambiguous cases.

The 3D construction is the 2D one applied recursively, one codimension at a
time, with each level solved by the shared primitives in `simplex_roots`:

    edges   1 equation on a 1-simplex   f_i = f_j              crossing
    faces   2 equations on a 2-simplex  f_i = f_j = f_k        triple point
    tets    3 equations on a 3-simplex  f_i = ... = f_l        quadruple point

Each level is located once per *shared* entity of the grid, so adjacent cells
see the same vertex and the surface closes up across cell walls by
construction. This is what makes the topology a theorem rather than a
post-process: the 2D extractor gets watertight curves from shared edge
crossings, and the 3D extractor gets watertight surfaces from shared edge
crossings and shared face triple points.

Patches
-------
Within one tet, the p|q interface patch is bounded by

  * one segment on each tet face carrying both p and q, and
  * the triple-curve segments for triples (p, q, *) interior to the tet.

Counting those gives 3 or 4 boundary segments in every case, never more, so
each patch is a triangle or a quadrilateral. A quadrilateral is split by
fanning from the lowest-indexed of its own vertices, which is a valid fan for
any polygon vertex and needs no loop ordering: the triangles are exactly
(apex, u, v) for each boundary segment (u, v) not touching the apex. No vertex
is invented, so every vertex sits on the true interface and carries an exact
implicit-function-theorem gradient.

The label triples in a tet meet along a triple curve, represented by the
segments joining face triple points to the tet's interior anchor (the
quadruple point when four labels are present, otherwise the other triple
point). These are the non-manifold edges of the output, where three material
patches meet at 120 degrees under Plateau's law.

Differentiability works exactly as in 2D: positions are found to solver
tolerance under `no_grad`, then one implicit-function-theorem correction
reattaches dx/dtheta without changing the value. Accuracy is set by the solver
and gradient correctness by the reattachment, independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from itertools import permutations

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
TRIPLE = 1
QUADRUPLE = 2

# Local edges and faces of a tet; face m is the one opposite corner m.
EDGE_TABLE = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FACE_TABLE = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
# The three label pairs drawn from a face's three labels.
FACE_PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass
class MultiLabelSurface:
    """Extracted labelled interface complex.

    vertices          (V, 3)  differentiable w.r.t. field parameters
    triangles         (S, 3)  long, oriented from the lower to the higher label
    triangle_labels   (S, 2)  long, the sorted label pair the triangle separates
    vertex_kind       (V,)    long, CROSSING / TRIPLE / QUADRUPLE
    triple_segments   (C, 2)  long, the non-manifold curve where 3 labels meet
    triple_labels     (C, 3)  long, the label triple along each such segment
    boundary_segments (B, 2)  long, patch edges lying on the domain boundary
    """

    vertices: Tensor
    triangles: Tensor
    triangle_labels: Tensor
    vertex_kind: Tensor
    vertex_labels: Tensor
    vertex_on_domain_boundary: Tensor
    # How far outside its own simplex the vertex was solved, in barycentric
    # units. Zero means the cell really did contain the feature its corner
    # labels advertised; anything positive means it did not, and the vertex is
    # a point of the equal-logit set on a branch the argmax never reaches.
    vertex_overshoot: Tensor
    # Residual of the equal-logit equations at the emitted vertex. Nonzero means
    # the solve found nothing: for a nonlinear field the branch the corner
    # labels imply need not exist near the cell at all.
    vertex_residual: Tensor
    triple_segments: Tensor
    triple_labels: Tensor
    boundary_segments: Tensor
    boundary_segment_labels: Tensor
    node_labels: Tensor
    nodes: Tensor
    tets: Tensor
    diagnostics: dict = dc_field(default_factory=dict)

    @property
    def num_quadruple_points(self) -> int:
        return int((self.vertex_kind == QUADRUPLE).sum())

    def quadruple_points(self) -> Tensor:
        return self.vertices[self.vertex_kind == QUADRUPLE]

    def triple_points(self) -> Tensor:
        return self.vertices[self.vertex_kind == TRIPLE]

    def triangle_areas(self) -> Tensor:
        p = self.vertices[self.triangles]
        return 0.5 * torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).norm(dim=-1)

    def total_area(self) -> Tensor:
        return self.triangle_areas().sum()

    def area_by_pair(self) -> dict[tuple[int, int], Tensor]:
        areas = self.triangle_areas()
        out: dict[tuple[int, int], Tensor] = {}
        for pair in self.triangle_labels.unique(dim=0).tolist():
            mask = (self.triangle_labels == torch.tensor(
                pair, device=self.triangle_labels.device)).all(dim=1)
            out[(pair[0], pair[1])] = areas[mask].sum()
        return out

    def enclosed_volume(self, label: int) -> Tensor:
        """Signed volume of region `label` via the divergence theorem.

        Only exact for regions that do not touch the domain boundary, since the
        extracted complex contains no cap there.
        """
        involved = (self.triangle_labels == label).any(dim=1)
        tris = self.triangles[involved]
        # Normals run low -> high label, so they already point out of the low
        # region; flip when `label` is the high one.
        outward = self.triangle_labels[involved, 1] == label
        tris = torch.where(outward[:, None], tris[:, [0, 2, 1]], tris)
        p = self.vertices[tris]
        return (p[:, 0] * torch.linalg.cross(p[:, 1], p[:, 2])).sum() / 6.0


def build_tetrahedral_grid(resolution: int, box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                           dtype=None, device=None) -> tuple[Tensor, Tensor]:
    """Kuhn subdivision: (resolution+1)^3 nodes, 6 tets per cube.

    Every cube is cut the same way, along its 0->7 main diagonal, with the six
    tets given by the six orderings in which the three axis steps can be taken.
    Using an identical pattern in every cube makes the triangulation conforming:
    on a shared cube face both neighbours induce the same diagonal, so the two
    sides agree on the face's two triangles and the extracted surface has no
    cracks. A cube-based grid would instead need an ambiguity resolution rule.
    """
    dtype = dtype or torch.get_default_dtype()
    x0, x1, y0, y1, z0, z1 = box
    n = resolution + 1
    xs = torch.linspace(x0, x1, n, dtype=dtype, device=device)
    ys = torch.linspace(y0, y1, n, dtype=dtype, device=device)
    zs = torch.linspace(z0, z1, n, dtype=dtype, device=device)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    nodes = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    iz, iy, ix = torch.meshgrid(
        torch.arange(resolution, device=device),
        torch.arange(resolution, device=device),
        torch.arange(resolution, device=device),
        indexing="ij",
    )
    base = ((iz * n + iy) * n + ix).reshape(-1)
    stride = (1, n, n * n)  # steps along x, y, z in the flat node index

    def corner(bits: int) -> Tensor:
        off = sum(stride[axis] for axis in range(3) if bits >> axis & 1)
        return base + off

    v_lo, v_hi = corner(0), corner(7)
    tets = []
    for p0, p1, _p2 in permutations(range(3)):
        b1 = 1 << p0
        b2 = b1 | (1 << p1)
        tets.append(torch.stack([v_lo, corner(b1), corner(b2), v_hi], dim=-1))
    return nodes, torch.cat(tets, dim=0)


def _sorted_key(indices: Tensor, num_nodes: int) -> Tensor:
    """Order-independent integer key for a set of node indices.

    This is a base-`num_nodes` packing, so it is injective only while
    `num_nodes ** arity` fits in a signed 64-bit integer. For triangular faces
    that ceiling is $2^{21}$ nodes, a $127^3$ grid; past it the key wraps
    silently and distinct faces collide. Use `_keys_fit` before relying on it,
    as `_unique_entities` does.
    """
    s = indices.sort(dim=-1).values
    key = torch.zeros(s.shape[:-1], dtype=torch.long, device=indices.device)
    for c in range(s.shape[-1]):
        key = key * num_nodes + s[..., c]
    return key


def _keys_fit(num_nodes: int, arity: int) -> bool:
    """Whether a base-`num_nodes` key of this arity stays inside int64.

    The largest key is `num_nodes ** arity - 1`, so the condition is exactly
    `num_nodes ** arity <= 2 ** 63`. Done in Python integers rather than with a
    logarithm, because the interesting case is the boundary itself: $2^{21}$
    nodes with triangular faces lands on it precisely, and still fits.
    """
    return num_nodes <= 1 or num_nodes ** arity <= 2 ** 63


def _decode_key(key: Tensor, num_nodes: int, arity: int) -> Tensor:
    parts = []
    for _ in range(arity):
        parts.append(key % num_nodes)
        key = key // num_nodes
    return torch.stack(parts[::-1], dim=-1)


def _unique_entities(local: Tensor, num_nodes: int):
    """Deduplicate per-tet entities shared between neighbouring tets.

    local: (T, n_local, arity) node indices.
    Returns (entity_nodes (E, arity), entity_id (T, n_local), tet_count (E,),
    sorted_keys (E,)); `sorted_keys` lets other entities look themselves up.
    """
    arity = local.shape[-1]
    if _keys_fit(num_nodes, arity):
        key = _sorted_key(local, num_nodes)
        uniq, inverse = torch.unique(key.reshape(-1), return_inverse=True)
        entity_nodes = _decode_key(uniq, num_nodes, arity)
        entity_id = inverse.reshape(local.shape[:-1])
        counts = torch.bincount(inverse, minlength=uniq.numel())
        return entity_nodes, entity_id, counts, uniq

    # Past the packing ceiling, fold one column at a time and compact the
    # running key after each, so it never has to distinguish num_nodes**arity
    # values at once -- only as many as there are distinct prefixes, which is
    # bounded by the number of rows. The returned key is therefore a grouping
    # label and not comparable with `_sorted_key`; the only caller that needs
    # that comparability works on edges, whose arity always fits.
    flat = local.sort(dim=-1).values.reshape(-1, arity)
    key = flat[:, 0]
    for c in range(1, arity):
        compact = torch.unique(key, return_inverse=True)[1]
        if int(compact.max()) + 1 > (2 ** 62) // max(num_nodes, 1):
            raise OverflowError("entity key would exceed int64; grid too large")
        key = compact * num_nodes + flat[:, c]

    uniq, inverse = torch.unique(key, return_inverse=True)
    counts = torch.bincount(inverse, minlength=uniq.numel())
    rep = torch.zeros(uniq.numel(), dtype=torch.long, device=local.device)
    rep.scatter_(0, inverse, torch.arange(flat.shape[0], device=local.device))
    return flat[rep], inverse.reshape(local.shape[:-1]), counts, uniq


def _gather_patch_segments(patch_id: Tensor, u: Tensor, v: Tensor, pair: Tensor,
                           num_patches: int):
    """Collect each patch's boundary segments into a fixed (P, 4, 2) array.

    A patch always has three or four boundary segments (see the module
    docstring), so four slots suffice; the count is returned so that malformed
    patches can be reported rather than silently truncated.
    """
    device = patch_id.device
    counts = torch.bincount(patch_id, minlength=num_patches)
    order = torch.argsort(patch_id, stable=True)
    owner = patch_id[order]
    start = torch.cumsum(counts, 0) - counts
    slot = torch.arange(order.numel(), device=device) - start[owner]

    seg = torch.full((num_patches, 4, 2), -1, dtype=torch.long, device=device)
    fits = slot < 4
    seg[owner[fits], slot[fits], 0] = u[order][fits]
    seg[owner[fits], slot[fits], 1] = v[order][fits]
    patch_pair = torch.zeros((num_patches, 2), dtype=torch.long, device=device)
    patch_pair[owner] = pair[order]
    return seg, patch_pair, counts


def _triangle_patch_corners(seg: Tensor) -> Tensor:
    """The three distinct corners of a patch bounded by three segments."""
    a, b = seg[:, 0, 0], seg[:, 0, 1]
    other = seg[:, 1]
    is_new = (other[:, 0] != a) & (other[:, 0] != b)
    c = torch.where(is_new, other[:, 0], other[:, 1])
    return torch.stack([a, b, c], dim=-1)


def _split_quad_patches(seg: Tensor, vertices: Tensor, normal_ref: Tensor):
    """Split each quadrilateral patch along a diagonal that does not fold.

    Fanning a quadrilateral from an arbitrary corner is only valid when the
    quadrilateral is star-shaped about that corner; picking a reflex corner
    produces two overlapping triangles that traverse their shared diagonal in
    the same direction, which breaks orientability even though the segment
    bookkeeping still looks correct. The loop is therefore ordered explicitly
    and the diagonal chosen so that both triangles wind the same way about the
    patch normal. For a simple quadrilateral at least one of the two diagonals
    always qualifies.
    """
    num = seg.shape[0]
    flat = seg.reshape(num, 8)
    v0 = flat.min(dim=1).values
    incident = (seg == v0[:, None, None]).any(dim=-1)  # (M, 4)
    opposite = torch.where(seg[..., 0] == v0[:, None], seg[..., 1], seg[..., 0])
    rank = incident.long().argsort(dim=1, descending=True, stable=True)
    v1 = opposite.gather(1, rank[:, 0:1]).squeeze(1)
    v3 = opposite.gather(1, rank[:, 1:2]).squeeze(1)
    known = (flat == v0[:, None]) | (flat == v1[:, None]) | (flat == v3[:, None])
    v2 = flat.masked_fill(known, -1).max(dim=1).values

    def wind(p: Tensor, q: Tensor, r: Tensor) -> Tensor:
        return (
            torch.linalg.cross(vertices[q] - vertices[p], vertices[r] - vertices[p]) * normal_ref
        ).sum(-1)

    through_v0 = (wind(v0, v1, v2) > 0) == (wind(v0, v2, v3) > 0)
    through_v1 = (wind(v1, v2, v3) > 0) == (wind(v1, v3, v0) > 0)
    first = torch.where(
        through_v0[:, None], torch.stack([v0, v1, v2], -1), torch.stack([v1, v2, v3], -1)
    )
    second = torch.where(
        through_v0[:, None], torch.stack([v0, v2, v3], -1), torch.stack([v1, v3, v0], -1)
    )
    return first, second, int((~through_v0 & ~through_v1).sum())


def extract(field: MultiLabelField, resolution: int = 24,
            box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), bisection_steps: int = 60,
            newton_steps: int = 8, simplex_newton_steps: int = 25,
            device=None, dtype=None) -> MultiLabelSurface:
    """Extract the labelled interface complex of `field` on a tetrahedral grid."""
    dtype = dtype or torch.get_default_dtype()
    nodes, tets = build_tetrahedral_grid(resolution, box, dtype=dtype, device=device)
    num_nodes = nodes.shape[0]
    num_classes = field.num_classes

    with torch.no_grad():
        node_labels = field.labels(nodes)

    edge_table = torch.tensor(EDGE_TABLE, device=device)
    face_table = torch.tensor(FACE_TABLE, device=device)
    edge_nodes, tet_edge_id, _, edge_keys = _unique_entities(tets[:, edge_table], num_nodes)
    face_nodes, tet_face_id, face_tet_count, _ = _unique_entities(tets[:, face_table], num_nodes)

    # ---- level 1: crossings on mixed edges, shared by every incident tet ----
    edge_labels = node_labels[edge_nodes]
    mixed_edge = edge_labels[:, 0] != edge_labels[:, 1]
    mixed_ids = mixed_edge.nonzero(as_tuple=True)[0]
    a = nodes[edge_nodes[mixed_ids, 0]]
    b = nodes[edge_nodes[mixed_ids, 1]]
    ea, eb = edge_labels[mixed_ids, 0], edge_labels[mixed_ids, 1]
    t = locate_segment_crossing(field, a, b, ea, eb, bisection_steps, newton_steps)
    crossings = a + reattach_segment_gradient(field, a, b, ea, eb, t)[:, None] * (b - a)

    edge_to_vertex = torch.full((edge_nodes.shape[0],), -1, dtype=torch.long, device=device)
    edge_to_vertex[mixed_ids] = torch.arange(mixed_ids.numel(), device=device)
    num_crossings = mixed_ids.numel()

    # ---- level 2: triple points on faces carrying three labels ----
    face_labels = node_labels[face_nodes]  # (F, 3)
    face_edge_key = _sorted_key(
        torch.stack([face_nodes, face_nodes.roll(-1, dims=1)], dim=-1), num_nodes
    )  # local edge m joins face corners m and (m+1) % 3
    face_edge_id = torch.searchsorted(edge_keys, face_edge_key)  # (F, 3)
    face_mixed = face_labels != face_labels.roll(-1, dims=1)
    triple_face = face_mixed.sum(dim=1) == 3

    tp_ids = triple_face.nonzero(as_tuple=True)[0]
    face_to_tp = torch.full((face_nodes.shape[0],), -1, dtype=torch.long, device=device)
    triples = crossings.new_zeros((0, 3))
    triple_residual = crossings.new_zeros((0,))
    triple_overshoot = crossings.new_zeros((0,))
    if tp_ids.numel():
        corners = nodes[face_nodes[tp_ids]]  # (M, 3, 3)
        labels3 = face_labels[tp_ids]
        x_init = crossings[edge_to_vertex[face_edge_id[tp_ids]]].mean(dim=1).detach()
        u = locate_simplex_equal_logits(
            field, corners, labels3,
            u_init=affine_coords(corners, x_init), newton_steps=simplex_newton_steps,
        )
        u_diff = reattach_simplex_gradient(field, corners, u, labels3)
        triples = affine_point(corners, u_diff)
        triple_residual = simplex_residual_norm(field, corners, u_diff.detach(), labels3)
        triple_overshoot = barycentric_overshoot(u_diff.detach())
        face_to_tp[tp_ids] = num_crossings + torch.arange(tp_ids.numel(), device=device)
    num_triples = tp_ids.numel()

    # ---- level 3: quadruple points in tets carrying four labels ----
    tet_labels = node_labels[tets]  # (T, 4)
    sorted_tl = tet_labels.sort(dim=1).values
    num_distinct = 1 + (sorted_tl[:, 1:] != sorted_tl[:, :-1]).sum(dim=1)
    quad_tet_ids = (num_distinct == 4).nonzero(as_tuple=True)[0]

    tp_of_face = face_to_tp[tet_face_id]  # (T, 4), -1 where the face is not a triple face
    tet_to_quad = torch.full((tets.shape[0],), -1, dtype=torch.long, device=device)
    quads = crossings.new_zeros((0, 3))
    quad_residual = crossings.new_zeros((0,))
    quad_overshoot = crossings.new_zeros((0,))
    if quad_tet_ids.numel():
        corners = nodes[tets[quad_tet_ids]]  # (M, 4, 3)
        labels4 = tet_labels[quad_tet_ids]
        face_pts = torch.cat([crossings, triples], dim=0)[tp_of_face[quad_tet_ids]]
        x_init = face_pts.mean(dim=1).detach()
        u = locate_simplex_equal_logits(
            field, corners, labels4,
            u_init=affine_coords(corners, x_init), newton_steps=simplex_newton_steps,
        )
        u_diff = reattach_simplex_gradient(field, corners, u, labels4)
        quads = affine_point(corners, u_diff)
        quad_residual = simplex_residual_norm(field, corners, u_diff.detach(), labels4)
        quad_overshoot = barycentric_overshoot(u_diff.detach())
        tet_to_quad[quad_tet_ids] = (
            num_crossings + num_triples + torch.arange(quad_tet_ids.numel(), device=device)
        )

    vertices = torch.cat([crossings, triples, quads], dim=0)
    num_vertices = vertices.shape[0]

    # ---- per-face boundary segments, one per label pair present on the face ----
    # A 3-label face contributes three segments radiating from its triple point;
    # a 2-label face contributes the single segment joining its two crossings.
    num_faces = face_nodes.shape[0]
    seg_u = torch.full((num_faces, 3), -1, dtype=torch.long, device=device)
    seg_v = torch.full((num_faces, 3), -1, dtype=torch.long, device=device)
    seg_pair = torch.zeros((num_faces, 3, 2), dtype=torch.long, device=device)

    if tp_ids.numel():
        cross_of_edge = edge_to_vertex[face_edge_id[tp_ids]]  # (M, 3)
        seg_u[tp_ids] = face_to_tp[tp_ids][:, None].expand(-1, 3)
        seg_v[tp_ids] = cross_of_edge
        la = face_labels[tp_ids]
        lb = la.roll(-1, dims=1)
        seg_pair[tp_ids] = torch.stack([torch.minimum(la, lb), torch.maximum(la, lb)], dim=-1)

    two_label_face = (face_mixed.sum(dim=1) == 2).nonzero(as_tuple=True)[0]
    if two_label_face.numel():
        order = face_mixed[two_label_face].long().argsort(dim=1, descending=True, stable=True)
        e0 = face_edge_id[two_label_face].gather(1, order[:, 0:1]).squeeze(1)
        e1 = face_edge_id[two_label_face].gather(1, order[:, 1:2]).squeeze(1)
        seg_u[two_label_face, 0] = edge_to_vertex[e0]
        seg_v[two_label_face, 0] = edge_to_vertex[e1]
        lab = face_labels[two_label_face]
        seg_pair[two_label_face, 0] = torch.stack(
            [lab.min(dim=1).values, lab.max(dim=1).values], dim=-1
        )

    # ---- interior triple-curve segments: each face triple point to the anchor ----
    # The anchor is the quadruple point when the tet has four labels, and
    # otherwise the tet's other triple point, so a 3-label tet emits the single
    # chord joining its two triple points.
    first_tp_slot = (tp_of_face >= 0).long().argmax(dim=1)
    anchor = tp_of_face.gather(1, first_tp_slot[:, None]).squeeze(1)
    anchor = torch.where(tet_to_quad >= 0, tet_to_quad, anchor)

    active_tets = (num_distinct >= 2).nonzero(as_tuple=True)[0]
    curve_tets = (num_distinct >= 3).nonzero(as_tuple=True)[0]

    tri_seg = torch.zeros((0, 2), dtype=torch.long, device=device)
    tri_seg_labels = torch.zeros((0, 3), dtype=torch.long, device=device)
    if curve_tets.numel():
        tp_here = tp_of_face[curve_tets]  # (M, 4)
        anc = anchor[curve_tets]
        sel = (tp_here >= 0) & (tp_here != anc[:, None])
        rows, slots = sel.nonzero(as_tuple=True)
        tri_seg = torch.stack([tp_here[rows, slots], anc[rows]], dim=-1)
        tri_seg_labels = face_labels[tet_face_id[curve_tets][rows, slots]].sort(dim=-1).values
        tri_seg_tet = curve_tets[rows]

    # ---- assemble patch boundary segments as (tet, label pair, u, v) rows ----
    tf = tet_face_id[active_tets].reshape(-1)
    owner = active_tets.repeat_interleave(4)
    valid = seg_u[tf] >= 0
    row_tet = owner[:, None].expand(-1, 3)[valid]
    row_u = seg_u[tf][valid]
    row_v = seg_v[tf][valid]
    row_pair = seg_pair[tf][valid]
    row_on_boundary = (face_tet_count[tf] == 1)[:, None].expand(-1, 3)[valid]

    boundary_segments = torch.stack([row_u, row_v], dim=-1)[row_on_boundary]
    boundary_segment_labels = row_pair[row_on_boundary]

    if tri_seg.shape[0]:
        pair_idx = torch.tensor(FACE_PAIRS, device=device)  # (3, 2)
        curve_pairs = tri_seg_labels[:, pair_idx]  # (C, 3, 2)
        row_tet = torch.cat([row_tet, tri_seg_tet.repeat_interleave(3)])
        row_u = torch.cat([row_u, tri_seg[:, 0].repeat_interleave(3)])
        row_v = torch.cat([row_v, tri_seg[:, 1].repeat_interleave(3)])
        row_pair = torch.cat([row_pair, curve_pairs.reshape(-1, 2)])

    # ---- triangulate each patch ----
    group = row_tet * (num_classes * num_classes) + row_pair[:, 0] * num_classes + row_pair[:, 1]
    _, patch_id = torch.unique(group, return_inverse=True)
    num_patches = int(patch_id.max()) + 1 if patch_id.numel() else 0
    seg, patch_pair, seg_count = _gather_patch_segments(
        patch_id, row_u, row_v, row_pair, num_patches
    )

    # Reference normal for the patch, pointing from the lower to the higher
    # label. Taken once per patch rather than per triangle so that a patch's
    # triangles cannot be oriented inconsistently with each other.
    if num_patches:
        present = seg[..., 0] >= 0
        pts = vertices.detach()[seg.clamp_min(0)]  # (P, 4, 2, 3)
        centre = (pts * present[..., None, None]).sum(dim=(1, 2)) / (
            2 * present.sum(dim=1)
        )[:, None]
        with torch.no_grad():
            grads = field.logit_grads(centre)
        rows = torch.arange(num_patches, device=device)
        normal_ref = grads[rows, patch_pair[:, 1]] - grads[rows, patch_pair[:, 0]]
    else:
        normal_ref = vertices.new_zeros((0, 3))

    tri_list, owner_list = [], []
    tri_patches = (seg_count == 3).nonzero(as_tuple=True)[0]
    if tri_patches.numel():
        tri_list.append(_triangle_patch_corners(seg[tri_patches]))
        owner_list.append(tri_patches)
    quad_patches = (seg_count == 4).nonzero(as_tuple=True)[0]
    folded = 0
    if quad_patches.numel():
        first, second, folded = _split_quad_patches(
            seg[quad_patches], vertices.detach(), normal_ref[quad_patches]
        )
        tri_list += [first, second]
        owner_list += [quad_patches, quad_patches]

    if tri_list:
        triangles = torch.cat(tri_list, dim=0)
        tri_patch = torch.cat(owner_list, dim=0)
    else:
        triangles = torch.zeros((0, 3), dtype=torch.long, device=device)
        tri_patch = torch.zeros((0,), dtype=torch.long, device=device)
    triangle_labels = patch_pair[tri_patch]

    # Orient every triangle along its patch's reference normal. Flipping is a
    # discrete relabelling of corners and does not affect gradients.
    if triangles.shape[0]:
        p = vertices[triangles].detach()
        normal = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        flip = (normal * normal_ref[tri_patch]).sum(-1) < 0
        triangles = torch.where(flip[:, None], triangles[:, [0, 2, 1]], triangles)

    vertex_kind = torch.cat(
        [
            torch.full((num_crossings,), CROSSING, dtype=torch.long, device=device),
            torch.full((num_triples,), TRIPLE, dtype=torch.long, device=device),
            torch.full((quads.shape[0],), QUADRUPLE, dtype=torch.long, device=device),
        ]
    )
    vertex_labels = torch.full((num_vertices, 4), -1, dtype=torch.long, device=device)
    vertex_labels[:num_crossings, :2] = torch.stack(
        [torch.minimum(ea, eb), torch.maximum(ea, eb)], dim=-1
    )
    if num_triples:
        vertex_labels[num_crossings:num_crossings + num_triples, :3] = (
            face_labels[tp_ids].sort(dim=-1).values
        )
    if quads.shape[0]:
        vertex_labels[num_crossings + num_triples:] = tet_labels[quad_tet_ids].sort(dim=-1).values

    vertex_overshoot = torch.cat(
        [crossings.new_zeros(num_crossings), triple_overshoot, quad_overshoot]
    )
    vertex_residual = torch.cat(
        [crossings.new_zeros(num_crossings), triple_residual, quad_residual]
    )

    node_on_boundary = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    for axis, (lo_v, hi_v) in enumerate(zip(box[0::2], box[1::2])):
        node_on_boundary |= (nodes[:, axis] == lo_v) | (nodes[:, axis] == hi_v)
    vertex_on_domain_boundary = torch.zeros(num_vertices, dtype=torch.bool, device=device)
    vertex_on_domain_boundary[:num_crossings] = node_on_boundary[edge_nodes[mixed_ids]].all(dim=1)
    if num_triples:
        vertex_on_domain_boundary[num_crossings:num_crossings + num_triples] = (
            node_on_boundary[face_nodes[tp_ids]].all(dim=1)
        )

    def _peak(x: Tensor) -> float:
        return float(x.max()) if x.numel() else 0.0

    diagnostics = {
        "resolution": resolution,
        "num_nodes": num_nodes,
        "num_tets": int(tets.shape[0]),
        "num_faces": num_faces,
        "num_crossings": num_crossings,
        "num_triple_points": num_triples,
        "num_quadruple_points": int(quads.shape[0]),
        "num_triangles": int(triangles.shape[0]),
        "num_patches": num_patches,
        # A patch should always be a triangle or a quadrilateral; anything else
        # means the case analysis was violated, and a fold means no diagonal
        # split the patch cleanly.
        "malformed_patches": int(((seg_count != 3) & (seg_count != 4)).sum()),
        "folded_patches": folded,
        "num_triple_segments": int(tri_seg.shape[0]),
        "num_three_label_tets": int((num_distinct == 3).sum()),
        "num_four_label_tets": int(quad_tet_ids.numel()),
        "max_triple_residual": _peak(triple_residual),
        "max_quad_residual": _peak(quad_residual),
        "max_triple_overshoot": _peak(triple_overshoot),
        "max_quad_overshoot": _peak(quad_overshoot),
        "labels_present": sorted(set(node_labels.tolist())),
    }

    return MultiLabelSurface(
        vertices=vertices,
        triangles=triangles,
        triangle_labels=triangle_labels,
        vertex_kind=vertex_kind,
        vertex_labels=vertex_labels,
        vertex_on_domain_boundary=vertex_on_domain_boundary,
        vertex_overshoot=vertex_overshoot,
        vertex_residual=vertex_residual,
        triple_segments=tri_seg,
        triple_labels=tri_seg_labels,
        boundary_segments=boundary_segments,
        boundary_segment_labels=boundary_segment_labels,
        node_labels=node_labels,
        nodes=nodes,
        tets=tets,
        diagnostics=diagnostics,
    )


def _edge_pair_keys(u: Tensor, v: Tensor, pair: Tensor, num_vertices: int,
                    num_classes: int) -> Tensor:
    """Undirected (edge, label pair) key, for matching triangle sides."""
    lo = torch.minimum(u, v)
    hi = torch.maximum(u, v)
    return ((lo * num_vertices + hi) * num_classes + pair[:, 0]) * num_classes + pair[:, 1]


def check_topology(surface: MultiLabelSurface) -> dict:
    """Verify the watertightness and orientation invariants in 3D.

    Every side of every triangle is examined together with the label pair of
    the patch it belongs to. A side is allowed to be used once only if it lies
    on a triple curve, where three patches meet along a non-manifold edge, or
    on the domain boundary, where the surface is legitimately cut off. Every
    other side must be used exactly twice, and the two uses must traverse it in
    opposite directions, which certifies that each patch is a closed, coherently
    oriented surface. Separately, each triple-curve segment must carry exactly
    the three patches of its label triple.
    """
    num_vertices = surface.vertices.shape[0]
    num_classes = int(surface.node_labels.max()) + 1
    tris = surface.triangles
    if tris.numel() == 0:
        return {"ok": True, "num_triangles": 0}

    u = tris[:, [0, 1, 2]].reshape(-1)
    v = tris[:, [1, 2, 0]].reshape(-1)
    pair = surface.triangle_labels.repeat_interleave(3, dim=0)
    key = _edge_pair_keys(u, v, pair, num_vertices, num_classes)
    uniq, inverse = torch.unique(key, return_inverse=True)
    use_count = torch.bincount(inverse, minlength=uniq.numel())

    # Directed traversals must cancel on any side used twice.
    direction = torch.where(u < v, 1, -1)
    net = torch.zeros(uniq.numel(), dtype=torch.long, device=tris.device)
    net.scatter_add_(0, inverse, direction)

    # Sides permitted to be used once: triple curves and the domain boundary.
    allowed_single = []
    if surface.triple_segments.numel():
        pair_idx = torch.tensor(((0, 1), (0, 2), (1, 2)), device=tris.device)
        cp = surface.triple_labels[:, pair_idx].reshape(-1, 2)
        cu = surface.triple_segments[:, 0].repeat_interleave(3)
        cv = surface.triple_segments[:, 1].repeat_interleave(3)
        allowed_single.append(_edge_pair_keys(cu, cv, cp, num_vertices, num_classes))
    if surface.boundary_segments.numel():
        allowed_single.append(
            _edge_pair_keys(
                surface.boundary_segments[:, 0], surface.boundary_segments[:, 1],
                surface.boundary_segment_labels, num_vertices, num_classes,
            )
        )
    single_ok = (
        torch.isin(uniq, torch.cat(allowed_single)) if allowed_single
        else torch.zeros_like(uniq, dtype=torch.bool)
    )

    bad_open = int(((use_count == 1) & ~single_ok).sum())
    bad_overused = int((use_count > 2).sum())
    bad_orientation = int(((use_count == 2) & (net != 0)).sum())

    # Every triple-curve segment must be met by exactly its three patches.
    curve_patches_wrong = 0
    if surface.triple_segments.numel():
        curve_keys = allowed_single[0].reshape(-1, 3)
        present = torch.isin(curve_keys, uniq[use_count >= 1])
        curve_patches_wrong = int((~present).sum())

    degenerate = int((surface.triangle_areas() <= 0).sum())
    mislabelled = int((surface.triangle_labels[:, 0] == surface.triangle_labels[:, 1]).sum())

    return {
        "ok": bad_open == 0 and bad_overused == 0 and bad_orientation == 0
        and curve_patches_wrong == 0 and degenerate == 0 and mislabelled == 0,
        "unpaired_sides": bad_open,
        "sides_used_more_than_twice": bad_overused,
        "inconsistently_oriented_sides": bad_orientation,
        "missing_patches_on_triple_curves": curve_patches_wrong,
        "degenerate_triangles": degenerate,
        "triangles_with_equal_labels": mislabelled,
        "num_triangles": int(tris.shape[0]),
        "num_sides": int(uniq.numel()),
        "num_boundary_sides": int(surface.boundary_segments.shape[0]),
    }


def certify(field: MultiLabelField, resolution: int = 24,
            box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), probes: int = 15,
            device=None, dtype=None) -> dict:
    """Check the assumptions the corner-label case analysis rests on.

    `extract` reads a cell's structure off its corner labels: one crossing per
    mixed edge, one triple point per three-label face, one quadruple point per
    four-label tet. Those readings are assumptions, not consequences, and this
    reports where they fail:

    ``edges_with_extra_label``
        A mixed edge along which some third label is the argmax carries more
        than one crossing, so the single crossing placed on it is wrong.
    ``faces_without_interior_triple_point``
        A face can show three distinct corner labels while the three regions
        never meet inside it, because a middle label separates the other two.
        The point solved for then lies outside the face and is a point of the
        triple set on a branch that the argmax never reaches.
    ``tets_without_interior_quadruple_point``
        The same failure one dimension up.
    ``tets_with_unseen_labels``
        A label occupying a cell's interior without reaching any of its corners
        is invisible to the case analysis entirely.

    The counts matter more than they look. In 2D the triple set is a set of
    isolated points, so a triangle either contains one or is nowhere near one
    and false positives are rare. In 3D the triple set is a curve, so the
    number of faces that straddle it without containing it grows at the same
    rate as the number that do contain it, and the failure fraction does not
    shrink under refinement.
    """
    dtype = dtype or torch.get_default_dtype()
    nodes, tets = build_tetrahedral_grid(resolution, box, dtype=dtype, device=device)
    num_nodes = nodes.shape[0]
    with torch.no_grad():
        node_labels = field.labels(nodes)

    edge_table = torch.tensor(EDGE_TABLE, device=device)
    face_table = torch.tensor(FACE_TABLE, device=device)
    edge_nodes, _, _, _ = _unique_entities(tets[:, edge_table], num_nodes)
    face_nodes, _, _, _ = _unique_entities(tets[:, face_table], num_nodes)

    edge_labels = node_labels[edge_nodes]
    mixed = (edge_labels[:, 0] != edge_labels[:, 1]).nonzero(as_tuple=True)[0]
    a, b = nodes[edge_nodes[mixed, 0]], nodes[edge_nodes[mixed, 1]]
    i, j = edge_labels[mixed, 0], edge_labels[mixed, 1]
    extra = torch.zeros(mixed.numel(), dtype=torch.bool, device=device)
    with torch.no_grad():
        for p in torch.linspace(0.0, 1.0, probes + 2, dtype=dtype, device=device)[1:-1]:
            lab = field.labels(a + p * (b - a))
            extra |= (lab != i) & (lab != j)

    def _outside(entity_nodes: Tensor, entity_labels: Tensor) -> int:
        if entity_nodes.numel() == 0:
            return 0
        corners = nodes[entity_nodes]
        u = locate_simplex_equal_logits(
            field, corners, entity_labels,
            u_init=affine_coords(corners, corners.mean(dim=1)), newton_steps=30,
        )
        return int((barycentric_overshoot(u) > 0).sum())

    face_labels = node_labels[face_nodes]
    triple_faces = ((face_labels != face_labels.roll(-1, dims=1)).sum(dim=1) == 3)
    tet_labels = node_labels[tets]
    sorted_tl = tet_labels.sort(dim=1).values
    quad_tets = (1 + (sorted_tl[:, 1:] != sorted_tl[:, :-1]).sum(dim=1)) == 4

    faces_bad = _outside(face_nodes[triple_faces], face_labels[triple_faces])
    tets_bad = _outside(tets[quad_tets], tet_labels[quad_tets])

    # Sample tet interiors for labels that never reach a corner.
    grid = torch.linspace(0.0, 1.0, 7, dtype=dtype, device=device)
    bary = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    bary = bary[bary.sum(-1) <= 1.0]
    bary = torch.cat([1.0 - bary.sum(-1, keepdim=True), bary], dim=-1)
    active = ((tet_labels != tet_labels[:, 0:1]).any(dim=1)).nonzero(as_tuple=True)[0]
    unseen = 0
    for chunk in active.split(4000):
        pts = torch.einsum("sb,tbd->tsd", bary, nodes[tets[chunk]])
        with torch.no_grad():
            lab = field.labels(pts.reshape(-1, 3)).reshape(pts.shape[0], -1)
        seen = torch.zeros(lab.shape[0], field.num_classes, dtype=torch.bool, device=device)
        seen.scatter_(1, lab, True)
        at_corners = torch.zeros_like(seen)
        at_corners.scatter_(1, tet_labels[chunk], True)
        unseen += int((seen & ~at_corners).any(dim=1).sum())

    num_triple_faces = int(triple_faces.sum())
    num_quad_tets = int(quad_tets.sum())
    return {
        "ok": int(extra.sum()) == 0 and faces_bad == 0 and tets_bad == 0 and unseen == 0,
        "resolution": resolution,
        "mixed_edges": int(mixed.numel()),
        "edges_with_extra_label": int(extra.sum()),
        "three_label_faces": num_triple_faces,
        "faces_without_interior_triple_point": faces_bad,
        "four_label_tets": num_quad_tets,
        "tets_without_interior_quadruple_point": tets_bad,
        "active_tets": int(active.numel()),
        "tets_with_unseen_labels": unseen,
    }


def triple_curve_dihedrals(field: MultiLabelField, surface: MultiLabelSurface) -> list[list[float]]:
    """Dihedral angles, in degrees, between the three patches along each triple segment.

    Plateau's law in 3D says these are 120 degrees for an area-minimising
    complex, the direct analogue of the 2D junction angle test. As in 2D the
    patch tangent planes come from the field gradients rather than from
    neighbouring vertices, since the p|q patch is the level set of f_p - f_q and
    its normal at the segment is exactly grad(f_p - f_q) there. Measuring from
    mesh geometry instead would carry an O(h) curvature bias.
    """
    import math

    out: list[list[float]] = []
    if surface.triple_segments.numel() == 0:
        return out
    verts = surface.vertices.detach()
    midpoints = verts[surface.triple_segments].mean(dim=1)
    tangents = verts[surface.triple_segments[:, 1]] - verts[surface.triple_segments[:, 0]]
    tangents = tangents / tangents.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    grads = field.logit_grads(midpoints)  # (C, K, 3)

    for n in range(midpoints.shape[0]):
        i, j, k = surface.triple_labels[n].tolist()
        axis = tangents[n]
        dirs = []
        # Each patch leaves the curve in the direction lying in the patch and
        # perpendicular to the curve; project out the axis to work in the plane.
        for p, q, r in ((i, j, k), (j, k, i), (i, k, j)):
            normal = grads[n, p] - grads[n, q]
            normal = normal - (normal @ axis) * axis
            if normal.norm() < 1e-12:
                break
            normal = normal / normal.norm()
            outward = torch.linalg.cross(axis, normal)
            for sign in (1.0, -1.0):
                probe = midpoints[n] + sign * 1e-5 * outward
                with torch.no_grad():
                    f = field.logits(probe[None])[0]
                if min(float(f[p]), float(f[q])) > float(f[r]):
                    dirs.append(sign * outward)
                    break
        if len(dirs) != 3:
            continue
        # Measure the three in-plane gaps using a frame perpendicular to the axis.
        ref = dirs[0]
        perp = torch.linalg.cross(axis, ref)
        ang = sorted(
            math.degrees(math.atan2(float(d @ perp), float(d @ ref))) % 360 for d in dirs
        )
        out.append(sorted([ang[1] - ang[0], ang[2] - ang[1], 360.0 - (ang[2] - ang[0])]))
    return out
