"""Exact multi-label extraction from the arrangement of the linear interpolant.

`extract3d` reads each cell's structure off its corner labels. That reading is
an assumption, and in 3D it is wrong for a constant fraction of cells: a face
can show three distinct corner labels while the three regions never meet inside
it, because a middle label separates the other two. The fraction does not
shrink under refinement, because the 3D triple set is a curve and the faces
that straddle it grow in number at the same rate as the faces that contain it.
`extract3d.certify` measures this.

This module removes the assumption instead of patching it. Interpolate the
logits linearly over each tet and extract the argmax partition of the
*interpolant*, exactly. Three facts make that both well defined and cheap.

Each region is convex.
    On a tet the interpolated logits are affine, so region k is the tet
    intersected with the half-spaces f_k >= f_j. Every interface patch is
    therefore a convex polygon, and a convex polygon is determined by its
    corners alone: sort them by angle about the patch normal and fan. There is
    no case analysis to get wrong, and a fan of a convex polygon cannot fold,
    which is what went wrong in the corner-label version.

Every corner is a small closed-form solve.
    A corner of a patch is a point, so it is cut out by three independent
    linear conditions drawn from the tet's own faces and from equalities
    f_k = f_j. Counting how many of each gives exactly three kinds, the same
    three as before but now keyed by *label set* rather than one per cell:

        on a grid edge      f_k = f_j                 crossing
        on a grid face      f_k = f_j = f_l           triple point
        inside a tet        f_k = f_j = f_l = f_m     quadruple point

    An edge may carry several crossings and a face several triple points, which
    is precisely what the corner-label analysis could not express. Each is a
    1x1, 2x2 or 3x3 linear solve in the node logits, so positions are exact and
    ordinary autograd differentiates them; no root finder and no implicit
    function theorem are needed.

Neighbouring cells agree by construction.
    A vertex on a grid edge depends only on that edge's two node logit vectors,
    one on a face only on that face's three. Both tets sharing a face compute
    the same points from the same data, so the surface closes across cell walls
    without any welding step.

The one assumption left is that a label occupying a cell also reaches that
cell's boundary. A label region hidden strictly inside a single cell is
invisible here. Unlike the failure it replaces, that is a genuine resolution
condition and refinement removes it; `certify` reports it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from extract3d import (CROSSING, EDGE_TABLE, FACE_TABLE, QUADRUPLE, TRIPLE,
                       MultiLabelSurface, _unique_entities,
                       build_tetrahedral_grid)
from fields import MultiLabelField

TINY = 1e-30


def _perturb(node_logits: Tensor, amount: float) -> Tensor:
    """Displace the node logits by a tiny deterministic amount per (node, label).

    Everything above reads the arrangement off strict inequalities, so exact
    coincidences are the one thing that can break it, and on a regular grid they
    are not rare. Two show up immediately. A node can lie exactly on an
    interface, which happens for every sphere whose radius the grid step
    divides; the crossing on each incident edge then sits at the node, is
    rejected as out of range, and leaves a hole. A triple curve can meet a grid
    edge exactly, which happens whenever the geometry is axis-aligned to round
    numbers; the three pairwise crossings then coincide with each other and with
    a triple point on every incident face, and the patches around that point are
    assembled from several copies of the same corner.

    Rather than enumerate such cases, perturb the input until none of them
    occur, which is the usual remedy for a degenerate arrangement. Two
    properties make it sound here. The displacement depends only on the node and
    the label, so every cell touching a node sees the same value and cells still
    agree on their shared boundary; conformity is untouched. And it is a
    constant with respect to the field's parameters, so gradients are unchanged.
    The price is that positions move by O(`amount`), which at the default is ten
    orders of magnitude below the discretisation error it sits inside. Pass
    ``perturb=0`` to disable it and recover a bit-exact extraction on inputs
    that are already generic.
    """
    if amount <= 0.0:
        return node_logits
    raw = node_logits.detach()
    if raw.numel() == 0:
        return node_logits
    n, k = raw.shape
    key = torch.arange(n * k, device=raw.device, dtype=raw.dtype).reshape(n, k)
    frac = torch.sin(key * 12.9898) * 43758.5453
    frac = frac - frac.floor() - 0.5
    return node_logits + frac * (amount * raw.abs().amax().clamp_min(1.0))


def _padded_label_sets(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Boolean (M, K) label masks -> (M, Lmax) label indices padded with -1."""
    counts = mask.sum(dim=1)
    width = int(counts.max()) if counts.numel() else 0
    padded = torch.full((mask.shape[0], max(width, 1)), -1,
                        dtype=torch.long, device=mask.device)
    rows, labels = mask.nonzero(as_tuple=True)
    start = torch.cumsum(counts, 0) - counts
    padded[rows, torch.arange(rows.numel(), device=mask.device) - start[rows]] = labels
    return padded, counts


def _subsets(width: int, size: int, device) -> Tensor:
    if width < size:
        return torch.zeros((0, size), dtype=torch.long, device=device)
    return torch.combinations(torch.arange(width, device=device), r=size)


def _solve_equal_logits(corner_logits: Tensor, labels: Tensor):
    """Where do the given labels tie, in barycentric coordinates of a simplex?

    corner_logits (M, m+1, K) for an m-simplex, labels (M, m+1).
    The interpolated logit of label k is sum_a lambda_a L_a[k], so each equality
    f_{l_0} = f_{l_r} is affine in the free coordinates (lambda_1..lambda_m) and
    the tie point is one m x m solve. Returns those coordinates and a flag for
    whether the system was well posed.
    """
    m = labels.shape[1] - 1
    idx = labels[:, None, :].expand(-1, corner_logits.shape[1], -1)
    d = corner_logits.gather(2, idx)  # (M, m+1 corners, m+1 labels)
    # Row r of the system compares label 0 against label r+1.
    diff = d[..., 0:1] - d[..., 1:]  # (M, corners, m)
    A = (diff[:, 1:, :] - diff[:, 0:1, :]).transpose(1, 2)  # (M, m, m)
    rhs = -diff[:, 0, :]  # (M, m)
    det = torch.linalg.det(A.detach())
    good = det.abs() > TINY
    eye = torch.eye(m, dtype=A.dtype, device=A.device).expand_as(A)
    A_safe = torch.where(good[:, None, None], A, eye)
    return torch.linalg.solve(A_safe, rhs[..., None]).squeeze(-1), good


def _validate(corner_logits: Tensor, coords: Tensor, labels: Tensor, tol: float):
    """Is the tie point inside its simplex, and is the tied label the argmax there?

    A tie exists for any label set; it is a vertex of the arrangement only if it
    lies in the simplex and nothing beats it there. The test runs over every
    label, not just the ones the cell's boundary suggested, so that two cells
    sharing a face cannot disagree about whether their common point is real.

    Also returns, for ties that lie inside but lose, the label that beat them.
    That label is the argmax at a point of the cell, so it is certainly present
    in the cell, which is what makes the enumeration below self-correcting.
    """
    bary = torch.cat([1.0 - coords.sum(dim=1, keepdim=True), coords], dim=1)
    inside = (bary >= -tol).all(dim=1)
    values = torch.einsum("mc,mck->mk", bary, corner_logits)  # (M, K)
    tied = values.gather(1, labels[:, 0:1]).squeeze(1)
    beaten = (values > tied[:, None] + tol).any(dim=1)
    return inside & ~beaten, inside & beaten, values.argmax(dim=1)


def _frame(normal: Tensor) -> tuple[Tensor, Tensor]:
    """A deterministic orthonormal pair spanning the plane perpendicular to `normal`."""
    n = normal / normal.norm(dim=-1, keepdim=True).clamp_min(TINY)
    axis = torch.zeros_like(n)
    axis.scatter_(1, n.abs().argmin(dim=1, keepdim=True), 1.0)
    e1 = torch.linalg.cross(n, axis)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(TINY)
    return e1, torch.linalg.cross(n, e1)


def extract(field: MultiLabelField, resolution: int = 24,
            box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), tol: float = 1e-12,
            perturb: float = 1e-9, corner_label: bool = False,
            device=None, dtype=None) -> MultiLabelSurface:
    """Extract the exact argmax partition of the linearly interpolated logits.

    `corner_label=True` replaces the enumeration with the assumption a case
    table makes --- one crossing per mixed edge, between its endpoint labels,
    and one tie per cell whose corner labels are all distinct --- while leaving
    the interpolant, the closed-form solves and the assembly untouched. It
    exists so the ablation of Section 8.4 can change the combinatorics alone.
    Comparing against `extract3d` instead changes three things at once, since
    that module also solves by Newton iteration and against the field itself
    rather than its interpolant.
    """
    dtype = dtype or torch.get_default_dtype()
    nodes, tets = build_tetrahedral_grid(resolution, box, dtype=dtype, device=device)
    num_nodes = nodes.shape[0]
    node_logits = _perturb(field.logits(nodes), perturb)
    raw = node_logits.detach()
    num_classes = raw.shape[1]
    node_labels = raw.argmax(dim=1)

    edge_table = torch.tensor(EDGE_TABLE, device=device)
    face_table = torch.tensor(FACE_TABLE, device=device)
    edge_nodes, tet_edge_id, _, _ = _unique_entities(tets[:, edge_table], num_nodes)
    face_nodes, tet_face_id, face_tet_count, _ = _unique_entities(
        tets[:, face_table], num_nodes
    )

    # ---- level 1: every crossing on every mixed edge --------------------
    # A label's region is convex, so on a segment it is an interval; an edge
    # whose endpoints share a label therefore has no crossing at all, and
    # restricting to mixed edges loses nothing.
    edge_labels = node_labels[edge_nodes]
    mixed = (edge_labels[:, 0] != edge_labels[:, 1]).nonzero(as_tuple=True)[0]
    ends = edge_nodes[mixed]
    if corner_label:
        # One crossing per mixed edge, between the two endpoint labels. No
        # other pair is proposed and no dominance test is run: a third label
        # winning part of the edge is exactly what the case table cannot see.
        rows = torch.arange(mixed.numel(), device=device)
        cross_pair = edge_labels[mixed].sort(dim=1).values
    else:
        pairs = _subsets(num_classes, 2, device)
        with torch.no_grad():
            lo, hi = raw[ends[:, 0]], raw[ends[:, 1]]
            d0 = lo[:, pairs[:, 0]] - lo[:, pairs[:, 1]]
            d1 = hi[:, pairs[:, 0]] - hi[:, pairs[:, 1]]
            denom = d0 - d1
            t = d0 / torch.where(denom.abs() > TINY, denom, torch.full_like(denom, TINY))
            candidate = (denom.abs() > TINY) & (t > tol) & (t < 1.0 - tol)
            values = lo[:, None, :] + t[..., None] * (hi - lo)[:, None, :]  # (M, P, K)
            tied = values.gather(2, pairs[None, :, 0:1].expand(values.shape[0], -1, -1))
            alive = candidate & (values <= tied + tol).all(dim=2)
        rows, which = alive.nonzero(as_tuple=True)
        cross_pair = pairs[which]
    cross_edge = mixed[rows]
    a, b = nodes[ends[rows, 0]], nodes[ends[rows, 1]]
    la = node_logits[ends[rows, 0]].gather(1, cross_pair)
    lb = node_logits[ends[rows, 1]].gather(1, cross_pair)
    g0, g1 = la[:, 0] - la[:, 1], lb[:, 0] - lb[:, 1]
    t_exact = g0 / (g0 - g1)
    crossings = a + t_exact[:, None] * (b - a)
    num_crossings = crossings.shape[0]

    # Labels reachable on each edge: its endpoints plus everything the envelope
    # passes through on the way.
    edge_mask = torch.zeros((edge_nodes.shape[0], num_classes), dtype=torch.bool, device=device)
    edge_mask.scatter_(1, edge_labels, True)
    edge_mask[cross_edge.repeat_interleave(2), cross_pair.reshape(-1)] = True

    # ---- level 2: every triple point on every face -----------------------
    face_edges = tet_edge_id[:, [[3, 4, 5], [1, 2, 5], [0, 2, 4], [0, 1, 3]]]
    face_edge_id = torch.zeros((face_nodes.shape[0], 3), dtype=torch.long, device=device)
    face_edge_id[tet_face_id.reshape(-1)] = face_edges.reshape(-1, 3)
    if corner_label:
        triples, triple_face, triple_labels_v = _corner_label_level(
            node_logits, face_nodes, node_labels, nodes, device
        )
        quads, quad_tet, quad_labels_v = _corner_label_level(
            node_logits, tets, node_labels, nodes, device
        )
        face_mask = _corner_label_mask(face_nodes, node_labels, num_classes, device)
        tet_mask = _corner_label_mask(tets, node_labels, num_classes, device)
        face_rounds = tet_rounds = 0
        num_triples = triples.shape[0]
    else:
        face_mask = edge_mask[face_edge_id].any(dim=1)

        triples, triple_face, triple_labels_v, face_mask, face_rounds = _solve_level(
            node_logits, raw, face_nodes, face_mask, 3, nodes, tol
        )
        num_triples = triples.shape[0]

        # ---- level 3: every quadruple point inside every tet -------------
        # Seeded from the faces' refined label sets rather than the raw edge union.
        tet_mask = face_mask[tet_face_id].any(dim=1)
        quads, quad_tet, quad_labels_v, tet_mask, tet_rounds = _solve_level(
            node_logits, raw, tets, tet_mask, 4, nodes, tol
        )
    # Size of the candidate label set actually enumerated in each tet, which is
    # what the cost of the direct route scales with.
    tet_label_count = tet_mask.sum(dim=1)

    # A label can win strictly inside a tet without reaching its boundary, in
    # which case nothing on the boundary proposes it. Refinement recovers it
    # whenever it beats an enumerated tie; probing the interior screens for the
    # residue, which is a resolution condition rather than a bug. The probes
    # are the barycentre and four points pulled towards each corner. Five
    # samples is a screen and not a decision procedure: a region small enough
    # to fall between them is missed, so a zero here is evidence and not proof.
    tet_unreached = torch.zeros(tets.shape[0], dtype=torch.bool, device=device)
    for w in ([0.25] * 4, [0.4, 0.2, 0.2, 0.2], [0.2, 0.4, 0.2, 0.2],
              [0.2, 0.2, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4]):
        probe = raw[tets[:, 0]] * w[0]
        for c in range(1, 4):
            probe = probe + raw[tets[:, c]] * w[c]
        win = probe.argmax(dim=1)
        tet_unreached |= ~tet_mask.gather(1, win[:, None]).squeeze(1)

    vertices = torch.cat([crossings, triples, quads], dim=0)
    num_vertices = vertices.shape[0]
    vertex_kind = torch.cat([
        torch.full((num_crossings,), CROSSING, dtype=torch.long, device=device),
        torch.full((num_triples,), TRIPLE, dtype=torch.long, device=device),
        torch.full((quads.shape[0],), QUADRUPLE, dtype=torch.long, device=device),
    ])
    vertex_labels = torch.full((num_vertices, 4), -1, dtype=torch.long, device=device)
    vertex_labels[:num_crossings, :2] = cross_pair
    vertex_labels[num_crossings:num_crossings + num_triples, :3] = triple_labels_v
    vertex_labels[num_crossings + num_triples:] = quad_labels_v

    # ---- gather each patch's corners as (tet, label pair, vertex) rows ----
    row_tet, row_pair, row_vertex = _patch_rows(
        tet_edge_id, tet_face_id, cross_edge, cross_pair, num_crossings,
        triple_face, triple_labels_v, num_triples, quad_tet, quad_labels_v, device
    )

    group = row_tet * (num_classes * num_classes) + row_pair[:, 0] * num_classes + row_pair[:, 1]
    _, patch_id = torch.unique(group, return_inverse=True)
    num_patches = int(patch_id.max()) + 1 if patch_id.numel() else 0

    triangles, triangle_labels = _fan_convex_patches(
        patch_id, num_patches, row_pair, row_vertex, vertices, nodes, tets,
        node_logits.detach(), row_tet, device
    )

    # ---- triple curves: for each tet and triple, the two points on it ----
    tri_seg, tri_seg_labels = _triple_curves(
        tet_face_id, triple_face, triple_labels_v, num_crossings, num_triples,
        quad_tet, quad_labels_v, num_classes, device
    )

    on_box = torch.zeros((num_vertices, 6), dtype=torch.bool, device=device)
    for axis in range(3):
        for side, value in enumerate(box[2 * axis:2 * axis + 2]):
            on_box[:, 2 * axis + side] = (vertices[:, axis].detach() - value).abs() < 1e-9
    boundary_segments, boundary_segment_labels = _boundary_segments(
        triangles, triangle_labels, on_box, device
    )

    diagnostics = {
        "resolution": resolution,
        "num_nodes": num_nodes,
        "num_tets": int(tets.shape[0]),
        "num_crossings": num_crossings,
        "num_triple_points": num_triples,
        "num_quadruple_points": int(quads.shape[0]),
        "num_triangles": int(triangles.shape[0]),
        "num_patches": num_patches,
        "num_triple_segments": int(tri_seg.shape[0]),
        # An edge carrying two crossings is exactly the configuration the
        # corner-label analysis cannot express.
        "edges_with_multiple_crossings": int(
            (torch.bincount(cross_edge, minlength=edge_nodes.shape[0]) > 1).sum()
        ),
        "faces_with_multiple_triple_points": int(
            (torch.bincount(triple_face, minlength=face_nodes.shape[0]) > 1).sum()
        ) if num_triples else 0,
        # A patch is a polygon, so fewer than three corners means a corner was
        # rejected by the containment tolerance at a near-degenerate tie. Zero
        # throughout in float64; in float32 it runs to several hundred on the
        # same fields, which is the clearest single symptom of running this in
        # the wrong precision.
        "patches_with_too_few_corners": int(
            (torch.bincount(patch_id, minlength=num_patches) < 3).sum()
        ) if num_patches else 0,
        "labels_present": sorted(set(node_labels.tolist())),
        # The enumeration is over subsets of each tet's candidate label set, so
        # these bound its cost far more tightly than K does.
        "max_labels_per_tet": int(tet_label_count.max()) if tet_label_count.numel() else 0,
        "mean_labels_per_tet": float(tet_label_count.double().mean())
        if tet_label_count.numel() else 0.0,
        "tets_over_four_labels": int((tet_label_count > 4).sum()),
        # Enumerations run before the mask stopped changing. One means the
        # boundary seed was already complete everywhere; two means either that
        # one round of growth was then confirmed, or that the first round
        # found only ties beaten by labels already in the mask.
        "seed_refinement_rounds": max(int(face_rounds), int(tet_rounds)),
        # Tets in which one of five interior probes is won by a label the
        # enumeration never saw: a screen for the hidden-region condition. See
        # the note above -- zero here is evidence, not a guarantee.
        "tets_with_unreached_label": int(tet_unreached.sum()),
    }

    return MultiLabelSurface(
        vertices=vertices,
        triangles=triangles,
        triangle_labels=triangle_labels,
        vertex_kind=vertex_kind,
        vertex_labels=vertex_labels,
        vertex_on_domain_boundary=on_box.any(dim=1),
        # Both are identically zero here: a vertex is emitted only after it has
        # been verified to lie in its own cell and to solve its equations.
        vertex_overshoot=vertices.new_zeros(num_vertices),
        vertex_residual=vertices.new_zeros(num_vertices),
        triple_segments=tri_seg,
        triple_labels=tri_seg_labels,
        boundary_segments=boundary_segments,
        boundary_segment_labels=boundary_segment_labels,
        node_labels=node_labels,
        nodes=nodes,
        tets=tets,
        diagnostics=diagnostics,
    )


def _enumerate_ties(raw: Tensor, entity_nodes: Tensor, mask: Tensor, size: int, tol: float):
    """Solve every candidate tie of `size` labels drawn from each cell's mask."""
    device = entity_nodes.device
    padded, counts = _padded_label_sets(mask)
    subsets = _subsets(padded.shape[1], size, device)
    active = (counts >= size).nonzero(as_tuple=True)[0]
    empty = torch.zeros((0,), dtype=torch.long, device=device)
    none = torch.zeros((0,), dtype=torch.bool, device=device)
    nothing = (empty, torch.zeros((0, size), dtype=torch.long, device=device), none, none, empty)
    if active.numel() == 0 or subsets.numel() == 0:
        return nothing

    ent = active.repeat_interleave(subsets.shape[0])
    labels = padded[active][:, subsets.reshape(-1)].reshape(-1, size)
    keep = (labels >= 0).all(dim=1)
    ent, labels = ent[keep], labels[keep]
    if ent.numel() == 0:
        return nothing

    corner_logits = raw[entity_nodes[ent]]
    coords, well_posed = _solve_equal_logits(corner_logits, labels)
    valid, blocked, winner = _validate(corner_logits, coords, labels, tol)
    return ent, labels, valid & well_posed, blocked & well_posed, winner


def _solve_level(node_logits: Tensor, raw: Tensor, entity_nodes: Tensor, entity_mask: Tensor,
                 size: int, nodes: Tensor, tol: float):
    """Find every point in these simplices where `size` labels tie and win.

    Which label sets are worth trying is seeded from the labels reachable on the
    cell's boundary, which can miss a label that occupies the cell's interior
    without touching its edges. Missing one is not harmless: the tie it should
    have produced is a corner of a neighbouring patch, and leaving it out
    punches a hole in the surface.

    The seed is therefore refined rather than trusted. Whenever a tie lands
    inside the cell but is beaten, whatever beat it is present in that cell, so
    it joins the cell's label set and the enumeration runs again. This is run to
    a fixed point rather than for a fixed number of passes. It terminates
    because the mask only ever grows and the loop stops on the first round that
    fails to grow it, so there are at most K passes. Note that a round can find
    blocked ties and still add nothing, when whatever beat them is already in
    the mask; that is the no-progress exit, not the empty-`blocked` one. In
    practice one round suffices; `rounds` is returned so a caller can tell.
    """
    device = entity_nodes.device
    mask = entity_mask.clone()
    ent = labels = valid = None
    rounds = 0
    while True:
        ent, labels, valid, blocked, winner = _enumerate_ties(
            raw, entity_nodes, mask, size, tol
        )
        rounds += 1
        if not bool(blocked.any()):
            break
        discovered = mask.clone()
        discovered[ent[blocked], winner[blocked]] = True
        if bool((discovered == mask).all()):
            break
        mask = discovered

    if ent is None or ent.numel() == 0:
        return (nodes.new_zeros((0, 3)), torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0, size), dtype=torch.long, device=device), mask, rounds)

    ent, labels = ent[valid], labels[valid]
    corner_ids = entity_nodes[ent]
    coords, _ = _solve_equal_logits(node_logits[corner_ids], labels)
    bary = torch.cat([1.0 - coords.sum(dim=1, keepdim=True), coords], dim=1)
    points = torch.einsum("mc,mcd->md", bary, nodes[corner_ids])
    return points, ent, labels.sort(dim=1).values, mask, rounds


def _corner_label_mask(entity_nodes: Tensor, node_labels: Tensor, num_classes: int,
                       device) -> Tensor:
    """The labels a case table believes a cell contains: those at its corners."""
    mask = torch.zeros((entity_nodes.shape[0], num_classes), dtype=torch.bool, device=device)
    mask.scatter_(1, node_labels[entity_nodes], True)
    return mask


def _corner_label_level(node_logits: Tensor, entity_nodes: Tensor, node_labels: Tensor,
                        nodes: Tensor, device):
    """One tie per cell whose corner labels are all distinct.

    This is the case-table reading of a cell: the corner labels name the tie to
    solve, a cell repeating a label is taken to hold no tie, and no other label
    set is tried. The solve itself is the same closed form the enumeration
    uses, so the only thing that differs is which ties get proposed.

    The result is emitted whether or not it lands inside its cell. That is not
    an oversight: a tie solved from corner labels that falls outside the cell
    is the case table's characteristic failure, and clipping it here would hide
    the effect the ablation is meant to measure.
    """
    size = entity_nodes.shape[1]
    lab = node_labels[entity_nodes]  # (E, size)
    srt = lab.sort(dim=1).values
    ent = (srt[:, 1:] != srt[:, :-1]).all(dim=1).nonzero(as_tuple=True)[0]
    empty = (nodes.new_zeros((0, 3)),
             torch.zeros((0,), dtype=torch.long, device=device),
             torch.zeros((0, size), dtype=torch.long, device=device))
    if ent.numel() == 0:
        return empty

    labels = lab[ent]
    corner_ids = entity_nodes[ent]
    coords, good = _solve_equal_logits(node_logits[corner_ids], labels)
    # A singular system names no point at all; the Newton form of the same
    # baseline cannot place one either. Dropping it is not the containment
    # test in disguise.
    ent, labels, coords, corner_ids = ent[good], labels[good], coords[good], corner_ids[good]
    if ent.numel() == 0:
        return empty
    bary = torch.cat([1.0 - coords.sum(dim=1, keepdim=True), coords], dim=1)
    points = torch.einsum("mc,mcd->md", bary, nodes[corner_ids])
    return points, ent, labels.sort(dim=1).values


def _patch_rows(tet_edge_id, tet_face_id, cross_edge, cross_pair, num_crossings,
                triple_face, triple_labels_v, num_triples, quad_tet, quad_labels_v, device):
    """Every (tet, label pair, corner vertex) incidence of every patch.

    A patch corner contributes to one pair if it is a crossing, to all three of
    its pairs if it is a triple point, and to all six if it is a quadruple
    point, because the point lies on each of those interfaces at once.
    """
    # Crossings: attach each to every tet owning its edge.
    order = torch.argsort(cross_edge, stable=True)
    sorted_edge = cross_edge[order]
    counts = torch.bincount(sorted_edge, minlength=int(tet_edge_id.max()) + 1)
    start = torch.cumsum(counts, 0) - counts
    per_tet_counts = counts[tet_edge_id]  # (T, 6)
    owner_tet = torch.arange(tet_edge_id.shape[0], device=device)
    owner_tet = owner_tet[:, None].expand(-1, 6).reshape(-1).repeat_interleave(
        per_tet_counts.reshape(-1)
    )
    offsets = torch.repeat_interleave(
        start[tet_edge_id].reshape(-1), per_tet_counts.reshape(-1)
    )
    within = torch.arange(offsets.numel(), device=device)
    run_start = torch.cumsum(per_tet_counts.reshape(-1), 0) - per_tet_counts.reshape(-1)
    within = within - torch.repeat_interleave(run_start, per_tet_counts.reshape(-1))
    picked = order[offsets + within]
    rows_tet = [owner_tet]
    rows_pair = [cross_pair[picked]]
    rows_vertex = [picked]

    pair_of_three = torch.tensor(((0, 1), (0, 2), (1, 2)), device=device)
    if num_triples:
        counts = torch.bincount(triple_face, minlength=int(tet_face_id.max()) + 1)
        order = torch.argsort(triple_face, stable=True)
        start = torch.cumsum(counts, 0) - counts
        per = counts[tet_face_id]  # (T, 4)
        owner = torch.arange(tet_face_id.shape[0], device=device)
        owner = owner[:, None].expand(-1, 4).reshape(-1).repeat_interleave(per.reshape(-1))
        offs = torch.repeat_interleave(start[tet_face_id].reshape(-1), per.reshape(-1))
        w = torch.arange(offs.numel(), device=device)
        rs = torch.cumsum(per.reshape(-1), 0) - per.reshape(-1)
        w = w - torch.repeat_interleave(rs, per.reshape(-1))
        got = order[offs + w]
        pr = triple_labels_v[got][:, pair_of_three]  # (M, 3, 2)
        rows_tet.append(owner.repeat_interleave(3))
        rows_pair.append(pr.reshape(-1, 2))
        rows_vertex.append((num_crossings + got).repeat_interleave(3))

    if quad_tet.numel():
        pair_of_four = torch.combinations(torch.arange(4, device=device), r=2)
        pr = quad_labels_v[:, pair_of_four]  # (M, 6, 2)
        rows_tet.append(quad_tet.repeat_interleave(6))
        rows_pair.append(pr.reshape(-1, 2))
        rows_vertex.append(
            (num_crossings + num_triples
             + torch.arange(quad_tet.numel(), device=device)).repeat_interleave(6)
        )

    return torch.cat(rows_tet), torch.cat(rows_pair), torch.cat(rows_vertex)


def _fan_convex_patches(patch_id, num_patches, row_pair, row_vertex, vertices,
                        nodes, tets, raw_logits, row_tet, device):
    """Order each patch's corners by angle and fan them.

    Each patch is convex, so sorting its corners by angle about the patch normal
    recovers the polygon exactly; no clipping or case analysis is involved, and
    a fan of a convex polygon cannot self-overlap. Taking the normal along
    grad(f_high - f_low) also fixes the orientation, so every triangle comes out
    facing the higher label with no separate flip pass.
    """
    if num_patches == 0:
        return (torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 2), dtype=torch.long, device=device))

    counts = torch.bincount(patch_id, minlength=num_patches)
    patch_pair = torch.zeros((num_patches, 2), dtype=torch.long, device=device)
    patch_pair[patch_id] = row_pair
    patch_tet = torch.zeros(num_patches, dtype=torch.long, device=device)
    patch_tet[patch_id] = row_tet

    # Gradient of the interpolated (f_high - f_low) on each patch's tet.
    corners = nodes[tets[patch_tet]]  # (P, 4, 3)
    edges = corners[:, 1:, :] - corners[:, 0:1, :]  # (P, 3, 3)
    g = (raw_logits[tets[patch_tet]].gather(
        2, patch_pair[:, None, :].expand(-1, 4, -1)))  # (P, 4, 2)
    g = g[..., 1] - g[..., 0]
    normal = torch.linalg.solve(edges, (g[:, 1:] - g[:, 0:1])[..., None]).squeeze(-1)

    pos = vertices.detach()[row_vertex]
    centre = torch.zeros((num_patches, 3), dtype=pos.dtype, device=device)
    centre.index_add_(0, patch_id, pos)
    centre = centre / counts.clamp_min(1)[:, None]
    e1, e2 = _frame(normal)
    rel = pos - centre[patch_id]
    angle = torch.atan2((rel * e2[patch_id]).sum(-1), (rel * e1[patch_id]).sum(-1))

    order = torch.argsort(angle, stable=True)
    order = order[torch.argsort(patch_id[order], stable=True)]
    sorted_patch = patch_id[order]
    start = torch.cumsum(counts, 0) - counts
    slot = torch.arange(order.numel(), device=device) - start[sorted_patch]

    emit = slot >= 2
    idx = emit.nonzero(as_tuple=True)[0]
    apex = row_vertex[order[start[sorted_patch[idx]]]]
    triangles = torch.stack([apex, row_vertex[order[idx - 1]], row_vertex[order[idx]]], dim=-1)
    return triangles, patch_pair[sorted_patch[idx]]


def _triple_curves(tet_face_id, triple_face, triple_labels_v, num_crossings, num_triples,
                   quad_tet, quad_labels_v, num_classes, device):
    """Segments of the curve where three labels meet, one per tet and triple.

    Inside a tet the three-label set is a straight segment, so it is pinned down
    by the two points of that triple the tet owns: triple points on its faces
    and quadruple points in its interior whose label set contains the triple.
    """
    tet_ids, labels, verts = [], [], []
    if num_triples:
        counts = torch.bincount(triple_face, minlength=int(tet_face_id.max()) + 1)
        order = torch.argsort(triple_face, stable=True)
        start = torch.cumsum(counts, 0) - counts
        per = counts[tet_face_id]
        owner = torch.arange(tet_face_id.shape[0], device=device)
        owner = owner[:, None].expand(-1, 4).reshape(-1).repeat_interleave(per.reshape(-1))
        offs = torch.repeat_interleave(start[tet_face_id].reshape(-1), per.reshape(-1))
        w = torch.arange(offs.numel(), device=device)
        rs = torch.cumsum(per.reshape(-1), 0) - per.reshape(-1)
        got = order[offs + w - torch.repeat_interleave(rs, per.reshape(-1))]
        tet_ids.append(owner)
        labels.append(triple_labels_v[got])
        verts.append(num_crossings + got)
    if quad_tet.numel():
        triples_of_four = torch.combinations(torch.arange(4, device=device), r=3)
        tet_ids.append(quad_tet.repeat_interleave(4))
        labels.append(quad_labels_v[:, triples_of_four].reshape(-1, 3))
        verts.append((num_crossings + num_triples
                      + torch.arange(quad_tet.numel(), device=device)).repeat_interleave(4))
    if not tet_ids:
        return (torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device))

    tet_ids = torch.cat(tet_ids)
    labels = torch.cat(labels)
    verts = torch.cat(verts)
    key = ((tet_ids * num_classes + labels[:, 0]) * num_classes + labels[:, 1]) * num_classes \
        + labels[:, 2]
    uniq, inverse = torch.unique(key, return_inverse=True)
    counts = torch.bincount(inverse, minlength=uniq.numel())
    order = torch.argsort(inverse, stable=True)
    start = torch.cumsum(counts, 0) - counts
    # Only pairs define a segment; a lone endpoint means the curve leaves through
    # a face the tet does not own, which cannot happen for an exact arrangement.
    full = (counts == 2).nonzero(as_tuple=True)[0]
    seg = torch.stack([verts[order[start[full]]], verts[order[start[full] + 1]]], dim=-1)
    return seg, labels[order[start[full]]]


def _boundary_segments(triangles: Tensor, triangle_labels: Tensor, on_box: Tensor, device):
    """Patch edges lying in a face of the domain box, where the surface is cut off."""
    if triangles.numel() == 0:
        return (torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0, 2), dtype=torch.long, device=device))
    u = triangles[:, [0, 1, 2]].reshape(-1)
    v = triangles[:, [1, 2, 0]].reshape(-1)
    shared = (on_box[u] & on_box[v]).any(dim=1)
    pair = triangle_labels.repeat_interleave(3, dim=0)
    return torch.stack([u[shared], v[shared]], dim=-1), pair[shared]
