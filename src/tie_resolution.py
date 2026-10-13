"""Exact ties: where our mesh differs from the reference, and how small eps can be.

Three measurements, all about nodes at which labels are exactly equal.

  1. The clinical field as given, against the reference implementation of
     Du et al. on the same tetrahedra. The reference evaluates its predicates
     exactly and places vertices on tied nodes; ours either leaves holes there
     (perturbation off) or splits each such vertex into a cluster of nearby
     ones (perturbation on). This locates every vertex that one mesh has and
     the other lacks.
  2. A lower bound on the perturbation. A tie is broken only if the displaced
     logits differ by more than the tolerance delta, and two labels at a node
     are displaced at least eps / K apart (times the logit scale), so eps has
     to be large compared with delta. Swept on the clinical field and on a
     field with a whole plane of tied nodes.
  3. Nodes at which three labels tie. The triple point near such a node is
     found from two of these gaps and can still fall within delta of a side of
     its face, so the separation of item 2 does not settle it. Counted on a
     field with a line of such nodes, at several resolutions.

Run with `python tie_resolution.py > ../data/tie_resolution_run.txt`.
Part 1 requires the Docker image of `compare_du.py`.
"""

from __future__ import annotations

import shutil
import tempfile

import torch
from torch import Tensor

import compare_du
import envelope3d
import extract3d
import realdata
from extract3d import _edge_pair_keys, build_tetrahedral_grid
from realdata_experiment import ORIGIN_VOX, RESOLUTION, WIDTH

torch.set_default_dtype(torch.float64)

DELTA = 1e-12


def _nearest(p: Tensor, q: Tensor, chunk: int = 512) -> Tensor:
    out = []
    for i in range(0, p.shape[0], chunk):
        d = torch.cdist(p[i:i + chunk], q, compute_mode="donot_use_mm_for_euclid_dist")
        out.append(d.min(dim=1).values)
    return torch.cat(out)


def _tie_separations(values: Tensor, eps: float) -> Tensor:
    """Gap between the two largest displaced logits at each exactly tied node."""
    top2 = values.topk(2, dim=1).values
    tied = (top2[:, 0] == top2[:, 1]).nonzero().flatten()
    displaced = envelope3d._perturb(values, eps)
    gaps = []
    for i in tied.tolist():
        labels = (values[i] == values[i].max()).nonzero().flatten()
        p = displaced[i, labels].sort(descending=True).values
        gaps.append(float(p[0] - p[1]))
    return torch.tensor(gaps)


def _closure(field, resolution, box, eps, tol=DELTA) -> str:
    surf = envelope3d.extract(field, resolution=resolution, box=box, perturb=eps, tol=tol)
    topo = extract3d.check_topology(surf)
    return f"closed {topo['ok']} ({topo['unpaired_sides']} unpaired sides)"


def clinical_against_reference(field, box) -> None:
    print("1. clinical field as given, against the reference")
    nodes, tets = build_tetrahedral_grid(RESOLUTION, box)
    with torch.no_grad():
        values = field.logits(nodes).detach()
    step = float(((nodes.max(0).values - nodes.min(0).values) / RESOLUTION).max())
    top2 = values.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    tied = (margin == 0).nonzero().flatten()
    near = ((margin > 0) & (margin < DELTA)).nonzero().flatten()
    special = torch.cat([tied, near])
    print(f"   {tied.numel()} nodes with an exact argmax tie, at values "
          f"{float(values[tied].max(1).values.min()):.3f}-{float(values[tied].max(1).values.max()):.3f}")
    print(f"   {near.numel()} more with 0 < margin < {DELTA:.0e}: margins "
          f"{float(margin[near].min()):.1e} to {float(margin[near].max()):.1e}")

    workdir = tempfile.mkdtemp(prefix="du_ties_")
    try:
        _theirs, _mesh, theirs = compare_du.run_reference(None, nodes, tets, workdir, values=values)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"   reference: {theirs.shape[0]} vertices; tied node to nearest reference vertex "
          f"{float(_nearest(nodes[tied], theirs).max()):.1e} mm, near-tied node "
          f"{float(_nearest(nodes[near], theirs).max()):.1e} mm")

    for eps in (0.0, 1e-11, 1e-10, 1e-9):
        surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=eps)
        ours = surf.vertices.detach()
        topo = extract3d.check_topology(surf)
        to_ours, to_theirs = _nearest(theirs, ours), _nearest(ours, theirs)
        print(f"   eps {eps:.0e}: ours {ours.shape[0]} vertices, closed {topo['ok']} "
              f"({topo['unpaired_sides']} unpaired sides)")
        print(f"      every reference vertex within {float(to_ours.max()):.2e} mm of one of ours, "
              f"every one of ours within {float(to_theirs.max()):.2e} mm of the reference")
        if eps == 0.0:
            missing = (to_ours > 1e-9).nonzero().flatten()
            on_nodes = int((_nearest(theirs[missing], nodes[special]) < 1e-9 * step).sum())
            print(f"      {missing.numel()} reference vertices with none of ours within 1e-9 mm; "
                  f"{on_nodes} of them on a tied or near-tied node")
        else:
            near_ties_ours = int((_nearest(ours, nodes[tied]) < 1e-6 * step).sum())
            near_ties_ref = int((_nearest(theirs, nodes[tied]) < 1e-6 * step).sum())
            print(f"      vertices within 1e-6 grid steps of an exactly tied node: "
                  f"ours {near_ties_ours}, reference {near_ties_ref}")


def lower_bound(field, box) -> None:
    print(f"2. lower bound on eps, tolerance delta = {DELTA:.0e}")
    nodes, _tets = build_tetrahedral_grid(RESOLUTION, box)
    with torch.no_grad():
        values = field.logits(nodes).detach()
    print("   clinical field")
    for eps in (1e-11, 1e-10, 1e-9):
        gaps = _tie_separations(values, eps)
        print(f"      eps {eps:.0e}: {int((gaps < DELTA).sum())} of {gaps.numel()} ties "
              f"separated by less than delta; {_closure(field, RESOLUTION, box, eps)}")
    print(f"      eps 1e-11 with delta lowered to 1e-13: "
          f"{_closure(field, RESOLUTION, box, 1e-11, tol=1e-13)}")

    res, cube = 48, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    grid, _ = build_tetrahedral_grid(res, cube)
    plane = torch.stack([grid[:, 0], torch.zeros(grid.shape[0])], dim=1)
    flat = compare_du.FixedValues(grid, plane)
    print(f"   two labels f1 = x1, f2 = 0 at resolution {res}, tied on the plane x1 = 0")
    for eps in (1e-11, 1e-10, 1e-9, 1e-8, 1e-7):
        gaps = _tie_separations(plane, eps)
        print(f"      eps {eps:.0e}: {int((gaps < DELTA).sum())} of {gaps.numel()} ties "
              f"separated by less than delta; {_closure(flat, res, cube, eps)}")


def _unpaired_ends(surf) -> Tensor:
    """Endpoints of the sides that `check_topology` reports as unpaired."""
    nv, nc = surf.vertices.shape[0], int(surf.node_labels.max()) + 1
    u = surf.triangles[:, [0, 1, 2]].reshape(-1)
    v = surf.triangles[:, [1, 2, 0]].reshape(-1)
    key = _edge_pair_keys(u, v, surf.triangle_labels.repeat_interleave(3, dim=0), nv, nc)
    _uniq, inverse, count = torch.unique(key, return_inverse=True, return_counts=True)
    allowed = [torch.empty(0, dtype=key.dtype)]
    if surf.triple_segments.numel():
        pairs = surf.triple_labels[:, torch.tensor(((0, 1), (0, 2), (1, 2)))].reshape(-1, 2)
        allowed.append(_edge_pair_keys(surf.triple_segments[:, 0].repeat_interleave(3),
                                       surf.triple_segments[:, 1].repeat_interleave(3),
                                       pairs, nv, nc))
    if surf.boundary_segments.numel():
        allowed.append(_edge_pair_keys(surf.boundary_segments[:, 0], surf.boundary_segments[:, 1],
                                       surf.boundary_segment_labels, nv, nc))
    single = (count[inverse] == 1) & ~torch.isin(key, torch.cat(allowed))
    return surf.vertices.detach()[torch.cat([u[single], v[single]])]


def triple_ties() -> None:
    cube = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    resolutions = range(36, 62, 2)
    print(f"3. three labels f1 = x1, f2 = 0, f3 = x2 at resolutions {resolutions.start}-"
          f"{resolutions.stop - 2} (even), two tied on the planes x1 = 0 and x2 = 0, "
          f"all three on the line x1 = x2 = 0")
    for eps in (1e-11, 1e-10, 1e-9, 1e-8):
        open_meshes, sides, line_nodes, farthest = 0, 0, 0, 0.0
        for res in resolutions:
            grid, _ = build_tetrahedral_grid(res, cube)
            values = torch.stack([grid[:, 0], torch.zeros(grid.shape[0]), grid[:, 1]], dim=1)
            line_nodes += int(((grid[:, 0] == 0) & (grid[:, 1] == 0)).sum())
            surf = envelope3d.extract(compare_du.FixedValues(grid, values), resolution=res,
                                      box=cube, perturb=eps)
            topo = extract3d.check_topology(surf)
            if not topo["ok"]:
                open_meshes += 1
                sides += topo["unpaired_sides"]
                ends = _unpaired_ends(surf)
                farthest = max(farthest, float(ends[:, :2].norm(dim=1).max()) * res / 2)
        where = f", all within {farthest:.2f} grid steps of the line" if sides else ""
        print(f"      eps {eps:.0e}: {open_meshes} of {len(resolutions)} meshes open, "
              f"{sides} unpaired sides{where} ({line_nodes} nodes with a three-way tie in all)")


def main() -> None:
    field, box, _block, _spacing = realdata.load_block(ORIGIN_VOX, WIDTH)
    clinical_against_reference(field, box)
    lower_bound(field, box)
    triple_ties()


if __name__ == "__main__":
    main()
