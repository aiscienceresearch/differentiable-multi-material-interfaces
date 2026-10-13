"""What the symbolic perturbation costs on the clinical field, in millimetres.

The clinical field is the one input in the paper with exact coincidences, so it
is where the perturbation does work. Two measurements:

  1. Our extractor at eps = 0 and at the default eps, scored against the
     interface of the *unperturbed* P1 interpolant, which is the partition the
     perturbation is meant to leave unchanged.
  2. The reference implementation of Du et al. on the field as given, against
     our extractor with the perturbation on at three sizes: how far each of
     their vertices is from the nearest of ours.

Run with `python perturbation_offsets.py > ../data/perturbation_offsets_run.txt`.
Part 2 requires the Docker image of `compare_du.py`.
"""

from __future__ import annotations

import shutil
import tempfile

import torch

import compare_du
import envelope3d
import extract3d
import realdata
from extract3d import build_tetrahedral_grid
from realdata_experiment import ORIGIN_VOX, PERTURB, RESOLUTION, WIDTH

torch.set_default_dtype(torch.float64)


def _one_way(p: torch.Tensor, q: torch.Tensor, chunk: int = 512) -> float:
    worst = 0.0
    for i in range(0, p.shape[0], chunk):
        d = torch.cdist(p[i:i + chunk], q, compute_mode="donot_use_mm_for_euclid_dist")
        worst = max(worst, float(d.min(dim=1).values.max()))
    return worst


def ours_against_unperturbed(field, box) -> None:
    print("1. our vertices against the interface of the unperturbed interpolant")
    for eps in (0.0, PERTURB):
        surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=eps)
        topo = extract3d.check_topology(surf)
        d = realdata.interface_distance(field, surf, box, RESOLUTION,
                                        surf.vertices.detach(), 0.0).abs()
        d = d[d.isfinite()]
        print(f"   eps {eps:.0e}: {surf.vertices.shape[0]} vertices, closed {topo['ok']} "
              f"({topo['unpaired_sides']} unpaired sides), "
              f"patches with too few corners {surf.diagnostics['patches_with_too_few_corners']}")
        print(f"      distance median {float(d.median()):.2e} mm, max {float(d.max()):.2e} mm")


def reference_against_perturbed(field, box) -> None:
    print("2. reference vertices against ours, field as given")
    nodes, tets = build_tetrahedral_grid(RESOLUTION, box)
    with torch.no_grad():
        values = field.logits(nodes).detach()
    top2 = values.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    print(f"   {compare_du.count_exact_ties(values)} of {values.shape[0]} nodes carry an "
          f"exact argmax tie; smallest nonzero margin {float(margin[margin > 0].min()):.1e}")
    workdir = tempfile.mkdtemp(prefix="du_offsets_")
    try:
        theirs, _mesh, their_points = compare_du.run_reference(
            None, nodes, tets, workdir, values=values)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"   reference: {their_points.shape[0]} vertices")
    for eps in (1e-11, 1e-10, 1e-9):
        surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=eps)
        ours = surf.vertices.detach()
        print(f"   eps {eps:.0e}: ours {ours.shape[0]} vertices; every reference vertex "
              f"within {_one_way(their_points, ours):.1e} of one of ours")


def main() -> None:
    field, box, _block, _spacing = realdata.load_block(ORIGIN_VOX, WIDTH)
    ours_against_unperturbed(field, box)
    reference_against_perturbed(field, box)


if __name__ == "__main__":
    main()
