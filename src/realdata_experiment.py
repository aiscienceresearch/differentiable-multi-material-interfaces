"""The real-data experiment: run both methods on one clinical segmentation field.

Reports, for a junction-rich region of TotalSegmentator's Couinaud liver-segment
output on a thoracoabdominal CT:

  * that the extracted complex is watertight and coherently oriented;
  * how far each method's vertices sit from the field's own interface, measured
    as the gap between the two leading channels of the interpolant, which is
    zero exactly on an interface;
  * how far SurfaceNets' duplicated non-manifold points separate under its
    smoothing pass, against which the same quantity for us is identically zero
    because the three patches of a triple curve share one vertex index.

The second measurement needs a word of fairness. SurfaceNets consumes the
integer label map, not the probability field, so it cannot place a vertex on
the field's interface even in principle. That is the point being measured --
what thresholding discards -- and not a defect of its implementation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import baseline_surfacenets as bl
import envelope3d
import extract3d
import realdata
from extract3d import CROSSING, QUADRUPLE, TRIPLE

ORIGIN_VOX = (29, 118, 165)   # junction-richest 49^3 window; see notes in report()
WIDTH = 49
RESOLUTION = WIDTH - 1
PERTURB = 1e-9
OUT = Path(__file__).resolve().parent.parent / "data" / "realdata_results.json"


def run():
    torch.set_default_dtype(torch.float64)
    field, box, block, spacing = realdata.load_block(ORIGIN_VOX, WIDTH)

    alignment = realdata.check_nodes_are_voxels(field, box, RESOLUTION, block)

    start = time.time()
    surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=PERTURB)
    elapsed = time.time() - start

    topology = extract3d.check_topology(surf)

    kinds = {name: int((surf.vertex_kind == value).sum())
             for name, value in (("crossing", CROSSING), ("triple", TRIPLE),
                                 ("quadruple", QUADRUPLE))}
    patches = sorted({tuple(p) for p in surf.triangle_labels.tolist()})

    # Interface accuracy, both methods against the same interpolant, in mm.
    labels = block.argmax(0).astype(np.int16)
    divergence = bl.duplicated_point_divergence(labels, spacing)
    _, sn_points, _, _ = bl.surface_and_duplicates(labels, spacing)

    def distance(points):
        d = realdata.interface_distance(field, surf, box, RESOLUTION,
                                        points, PERTURB).abs()
        d = d[d.isfinite()]
        return {"median_mm": float(d.median()),
                "p90_mm": float(d.quantile(0.9)),
                "p99_mm": float(d.quantile(0.99)),
                "max_mm": float(d.max())}

    ours_dist = distance(surf.vertices.detach())
    sn_dist = distance(torch.as_tensor(sn_points))

    results = {
        "volume": {
            "origin_voxel": list(ORIGIN_VOX),
            "width": WIDTH,
            "spacing_mm": [round(float(s), 4) for s in spacing],
            "box_mm": [round(float(b), 3) for b in box],
            "labels_present": sorted(int(v) for v in np.unique(labels)),
        },
        "grid": {
            "resolution": RESOLUTION,
            "tetrahedra": 6 * RESOLUTION ** 3,
            "node_voxel_alignment_error": alignment,
            "extract_seconds": round(elapsed, 2),
        },
        "ours": {
            "vertices": int(surf.vertices.shape[0]),
            "triangles": int(surf.triangles.shape[0]),
            "vertex_kinds": kinds,
            "patches": len(patches),
            "patch_label_pairs": [list(p) for p in patches],
            "triple_segments": int(surf.triple_segments.shape[0]),
            "max_overshoot": float(surf.vertex_overshoot.max()),
            "max_residual": float(surf.vertex_residual.max()),
            "interface_distance": ours_dist,
            "junction_vertex_divergence_mm": 0.0,
        },
        "topology": {k: (bool(v) if isinstance(v, bool) else int(v))
                     for k, v in topology.items()},
        "surfacenets": {**divergence, "interface_distance": sn_dist},
    }
    OUT.write_text(json.dumps(results, indent=2))
    return results


def report(r: dict) -> None:
    v, g, o, t, s = (r["volume"], r["grid"], r["ours"], r["topology"], r["surfacenets"])
    vx = s["voxel_mm"]

    print("Region: %d^3 voxels at %s, spacing %s mm" %
          (v["width"], tuple(v["origin_voxel"]), v["spacing_mm"]))
    print("Labels present: %s" % v["labels_present"])
    print("Grid: %d tetrahedra; node/voxel alignment error %.1e"
          % (g["tetrahedra"], g["node_voxel_alignment_error"]))
    print()
    print("Ours: %d vertices, %d triangles, %d patches, %d triple segments in %.1fs"
          % (o["vertices"], o["triangles"], o["patches"], o["triple_segments"],
             g["extract_seconds"]))
    print("  vertex kinds: %s" % o["vertex_kinds"])
    print("  watertight and coherently oriented: %s "
          "(unpaired %d, over-used %d, misoriented %d, incomplete triple curves %d)"
          % (t["ok"], t["unpaired_sides"], t["sides_used_more_than_twice"],
             t["inconsistently_oriented_sides"], t["missing_patches_on_triple_curves"]))
    print("  overshoot %.1e, residual %.1e" % (o["max_overshoot"], o["max_residual"]))
    print("  distance to the field's interface: median %.1e mm, p99 %.1e mm"
          % (o["interface_distance"]["median_mm"], o["interface_distance"]["p99_mm"]))
    print()
    print("vtkSurfaceNets3D: %d points, %d quads" % (s["points"], s["quads"]))
    print("  duplicated non-manifold points: %d in %d groups"
          % (s["duplicated_points"], s["duplicated_groups"]))
    print("  separation after smoothing: max %.3f mm (%.2f voxels), mean %.3f mm (%.2f voxels)"
          % (s["max_divergence_mm"], s["max_divergence_mm"] / vx,
             s["mean_divergence_mm"], s["mean_divergence_mm"] / vx))
    d = s["interface_distance"]
    print("  distance to the field's interface: median %.3f mm (%.2f voxels), "
          "p90 %.3f, p99 %.3f" % (d["median_mm"], d["median_mm"] / vx,
                                  d["p90_mm"], d["p99_mm"]))
    print("  (the linearised distance degenerates where the gap's gradient is flat,")
    print("   so the maximum of %.1f mm is a metric artefact, not a mesh error)"
          % d["max_mm"])


if __name__ == "__main__":
    report(run())
