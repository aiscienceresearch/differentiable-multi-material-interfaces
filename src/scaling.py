"""Cost of the extraction against grid resolution and against K.

The paper's cost section claims the direct route is asymptotically the worse of
the two constructions and is affordable anyway, because the work per entity is
uniform enough to run as a fixed number of batched array operations. That is a
claim about how cost grows, so it needs measuring rather than asserting.

Two sweeps. The first holds the field fixed and refines the grid, so the
tetrahedron count grows as the cube of the resolution; if the batched
formulation behaves, wall time should track the tetrahedron count roughly
linearly rather than growing faster. The second holds the grid fixed and raises
K, which is where the binomial over candidate label subsets would show up if the
candidate sets grew with K -- the cost section argues they do not.

Everything is single-threaded CPU float64, which is what the rest of the paper
reports. Peak memory is the high-water mark of the process during the call,
sampled against the resident set before it. Each row runs in its own
subprocess, because neither Python nor torch returns freed pages to the
operating system: a row that followed a larger one in the same process would
be served out of the pages the larger one still holds, and its growth over the
resident set would read far below the memory it actually needs.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time

import psutil
import torch

import envelope3d
from fields import NeuralMultiLabelField, PowerDiagramField

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def _measure(field, resolution: int) -> dict:
    """Wall time and peak resident-set growth for one extraction.

    Reading the resident set once after the call misses the high-water mark,
    which is reached in the middle of the batched solves and released before
    they return, so a sampling thread watches it while the work runs.
    """
    proc = psutil.Process(os.getpid())
    gc.collect()
    before = proc.memory_info().rss

    peak = before
    stop = threading.Event()

    def watch():
        nonlocal peak
        while not stop.is_set():
            try:
                peak = max(peak, proc.memory_info().rss)
            except psutil.Error:
                return
            time.sleep(0.002)

    sampler = threading.Thread(target=watch, daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    surf = envelope3d.extract(field, resolution=resolution, box=BOX)
    elapsed = time.perf_counter() - t0
    stop.set()
    sampler.join()
    peak = max(peak, proc.memory_info().rss)

    diag = surf.diagnostics if hasattr(surf, "diagnostics") else {}
    out = {
        "resolution": resolution,
        "tets": 6 * resolution ** 3,
        "seconds": elapsed,
        "peak_mb": (peak - before) / 2 ** 20,
        "vertices": int(surf.vertices.shape[0]),
        "triangles": int(surf.triangles.shape[0]),
        "mean_labels_per_tet": float(diag.get("mean_labels_per_tet", float("nan"))),
    }
    del surf
    gc.collect()
    return out


def _power_diagram(k: int, seed: int) -> PowerDiagramField:
    """The field of one row, built from its own seed so a subprocess can repeat it."""
    torch.manual_seed(seed)
    sites = (torch.rand((k, 3), dtype=torch.float64) - 0.5) * 1.2
    return PowerDiagramField(sites)


def measure_row(kind: str, k: int, seed: int, resolution: int) -> dict:
    """One row of either sweep, in whatever process calls this."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if kind == "neural":
        field = NeuralMultiLabelField(num_classes=k, hidden=48, dim=3,
                                      seed=seed, box=BOX)
    else:
        field = _power_diagram(k, seed)
    return _measure(field, resolution)


def _row_in_subprocess(kind: str, k: int, seed: int, resolution: int) -> dict:
    """Measure one row in a fresh interpreter, so its baseline is its own."""
    spec = json.dumps({"kind": kind, "k": k, "seed": seed, "resolution": resolution})
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--row", spec],
                         capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.abspath(__file__)))
    return json.loads(out.stdout.strip().splitlines()[-1])


def sweep_resolution(resolutions=(16, 24, 32, 48, 64, 80)) -> list[dict]:
    print("\nA. Refining the grid, K=6 power diagram")
    print(f"   {'res':>4} {'tets':>10} {'verts':>9} {'tris':>9} "
          f"{'seconds':>9} {'peak MB':>9} {'us/tet':>8}")
    rows = []
    for r in resolutions:
        row = _row_in_subprocess("power", 6, 0, r)
        rows.append(row)
        print(f"   {row['resolution']:>4} {row['tets']:>10,} {row['vertices']:>9,} "
              f"{row['triangles']:>9,} {row['seconds']:>9.2f} {row['peak_mb']:>9.0f} "
              f"{1e6 * row['seconds'] / row['tets']:>8.2f}")
    return rows


def sweep_classes(ks=(3, 4, 6, 9, 12, 16), resolution: int = 32) -> list[dict]:
    print(f"\nB. Raising K at resolution {resolution}")
    print(f"   {'K':>3} {'verts':>9} {'tris':>9} {'seconds':>9} {'peak MB':>9} "
          f"{'labels/tet':>11}")
    rows = []
    for k in ks:
        row = _row_in_subprocess("power", k, k, resolution)
        row["K"] = k
        rows.append(row)
        print(f"   {k:>3} {row['vertices']:>9,} {row['triangles']:>9,} "
              f"{row['seconds']:>9.2f} {row['peak_mb']:>9.0f} "
              f"{row['mean_labels_per_tet']:>11.2f}")
    return rows


def _fit_exponent(rows, xkey="tets") -> float:
    """Slope of log(time) against log(x): 1.0 is linear in the tet count."""
    xs = torch.tensor([float(r[xkey]) for r in rows]).log()
    ys = torch.tensor([r["seconds"] for r in rows]).log()
    xm, ym = xs.mean(), ys.mean()
    return float(((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum())


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    print("=" * 74)
    print("Extraction cost. Single-threaded CPU, float64.")
    print("=" * 74)

    res_rows = sweep_resolution()
    print(f"\n   time grows as (tetrahedron count)^{_fit_exponent(res_rows):.2f}")

    cls_rows = sweep_classes()
    print(f"\n   time grows as K^{_fit_exponent(cls_rows, 'K'):.2f}")

    # The neural field is the non-analytic case, checked separately because the
    # field evaluation itself is a cost the power diagrams do not have.
    print("\nC. Neural field, K=6, for a non-analytic input")
    for r in (24, 32, 48):
        row = _row_in_subprocess("neural", 6, 100, r)
        print(f"   res {row['resolution']:>3}  {row['tets']:>9,} tets  "
              f"{row['seconds']:>7.2f} s  {row['peak_mb']:>6.0f} MB")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    if "--row" in sys.argv:
        print(json.dumps(measure_row(**json.loads(sys.argv[sys.argv.index("--row") + 1]))))
    else:
        main()
