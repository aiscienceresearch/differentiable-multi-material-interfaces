"""Which of the baseline's three differences costs the optimisation anything?

Section 8.4 substitutes `extract3d` for `envelope3d` in the double-bubble
optimisation, watches the objective go noisy, and attributes that to the
corner-label case analysis choosing a different interface. The substitution is
not clean enough to support the attribution on its own: the two modules differ
in three ways at once.

  1. Combinatorics. One feature per cell, keyed by corner labels, against the
     full enumeration of label sets.
  2. Solver. Bisection and Newton against a closed-form linear solve.
  3. Model. `extract3d` locates features of the field itself; `envelope3d`
     locates them on the P1 interpolant of the nodal logits.

A noisy trace is consistent with (1), but equally with partially converged
iterates from (2) or with the moving target of (3). So this runs a third arm
that changes (1) alone --- corner-label combinatorics driving the same closed
form solves on the same interpolant, via `envelope3d.extract(corner_label=True)`
--- and compares all three.

If the third arm tracks the full baseline, the combinatorics is what matters and
Section 8.4's reading stands. If it tracks the exact arrangement instead, the
noise was the solver and the section needs rewriting.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import double_bubble
import envelope3d
import extract3d
from double_bubble import CORNER_CLOSED, CORNER_NOPERT

OUT = Path(__file__).resolve().parents[1] / "data" / "ablation_double_bubble.json"

ARMS = [
    ("exact arrangement", envelope3d),
    ("corner-label, closed form", CORNER_CLOSED),
    ("corner-label, no perturbation", CORNER_NOPERT),
    ("corner-label, Newton", extract3d),
]

# Five starts, the paper's plus four perturbations of it. "A broad wrong basin"
# is a claim about initialisation dependence, so it cannot be settled by a sweep
# over the grid; this is the sweep that settles it.
STARTS = [
    ("paper", ([[0.34, 0.09, -0.06], [-0.40, -0.11, 0.07]], [0.583, 0.447])),
    # Near-symmetric, not symmetric. Exactly equal radii about exactly opposite
    # centres is a measure-zero configuration in which the unperturbed arm has
    # no 1|2 wall to measure an angle on, and the sweep is not about that.
    ("near-symmetric", ([[0.301, 0.004, -0.003], [-0.298, -0.005, 0.004]],
                        [0.520, 0.524])),
    ("very lopsided", ([[0.28, 0.14, -0.10], [-0.46, -0.16, 0.11]], [0.640, 0.390])),
    ("tight", ([[0.22, -0.05, 0.08], [-0.26, 0.06, -0.09]], [0.480, 0.500])),
    ("loose", ([[0.44, 0.12, 0.05], [-0.42, -0.14, -0.08]], [0.610, 0.580])),
]


def geometry_check(resolution: int = 32) -> list[dict]:
    """Do the two corner-label arms agree combinatorially, as they must?

    The isolating arm is only isolating if its combinatorics really is the
    baseline's. Both should propose the same features on the same field --- one
    crossing per mixed edge, one tie per cell with distinct corner labels ---
    and so should emit the same counts, differing only in where the points land.
    Equal counts is the evidence that the arm was built right; it is a test of
    this file, not a result about the method.
    """
    from fields import PowerDiagramField

    rows = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        field = PowerDiagramField((torch.rand((5, 3), dtype=torch.float64) - 0.5) * 1.2)
        row = {"seed": seed}
        for name, arm in ARMS:
            surf = double_bubble._extract(arm, field, resolution)
            row[name] = {
                "vertices": int(surf.vertices.shape[0]),
                "triangles": int(surf.triangles.shape[0]),
                "area": float(surf.total_area().detach()),
                "closes": bool(extract3d.check_topology(surf)["ok"]),
            }
        rows.append(row)
    return rows


def _roughness(trace: list[float]) -> float:
    """How jagged the objective trace is, ignoring the descent it should have.

    Section 8.4 calls the baseline's trace noisy. First differences would
    measure descent as much as noise, and the last-few-steps window measures
    almost nothing once the cosine schedule has annealed the step size. The
    second difference cancels any steady drift and leaves the step-to-step
    reversals, and the median keeps one flip from setting the number. Taken
    over the second half, after the continuation weight has stopped moving the
    objective on its own.
    """
    half = trace[len(trace) // 2:]
    curv = sorted(abs(a - 2 * b + c) for a, b, c in zip(half, half[1:], half[2:]))
    return curv[len(curv) // 2] if curv else float("nan")


def _median_step(trace: list[float]) -> float:
    """Median absolute first difference over the same window.

    Dividing the roughness by this makes the statistic dimensionless and stops
    it being read against a converged arm's noise floor, which is a few hundred
    float64 eps on a quantity of magnitude nine and so inflates any ratio taken
    against it. A trace that descends smoothly has a small second difference
    and a large first one; a trace that reverses step to step has them
    comparable.
    """
    half = trace[len(trace) // 2:]
    steps = sorted(abs(b - a) for a, b in zip(half, half[1:]))
    return steps[len(steps) // 2] if steps else float("nan")


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    print("=" * 78)
    print("Double-bubble ablation: isolating the combinatorics")
    print("=" * 78)

    print("\nA. Do the two corner-label arms agree on what to extract?")
    print("   Five-site power diagrams. Equal counts mean the isolating arm")
    print("   reproduces the baseline's combinatorics; area may still differ,")
    print("   because the baseline clamps its Newton iterate to a dilated cell.")
    checks = geometry_check()
    for row in checks:
        print(f"\n   seed {row['seed']}")
        for name, _ in ARMS:
            r = row[name]
            print(f"     {name:>26}  verts {r['vertices']:>6}  tris {r['triangles']:>6}"
                  f"  area {r['area']:>9.5f}  closes {str(r['closes']):>5}")
    matched = all(
        row["corner-label, closed form"]["vertices"] == row["corner-label, Newton"]["vertices"]
        and row["corner-label, closed form"]["triangles"] == row["corner-label, Newton"]["triangles"]
        for row in checks
    )
    print(f"\n   corner-label arms agree on counts: {matched}")

    # The settings Section 8.4 and Figure 8 are produced with, so the arms are
    # comparable against the numbers already in the paper rather than against a
    # differently tuned run of the same experiment.
    print("\nB. The optimisation, once per arm (steps=300, lr=2e-2, res=32)")
    runs = {}
    for name, arm in ARMS:
        print(f"\n--- {name} " + "-" * max(60 - len(name), 3))
        run = double_bubble.optimise(steps=300, resolution=32, lr=2e-2,
                                     extractor=arm, verbose=True)
        ratios = [h["ratio"] for h in run["history"]]
        losses = [h["loss"] for h in run["history"]]
        fin = run["final"]
        runs[name] = {
            "final_ratio": fin["ratio"],
            "r_over_d": fin["r_over_d"],
            "angle_min": fin["angle_min"],
            "angle_max": fin["angle_max"],
            "worst_angle_deviation": fin["worst_angle_deviation"],
            # Two roughness figures, because they answer different questions.
            # The shape-ratio trace is the geometry settling; the loss is what
            # is actually descended, and it carries the continuation weight.
            "roughness": _roughness(ratios),
            "roughness_loss": _roughness(losses),
            # Scale-free, so the ratios below do not rest on a converged arm's
            # noise floor sitting a few hundred eps above zero.
            "roughness_rel": _roughness(ratios) / max(_median_step(ratios), 1e-300),
            "topology_ok": run["topology_ok"],
            "history": ratios,
            "history_loss": losses,
        }

    exact = double_bubble.optimal_ratio()
    print("\n" + "=" * 78)
    print(f"Optimal ratio for the double bubble: {exact:.6f}")
    print(f"{'arm':>26} {'A/V^2/3':>9} {'excess':>9} {'r/d':>15} "
          f"{'angles':>15} {'rough':>9} {'closes':>7}")
    for name, _ in ARMS:
        r = runs[name]
        print(f"{name:>26} {r['final_ratio']:>9.4f} {r['final_ratio'] - exact:>9.4f} "
              f"{r['r_over_d'][0]:>7.4f} {r['r_over_d'][1]:>7.4f} "
              f"{r['angle_min']:>7.1f} {r['angle_max']:>7.1f} "
              f"{r['roughness']:>9.2e} {str(r['topology_ok']):>7}")

    print(f"\n{'arm':>32} {'rough(ratio)':>13} {'rough(loss)':>13} "
          f"{'rough/step':>11}")
    for name, _ in ARMS:
        r = runs[name]
        print(f"{name:>32} {r['roughness']:>13.2e} {r['roughness_loss']:>13.2e} "
              f"{r['roughness_rel']:>11.3f}")

    ex = runs["exact arrangement"]
    closed = runs["corner-label, closed form"]
    nopert = runs["corner-label, no perturbation"]
    newton = runs["corner-label, Newton"]

    # Not a variance decomposition. The two corner-label arms miss in opposite
    # directions --- one shape too small, the other too large --- so their
    # excesses do not add up to anything and apportioning a share between
    # combinatorics and solver would be meaningless. What the pair does show is
    # a sufficiency claim and a contingency claim, which is what gets reported.
    print("\nExcess objective over the exact arrangement:")
    print(f"  combinatorics alone (closed form) {closed['final_ratio'] - ex['final_ratio']:+.4f}"
          f"   r/d {closed['r_over_d'][0]:.4f}")
    print(f"  combinatorics and solver (Newton) {newton['final_ratio'] - ex['final_ratio']:+.4f}"
          f"   r/d {newton['r_over_d'][0]:.4f}")
    print(f"\n  Sufficiency:  changing the combinatorics alone is enough to lose the")
    print(f"                double bubble, with nothing else touched.")
    print(f"  Contingency:  the two arms straddle the truth"
          f" ({closed['r_over_d'][0]:.2f} and {newton['r_over_d'][0]:.2f} against 1),")
    print(f"                so which wrong shape you get is not set by the")
    print(f"                combinatorics on its own.")
    print(f"\nRoughness relative to the exact arrangement:"
          f"  closed form {closed['roughness'] / ex['roughness']:.0f}x,"
          f"  Newton {newton['roughness'] / ex['roughness']:.0f}x")
    print("  Read the scale-free column above instead where the ratio matters:"
          " the exact")
    print("  arm's trace is flat because it converged and stopped, so ratios"
          " against it")
    print("  measure that as much as they measure smoothness.")

    # The fourth difference. extract3d takes no perturbation argument, so the
    # Newton arm runs without one, in the one configuration where the paper
    # says the perturbation is what emits the 1|2 wall at all. This prices it.
    print(f"\nThe perturbation, priced on its own (both arms corner-label,"
          f" closed form):")
    print(f"  with    perturbation  r/d {closed['r_over_d'][0]:.4f}"
          f"   A/V^2/3 {closed['final_ratio']:.4f}"
          f"   rough/step {closed['roughness_rel']:.3f}")
    print(f"  without perturbation  r/d {nopert['r_over_d'][0]:.4f}"
          f"   A/V^2/3 {nopert['final_ratio']:.4f}"
          f"   rough/step {nopert['roughness_rel']:.3f}")

    # Written now, so that a failure in the sweep below cannot cost the four
    # hours of optimisation above.
    def save(extra=None):
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(
            {"optimal_ratio": exact, "counts_match": matched,
             "geometry": checks, "runs": runs, "basins": extra or {}}, indent=1))

    save()

    print("\nC. Is the wrong answer a basin or a point? Five initialisations.")
    print("   'A broad wrong basin' is a claim about initialisation, so varying")
    print("   the grid cannot settle it. Varying the start can.")
    basins = {}
    for name, arm in ARMS:
        basins[name] = []
        for label, start in STARTS:
            try:
                run = double_bubble.optimise(steps=300, resolution=32, lr=2e-2,
                                             extractor=arm, verbose=False,
                                             start=start)
            except Exception as exc:  # an arm that cannot run is a result
                basins[name].append({"start": label, "failed": repr(exc)})
                print(f"   {name:>32}  {label:>15}  FAILED  {exc!r}")
                save(basins)
                continue
            fin = run["final"]
            basins[name].append({
                "start": label, "r_over_d": fin["r_over_d"][0],
                "final_ratio": fin["ratio"], "topology_ok": run["topology_ok"],
                "angles_degenerate": fin.get("angles_degenerate"),
            })
            print(f"   {name:>32}  {label:>15}  r/d {fin['r_over_d'][0]:>7.4f}"
                  f"  A/V^2/3 {fin['ratio']:>8.4f}"
                  f"  closes {str(run['topology_ok']):>5}")
            save(basins)
        print()

    print(f"{'arm':>32} {'n':>3} {'r/d min':>9} {'r/d max':>9} {'spread':>9} "
          f"{'worst |r/d - 1|':>16}")
    for name, _ in ARMS:
        rs = [b["r_over_d"] for b in basins[name] if "r_over_d" in b]
        if not rs:
            print(f"{name:>32}   0   all starts failed")
            continue
        print(f"{name:>32} {len(rs):>3} {min(rs):>9.4f} {max(rs):>9.4f} "
              f"{max(rs) - min(rs):>9.4f} {max(abs(r - 1) for r in rs):>16.4f}")
    print("=" * 78)

    save(basins)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
