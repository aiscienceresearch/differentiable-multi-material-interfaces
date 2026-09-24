"""The 3D optimization demo: an equal-volume double bubble, found by descent.

This is the 3D counterpart of the equal-area disk in `demo_optimize.py`, and it
exists to close the one gap that matters for the differentiability claim. In 3D
we show elsewhere that the geometry is exact and that the extractor is
differentiable in principle; here the loss is actually minimised through a
triple curve, and the answer is known in closed form.

The objective is total interface area at prescribed equal volumes. By the
double bubble theorem the minimiser is the standard double bubble: two
spherical caps of equal radius $R$ whose centres are $R$ apart, separated by
the flat disk their bisector cuts, with the three surfaces meeting at $120$
degrees along a circle. `BubbleField` contains that configuration exactly, so
a failure to reach it is a failure of the gradients rather than of the
parametrization.

The objective is posed in scale-free form, and that is not cosmetic. The
extracted volume of a curved region carries an $O(h^2)$ bias --- at resolution
$24$ it sits $1.5\\%$ below the closed-form value --- so a penalty that drives
the extracted volume to the analytic target asks the optimizer to make up the
difference by inflating the bubble. Adam moves every parameter at roughly the
same rate, so it inflates the separation as fast as the radii, and the lobes
come apart onto the two-separate-balls configuration. Minimising

    A / V^(2/3)  +  balance penalty  +  a weak anchor on the scale

removes the coupling: discretisation bias can only move the scale, which the
anchor pins and which nothing else depends on, while the shape is set by a
term that does not know how big the bubble is. The balance penalty is not
optional --- $A/V^{2/3}$ alone is minimised by collapsing one lobe, which
gives $7.6766$ against the double bubble's $9.1394$.

What is reported:

  * $A / V^{2/3}$, the objective made dimensionless. The analytic minimum is
    $9.1394$; two separate balls of the same volumes give $9.6720$.
  * $r_1 / d$ and $r_2 / d$, both exactly $1$ at the minimiser. By
    Proposition 6 the three interface normals have lengths $2r_1$, $2r_2$,
    $2d$, so these two ratios being one *is* the statement that the Neumann
    triangle is equilateral and the dihedral angles are $120$ degrees.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch

import envelope3d
import extract3d
from fields import BubbleField, double_bubble_reference

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
PERTURB = 1e-9
CACHE = Path(__file__).resolve().parent.parent / "data" / "double_bubble.pkl"
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# A perfectly symmetric double bubble makes f_1 = f_2 identically on the
# separating plane, which is a genuine coincidence rather than a near miss:
# without the symbolic perturbation the 1|2 wall is not emitted at all.
assert PERTURB > 0.0


def dihedral_angles(field, surface) -> list[float]:
    """Angles between the three patches along the triple curve, in degrees.

    `extract3d.triple_curve_dihedrals` decides which side of the curve each
    patch lies on by probing $10^{-5}$ away from it, which is sound when the
    extracted curve coincides with the true one and not otherwise; on curved
    interfaces the extracted curve sits $O(h^2)$ off and the probe lands in the
    wrong region. Proposition 6 avoids the question: the three in-plane normals
    sum to zero, so they close a triangle whose angles follow from their
    lengths alone. The two agree to $3\\times10^{-14}$ on fields where the
    probe is reliable.
    """
    if surface.triple_segments.numel() == 0:
        return []
    verts = surface.vertices.detach()
    mid = verts[surface.triple_segments].mean(dim=1)
    axis = verts[surface.triple_segments[:, 1]] - verts[surface.triple_segments[:, 0]]
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    grads = field.logit_grads(mid)

    out: list[float] = []
    for n in range(mid.shape[0]):
        i, j, k = surface.triple_labels[n].tolist()
        lengths = []
        for p, q in ((i, j), (j, k), (k, i)):
            v = grads[n, p] - grads[n, q]
            v = v - (v @ axis[n]) * axis[n]
            lengths.append(float(v.norm()))
        a, b, c = lengths
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            # Two of the three logits can have equal gradients at a segment of
            # a degenerate configuration, leaving no angle to measure there.
            # That is a fact about the configuration and should be reported as
            # a missing measurement, not raised: the ablation sweeps starts
            # precisely to find arms that degenerate, and one of them taking
            # the whole sweep down with it would defeat the point.
            if x * y < 1e-30:
                out.append(float("nan"))
                continue
            cos = (z * z - x * x - y * y) / (2.0 * x * y)
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
    return out


def separate_balls_ratio() -> float:
    """A / V^(2/3) for two disjoint balls of equal volume: the configuration
    the optimizer has to beat, and the one it falls back to if the two lobes
    ever come apart."""
    return 8.0 * math.pi / (4.0 * math.pi / 3.0) ** (2.0 / 3.0)


def optimal_ratio() -> float:
    """A / V^(2/3) for the standard double bubble, V being one lobe."""
    return (27.0 * math.pi / 4.0) / (9.0 * math.pi / 8.0) ** (2.0 / 3.0)


def measure(field, surface) -> dict:
    area = float(surface.total_area().detach())
    vols = [float(surface.enclosed_volume(k).detach()) for k in (1, 2)]
    mean_v = sum(vols) / len(vols)
    radii = [float(v) for v in field.radii().detach()]
    d = float(field.separation().detach())
    ang = dihedral_angles(field, surface)
    # Segments with no measurable angle are dropped rather than allowed to
    # poison min and max, and counted so that a mostly-degenerate surface
    # cannot report a clean angle range from its few surviving segments.
    good = [a for a in ang if a == a]
    return {
        "area": area,
        "volumes": vols,
        "ratio": area / mean_v ** (2.0 / 3.0),
        "r_over_d": [r / d for r in radii],
        "radii": radii,
        "separation": d,
        "volume_imbalance": abs(vols[0] - vols[1]) / mean_v,
        "angle_min": min(good) if good else float("nan"),
        "angle_max": max(good) if good else float("nan"),
        "angles_measured": len(good),
        "angles_degenerate": len(ang) - len(good),
        "worst_angle_deviation": (max(abs(a - 120.0) for a in good)
                                  if good else float("nan")),
    }


def scale_free_loss(surface, target: float, balance_weight: float,
                    scale_weight: float = 5.0):
    """A / V^(2/3), plus a penalty on the imbalance and an anchor on the scale."""
    area = surface.total_area()
    vols = torch.stack([surface.enclosed_volume(k) for k in (1, 2)])
    mean = vols.mean()
    ratio = area / mean.pow(2.0 / 3.0)
    balance = ((vols[0] - vols[1]) / mean).pow(2)
    scale = (mean / target - 1.0).pow(2)
    return ratio + balance_weight * balance + scale_weight * scale, ratio, vols


# A third arm for the ablation. `extract3d` differs from `envelope3d` in three
# ways at once --- corner-label combinatorics, Newton solves, and solving
# against the field rather than its interpolant --- so on its own it cannot say
# which of them costs the optimisation anything. This arm changes only the
# first, and is otherwise the operator used everywhere else in the paper.
CORNER_CLOSED = "corner-label combinatorics, closed-form solves"

# `extract3d` takes no perturbation argument, so the Newton baseline runs
# without one --- a fourth difference, and not an idle one here, since at the
# minimiser f1 == f2 on the separating plane and an unperturbed extractor need
# not emit the 1|2 wall at all. This arm is CORNER_CLOSED with the perturbation
# switched off, so that the fourth difference can be priced on its own.
CORNER_NOPERT = "corner-label combinatorics, closed form, no perturbation"


def _extract(extractor, field, resolution):
    """`extract3d` is the corner-label case analysis and `envelope3d` the exact
    arrangement; they take the same arguments apart from the perturbation,
    which only the latter needs. `CORNER_CLOSED` is the isolating arm."""
    if extractor is envelope3d:
        return extractor.extract(field, resolution=resolution, box=BOX, perturb=PERTURB)
    if isinstance(extractor, str) and extractor == CORNER_CLOSED:
        return envelope3d.extract(field, resolution=resolution, box=BOX,
                                  perturb=PERTURB, corner_label=True)
    if isinstance(extractor, str) and extractor == CORNER_NOPERT:
        return envelope3d.extract(field, resolution=resolution, box=BOX,
                                  perturb=0.0, corner_label=True)
    return extractor.extract(field, resolution=resolution, box=BOX)


def optimise(steps: int = 220, resolution: int = 32, radius: float = 0.55,
             lr: float = 1e-2, weight_range: tuple[float, float] = (10.0, 300.0),
             scale_weight: float = 20.0, extractor=envelope3d, verbose: bool = True,
             start=None):
    ref = double_bubble_reference(radius)
    target = ref["volume_each"]

    # Lopsided and off-axis, but overlapping: two disjoint balls have no
    # shared wall, hence no triple curve and no gradient pulling them together,
    # so the optimizer would have nothing to work with.
    #
    # `start` overrides it, so the ablation can ask whether a wrong answer is a
    # basin or a single point. Every figure in the paper uses the default.
    centres, radii = start or ([[0.34, 0.09, -0.06], [-0.40, -0.11, 0.07]],
                               [0.583, 0.447])
    field = BubbleField(centres=centres, radii=radii)

    init_surf = _extract(extractor, field, resolution)
    init = measure(field, init_surf)

    opt = torch.optim.Adam(field.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    w0, w1 = weight_range
    history = []

    for step in range(steps):
        opt.zero_grad()
        # Continuation on the balance penalty, as in the 2D studies: a weight
        # strong enough to pin the volumes from the start swamps the shape term
        # and freezes the geometry before it has found the right one.
        weight = w0 * (w1 / w0) ** (step / max(steps - 1, 1))
        surf = _extract(extractor, field, resolution)
        loss, ratio, vols = scale_free_loss(surf, target, weight, scale_weight)
        loss.backward()
        opt.step()
        sched.step()

        vd = vols.detach()
        history.append({
            "ratio": float(ratio.detach()),
            # The quantity actually being descended, as distinct from the shape
            # ratio above. The ablation measures roughness on both, since the
            # continuation weight moves this one on its own.
            "loss": float(loss.detach()),
            "imbalance": float(((vd[0] - vd[1]) / vd.mean()).abs()),
            "mean_volume": float(vd.mean()),
            "r_over_d": [float(r) / float(field.separation().detach())
                         for r in field.radii().detach()],
        })
        if verbose and (step % 20 == 0 or step == steps - 1):
            h = history[-1]
            print("  step %4d  A/V^(2/3) %.4f  r/d %.3f %.3f  imbalance %.2e  V %.4f"
                  % (step, h["ratio"], h["r_over_d"][0], h["r_over_d"][1],
                     h["imbalance"], h["mean_volume"]))

    final_surf = _extract(extractor, field, resolution)
    final = measure(field, final_surf)
    return {
        "field": field,
        "reference": ref,
        "initial": init,
        "final": final,
        "history": history,
        "init_surface": init_surf,
        "final_surface": final_surf,
        "resolution": resolution,
        "topology_ok": bool(extract3d.check_topology(final_surf)["ok"]),
    }


def report(run: dict) -> None:
    ref, init, fin = run["reference"], run["initial"], run["final"]
    exact, balls = optimal_ratio(), separate_balls_ratio()

    print("\n" + "=" * 78)
    print("Equal-volume double bubble, minimised through the triple curve")
    print("=" * 78)
    print("target volume per lobe %.6f, analytic radius %.4f, analytic area %.6f"
          % (ref["volume_each"], ref["radius"], ref["total_area"]))
    print("grid %d^3, watertight at the end: %s" % (run["resolution"], run["topology_ok"]))
    print()
    print("%-34s %12s %12s %12s" % ("", "initial", "final", "exact"))
    print("%-34s %12.4f %12.4f %12.4f" % ("A / V^(2/3)   (scale free)",
                                          init["ratio"], fin["ratio"], exact))
    print("%-34s %12.4f %12.4f %12.4f" % ("   two separate balls would give",
                                          balls, balls, balls))
    print("%-34s %12.4f %12.4f %12.4f" % ("r1 / d", init["r_over_d"][0],
                                          fin["r_over_d"][0], 1.0))
    print("%-34s %12.4f %12.4f %12.4f" % ("r2 / d", init["r_over_d"][1],
                                          fin["r_over_d"][1], 1.0))
    print("%-34s %12.2f %12.2f %12.2f" % ("smallest dihedral angle (deg)",
                                          init["angle_min"], fin["angle_min"], 120.0))
    print("%-34s %12.2f %12.2f %12.2f" % ("largest dihedral angle (deg)",
                                          init["angle_max"], fin["angle_max"], 120.0))
    print("%-34s %12.2e %12.2e %12.2e" % ("volume imbalance |V1-V2|/V",
                                          init["volume_imbalance"],
                                          fin["volume_imbalance"], 0.0))
    print()
    print("recovered radii %.4f, %.4f and separation %.4f"
          % (fin["radii"][0], fin["radii"][1], fin["separation"]))
    print("shape error in A / V^(2/3): %.2e relative"
          % (abs(fin["ratio"] - exact) / exact))


def convergence_study(resolutions=(16, 24, 32, 48), steps: int = 300,
                      lr: float = 2e-2, extractor=envelope3d) -> list[dict]:
    """Optimise from the same start on successively finer grids.

    The point is to separate what the optimizer got wrong from what the grid
    did. If the recovered shape converges to the closed-form answer under
    refinement, the residual at any one resolution is discretisation.
    """
    out = []
    for res in resolutions:
        run = optimise(steps=steps, resolution=res, lr=lr, extractor=extractor,
                       verbose=False)
        f = run["final"]
        out.append({
            "resolution": res,
            "topology_ok": run["topology_ok"],
            "ratio": f["ratio"],
            "ratio_error": abs(f["ratio"] - optimal_ratio()) / optimal_ratio(),
            "r_over_d_error": max(abs(v - 1.0) for v in f["r_over_d"]),
            "worst_angle_deviation": f["worst_angle_deviation"],
            "imbalance": f["volume_imbalance"],
        })
        print("  res %2d: A/V^(2/3) %.5f (rel %.2e), |r/d - 1| %.2e, worst angle %.3f deg"
              % (res, out[-1]["ratio"], out[-1]["ratio_error"],
                 out[-1]["r_over_d_error"], out[-1]["worst_angle_deviation"]))
    return out


def _mesh_arrays(surface) -> dict:
    return {
        "vertices": surface.vertices.detach().cpu().numpy(),
        "triangles": surface.triangles.detach().cpu().numpy(),
        "labels": surface.triangle_labels.detach().cpu().numpy(),
        "triple_segments": surface.triple_segments.detach().cpu().numpy(),
    }


def run_all(cache: Path | None = None, force: bool = False) -> dict:
    """Everything the figure needs, cached so layout can be iterated cheaply."""
    cache = cache or CACHE
    if cache.exists() and not force:
        with open(cache, "rb") as handle:
            return pickle.load(handle)

    print("optimising at the rendering resolution")
    run = optimise(steps=300, resolution=32, lr=2e-2)
    report(run)
    print("\nrefining the grid")
    study = convergence_study()
    print("\nthe same loss driven by corner-label case analysis")
    baseline = convergence_study(extractor=extract3d)

    data = {
        "baseline_study": baseline,
        "reference": run["reference"],
        "initial": run["initial"],
        "final": run["final"],
        "history": run["history"],
        "study": study,
        "resolution": run["resolution"],
        "topology_ok": run["topology_ok"],
        "init_mesh": _mesh_arrays(run["init_surface"]),
        "final_mesh": _mesh_arrays(run["final_surface"]),
        "optimal_ratio": optimal_ratio(),
        "separate_balls_ratio": separate_balls_ratio(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as handle:
        pickle.dump(data, handle)
    return data


PAIR_COLOUR = {(0, 1): "#4c72b0", (0, 2): "#dd8452", (1, 2): "#55a868"}


def figure(data: dict | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    data = data or run_all()
    fig = plt.figure(figsize=(14.6, 8.0))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.32, 1.0], hspace=0.0, wspace=0.85,
                            left=0.05, right=0.985, top=0.885, bottom=0.08)

    both = np.concatenate([data["init_mesh"]["vertices"], data["final_mesh"]["vertices"],
                           data["baseline_mesh"]["vertices"]])
    centre = 0.5 * (both.max(0) + both.min(0))
    lim = 0.55 * float((both.max(0) - both.min(0)).max())

    def render(ax, mesh, title, subtitle):
        verts, tris = mesh["vertices"], mesh["triangles"]
        pairs = mesh["labels"]
        light = np.array([0.4, 0.5, 0.75])
        # The wall is opaque and the two caps translucent, so the interior
        # surface and the circle where all three meet stay visible.
        for pair, alpha, zpos in (((1, 2), 1.0, -1.0), ((0, 1), 0.22, 1.0), ((0, 2), 0.22, 1.0)):
            sel = (pairs == np.array(pair)).all(axis=1)
            if not sel.any():
                continue
            poly = verts[tris[sel]]
            n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            lam = 0.55 + 0.45 * np.abs(n @ light / np.linalg.norm(light))
            base = np.array(matplotlib.colors.to_rgb(PAIR_COLOUR[pair]))
            col = Poly3DCollection(poly, facecolors=np.clip(base * lam[:, None], 0, 1),
                                   alpha=alpha, edgecolors="#ffffff", linewidths=0.05)
            col.set_sort_zpos(zpos)
            ax.add_collection3d(col)
        seg = mesh["triple_segments"]
        if len(seg):
            line = Line3DCollection(verts[seg], colors="#101010", linewidths=2.6)
            line.set_sort_zpos(2.0)
            ax.add_collection3d(line)
        ax.set_xlim(centre[0] - lim, centre[0] + lim)
        ax.set_ylim(centre[1] - lim, centre[1] + lim)
        ax.set_zlim(centre[2] - lim, centre[2] + lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.78)
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, pad=-14)
        ax.view_init(elev=20, azim=-58)

    init, fin, base = data["initial"], data["final"], data["baseline_final"]
    render(fig.add_subplot(grid[0, 0:2], projection="3d"), data["init_mesh"],
           "start: two lopsided lobes",
           r"$r_1/d$ = %.2f, %.2f;  angles %.0f$^\circ$ to %.0f$^\circ$"
           % (init["r_over_d"][0], init["r_over_d"][1], init["angle_min"], init["angle_max"]))
    render(fig.add_subplot(grid[0, 2:4], projection="3d"), data["final_mesh"],
           "exact arrangement: the standard double bubble",
           r"$r_1/d$ = %.3f, %.3f;  angles %.1f$^\circ$ to %.1f$^\circ$"
           % (fin["r_over_d"][0], fin["r_over_d"][1], fin["angle_min"], fin["angle_max"]))
    render(fig.add_subplot(grid[0, 4:6], projection="3d"), data["baseline_mesh"],
           "corner-label case analysis: a different shape",
           r"$r_1/d$ = %.3f, %.3f;  angles %.1f$^\circ$ to %.1f$^\circ$"
           % (base["r_over_d"][0], base["r_over_d"][1], base["angle_min"], base["angle_max"]))

    exact, balls = data["optimal_ratio"], data["separate_balls_ratio"]
    ax = fig.add_subplot(grid[1, 0:3])
    ax.plot([h["ratio"] for h in data["history"]], color="#1f4e79", lw=1.8,
            label="exact arrangement")
    ax.plot([h["ratio"] for h in data["baseline_history"]], color="#c44e52", lw=1.5,
            label="corner-label case analysis")
    ax.axhline(exact, color="#101010", ls="--", lw=1.4,
               label=r"standard double bubble, %.4f" % exact)
    ax.axhline(balls, color="#8c8c8c", ls=":", lw=1.4,
               label=r"two separate balls, %.4f" % balls)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel(r"$A/V^{2/3}$")
    ax.set_ylim(exact - 0.03, balls + 0.03)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.95)
    ax.set_title("the objective, in scale-free form", fontsize=10.5)

    ax = fig.add_subplot(grid[1, 3:6])
    res = np.array([s["resolution"] for s in data["study"]], dtype=float)
    for study, name, style, alpha in ((data["study"], "exact arrangement", "-", 1.0),
                                      (data["baseline_study"], "corner-label", "--", 0.85)):
        for key, metric, colour, marker in (
            ("ratio_error", r"$A/V^{2/3}$", "#1f4e79", "o"),
            ("r_over_d_error", r"$|r/d - 1|$", "#c44e52", "s"),
        ):
            ax.loglog(res, [s[key] for s in study], marker=marker, color=colour,
                      ls=style, lw=1.6, ms=5, alpha=alpha,
                      markerfacecolor=colour if style == "-" else "white",
                      label=f"{metric}, {name}")
    ref = data["study"][-1]["ratio_error"] * (res[-1] / res) ** 2
    ax.loglog(res, ref, ls=":", color="#8c8c8c", lw=1.3, label="second order")
    ax.set_xlabel("grid resolution")
    ax.set_ylabel("error in the recovered shape")
    ax.set_xticks(res)
    ax.set_xticklabels([f"{int(r)}" for r in res])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.legend(fontsize=7.4, loc="lower left", ncol=2, framealpha=0.95)
    ax.set_title("refinement fixes one of them", fontsize=10.5)

    fig.suptitle("Minimising interface area at equal volumes, with the loss passing "
                 "through the triple curve", fontsize=12.5, y=0.982)
    path = FIGDIR / "double_bubble.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")
    return str(path)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    figure(run_all())
