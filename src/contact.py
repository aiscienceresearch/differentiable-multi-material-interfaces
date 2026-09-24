"""Contact between solid objects, as geometry rather than as a penalty.

Write several objects as channels of one multi-label field and the region each
occupies is the argmax partition, so interpenetration is not discouraged, it is
unrepresentable: a point has one argmax and therefore one owner. What the exact
arrangement adds on top of that is the contact set itself. The surface shared by
objects i and j comes out as a single set of triangles carrying the label pair
(i, j) --- one triangulation, belonging to both objects, not two coincident
copies --- so its area is a function of the object parameters that autograd can
differentiate. Where three objects meet, the three contact patches close onto a
common curve, and that curve is in the output too.

That last part is what separates this from the guarantee offered by
intersection-free multi-object distance fields. Enforcing non-penetration as a
hard constraint keeps objects apart; it does not produce the shared patch, and
it has nothing to say about the curve along which three objects touch at once,
because each object still carries its own surface.

Two configurations have closed-form answers to check against.

Two balls of radius r whose centres are s apart. On the bisector plane both
interior logits equal r^2 - s^2/4 - y^2 - z^2, so the contact patch is a disk of
radius sqrt(r^2 - s^2/4):

    area(s) = pi (r^2 - s^2/4),   d area / d s = -pi s / 2,
    rim(s)  = 2 pi sqrt(r^2 - s^2/4),

for s < 2r and zero beyond.

Three balls of radius r with centres on a circle of radius R. All three interior
logits tie on the axis through the centre of that circle, and the background
joins them where additionally f = 0, which puts quadruple points at height
+-sqrt(r^2 - R^2) either side of the plane of centres. The curve along which the
three objects meet is the segment between them, of length 2 sqrt(r^2 - R^2).

Everything is run at a generic placement. Centring the configuration on the
origin puts the two-body patch in the plane x = 0 and the three-body curve on the
z axis, and both of those are grid planes and grid lines at every even
resolution; the symbolic perturbation handles it, but measuring convergence
there measures a symmetry rather than the method.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch

import envelope3d
import extract3d
from fields import BubbleField

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
PERTURB = 1e-9
CACHE = Path(__file__).resolve().parent.parent / "data" / "contact.pkl"
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# A generic shift, so that no interface lands on a grid plane and no triple
# curve on a grid line. Nothing depends on the particular numbers.
GENERIC = torch.tensor([0.0131, -0.0217, 0.0173])

TWO_RADIUS = 0.45
THREE_CIRCUM, THREE_RADIUS = 0.35, 0.50
OBJECT_PAIRS = ((1, 2), (1, 3), (2, 3))


# --------------------------------------------------------------------------
# configurations and their closed forms
# --------------------------------------------------------------------------

def two_body(separation: float, radius: float = TWO_RADIUS) -> BubbleField:
    half = separation / 2.0
    centres = torch.tensor([[half, 0.0, 0.0], [-half, 0.0, 0.0]]) + GENERIC
    return BubbleField(centres=centres, radii=[radius, radius])


def three_body(circum: float = THREE_CIRCUM, radii=None, phase: float = 0.0) -> BubbleField:
    ang = [2.0 * math.pi * k / 3.0 + phase for k in range(3)]
    centres = torch.tensor([[circum * math.cos(a), circum * math.sin(a), 0.0]
                            for a in ang]) + GENERIC
    return BubbleField(centres=centres, radii=radii or [THREE_RADIUS] * 3)


def exact_patch_area(s: float, radius: float = TWO_RADIUS) -> float:
    return 0.0 if s >= 2 * radius else math.pi * (radius ** 2 - s ** 2 / 4.0)


def exact_patch_slope(s: float, radius: float = TWO_RADIUS) -> float:
    """Discontinuous at onset: it tends to -pi r from below and is zero above."""
    return 0.0 if s >= 2 * radius else -math.pi * s / 2.0


def exact_rim_length(s: float, radius: float = TWO_RADIUS) -> float:
    if s >= 2 * radius:
        return 0.0
    return 2.0 * math.pi * math.sqrt(radius ** 2 - s ** 2 / 4.0)


def exact_triple_curve(circum: float = THREE_CIRCUM, radius: float = THREE_RADIUS) -> float:
    return 2.0 * math.sqrt(radius ** 2 - circum ** 2)


def exact_quadruple_height(circum: float = THREE_CIRCUM,
                           radius: float = THREE_RADIUS) -> float:
    return math.sqrt(radius ** 2 - circum ** 2)


# --------------------------------------------------------------------------
# measurement off the extracted complex
# --------------------------------------------------------------------------

def extract(field, resolution: int):
    return envelope3d.extract(field, resolution=resolution, box=BOX, perturb=PERTURB)


def _curve_length(surf, labels) -> torch.Tensor:
    """Total length of the curve carrying exactly this label triple."""
    zero = surf.vertices.sum() * 0.0
    if not surf.triple_segments.numel():
        return zero
    want = torch.tensor(sorted(labels), device=surf.triple_labels.device)
    sel = (surf.triple_labels.sort(dim=1).values == want).all(dim=1)
    if not bool(sel.any()):
        return zero
    p = surf.vertices[surf.triple_segments[sel]]
    return (p[:, 1] - p[:, 0]).norm(dim=-1).sum()


def _patch_area(surf, pair) -> torch.Tensor:
    area = surf.area_by_pair().get(tuple(pair))
    return surf.vertices.sum() * 0.0 if area is None else area


def measure_two(field, resolution: int):
    """Contact patch area and the length of the rim bounding it.

    The rim is where the two objects and the background all meet, so it is a
    triple curve of the partition and comes off the same complex as the patch.
    """
    surf = extract(field, resolution)
    return _patch_area(surf, (1, 2)), _curve_length(surf, (0, 1, 2)), surf


def measure_three(field, resolution: int):
    """The three contact patches and the curve where all three objects meet."""
    surf = extract(field, resolution)
    areas = {p: _patch_area(surf, p) for p in OBJECT_PAIRS}
    return areas, _curve_length(surf, (1, 2, 3)), surf


def _separation_grad(centre_grad: torch.Tensor) -> float:
    """Chain rule for centres placed at (+s/2, 0, 0) and (-s/2, 0, 0)."""
    return float((centre_grad[0, 0] - centre_grad[1, 0]) / 2.0)


# --------------------------------------------------------------------------
# studies
# --------------------------------------------------------------------------

def sweep_separation(resolution: int = 48, radius: float = TWO_RADIUS,
                     verbose: bool = True):
    """Walk two objects apart, from deep overlap through the loss of contact.

    The last rows are the point of it. As s approaches 2r the patch shrinks to
    nothing and the rim bounding it shrinks with it, and beyond that there is no
    contact set at all, so the derivative of the area jumps from -pi r to zero.
    Contact is differentiable while it persists, not while it forms.
    """
    fractions = (0.40, 0.60, 0.75, 0.85, 0.92, 0.96, 0.99, 1.00, 1.02)
    rows = []
    if verbose:
        print(f"\ntwo objects, radius {radius}, grid {resolution}^3, "
              f"cell {2.0 / resolution:.4f}")
        print(f"{'s/2r':>6} {'s':>7} | {'patch':>9} {'exact':>9} {'err':>8}"
              f" | {'d/ds':>9} {'exact':>9} {'err':>8}"
              f" | {'rim':>8} {'exact':>8} {'err':>8}")

    def rel(got, want):
        return abs(got - want) / abs(want) if abs(want) > 1e-12 else abs(got)

    for frac in fractions:
        s = 2.0 * radius * frac
        field = two_body(s, radius)
        area, rim, _ = measure_two(field, resolution)
        grad = torch.autograd.grad(area, field.centres, allow_unused=True)[0]
        slope = 0.0 if grad is None else _separation_grad(grad)

        a, c = float(area.detach()), float(rim.detach())
        ea = exact_patch_area(s, radius)
        eg = exact_patch_slope(s, radius)
        ec = exact_rim_length(s, radius)
        row = {"frac": frac, "s": s, "area": a, "exact_area": ea, "slope": slope,
               "exact_slope": eg, "rim": c, "exact_rim": ec,
               "area_err": rel(a, ea), "slope_err": rel(slope, eg),
               "rim_err": rel(c, ec)}
        rows.append(row)
        if verbose:
            print(f"{frac:6.2f} {s:7.4f} | {a:9.6f} {ea:9.6f} {row['area_err']:8.1e}"
                  f" | {slope:9.5f} {eg:9.5f} {row['slope_err']:8.1e}"
                  f" | {c:8.5f} {ec:8.5f} {row['rim_err']:8.1e}")
    return rows


def gradient_check(separation: float = 0.70, resolution: int = 48,
                   h: float = 1e-5, verbose: bool = True):
    """Autograd against a central difference on the extractor itself.

    The closed form says what the answer ought to be; this says whether the
    gradient is attached to what the extractor actually computed, which is a
    different question and the one that catches a severed graph.
    """
    field = two_body(separation)
    area, rim, _ = measure_two(field, resolution)
    ad_area = _separation_grad(torch.autograd.grad(area, field.centres,
                                                   retain_graph=True)[0])
    ad_rim = _separation_grad(torch.autograd.grad(rim, field.centres)[0])

    with torch.no_grad():
        up_a, up_c, _ = measure_two(two_body(separation + h), resolution)
        dn_a, dn_c, _ = measure_two(two_body(separation - h), resolution)
    fd_area = float((up_a - dn_a) / (2 * h))
    fd_rim = float((up_c - dn_c) / (2 * h))

    out = {"area_autograd": ad_area, "area_fd": fd_area,
           "area_rel": abs(ad_area - fd_area) / max(abs(fd_area), 1e-12),
           "rim_autograd": ad_rim, "rim_fd": fd_rim,
           "rim_rel": abs(ad_rim - fd_rim) / max(abs(fd_rim), 1e-12)}
    if verbose:
        print(f"\nautograd against central differences at s={separation}, h={h}")
        print(f"  d patch / ds  {ad_area:12.9f} vs {fd_area:12.9f}"
              f"   rel {out['area_rel']:.1e}")
        print(f"  d rim   / ds  {ad_rim:12.9f} vs {fd_rim:12.9f}"
              f"   rel {out['rim_rel']:.1e}")
    return out


def three_body_exactness(resolution: int = 48, verbose: bool = True):
    """The curve where three objects meet, against its closed form.

    Two things are checked that exist only because the arrangement is exact:
    the number of points at which that curve terminates on the background,
    which must be two, and their height above the plane of centres.
    """
    z = exact_quadruple_height()
    areas, curve, surf = measure_three(three_body(), resolution)
    q = surf.quadruple_points().detach()
    heights = sorted(float(v) - float(GENERIC[2]) for v in q[:, 2]) if q.shape[0] else []
    a = [float(areas[p].detach()) for p in OBJECT_PAIRS]
    got = float(curve.detach())

    out = {"resolution": resolution, "n_quadruple": int(q.shape[0]),
           "heights": heights, "exact_height": z,
           "height_err": max(abs(abs(v) - z) for v in heights) if heights else float("nan"),
           "curve": got, "exact_curve": 2 * z,
           "curve_err": abs(got - 2 * z) / (2 * z),
           "areas": a, "area_spread": (max(a) - min(a)) / max(a)}
    if verbose:
        print(f"\nthree objects in mutual contact, grid {resolution}^3")
        print(f"  points where the curve ends on the background: "
              f"{out['n_quadruple']} (exactly 2 expected)")
        print(f"  their height above the plane of centres: "
              f"{', '.join(f'{v:+.6f}' for v in heights)}   exact +-{z:.6f}"
              f"   err {out['height_err']:.2e}")
        print(f"  length of the curve where all three meet: {got:.6f}"
              f"   exact {2 * z:.6f}   rel err {out['curve_err']:.2e}")
        print(f"  the three contact patches: {', '.join(f'{v:.6f}' for v in a)}"
              f"   spread {out['area_spread']:.2e}")
    return out


def convergence(resolutions=(24, 32, 48, 64, 96, 128), verbose: bool = True):
    """Order of the contact patch area and of the three-object curve.

    The two quantities sit at opposite ends of the complex: the patch is a piece
    of surface, the curve is where three of them meet. The interior of the patch
    is reproduced with no error at all, since f_1 - f_2 is affine and its P1
    interpolant is itself; the error is entirely in where the boundary falls.
    """
    ea, ec = exact_patch_area(0.70), exact_triple_curve()
    rows = []
    if verbose:
        print(f"\n{'res':>5} | {'patch err':>11} {'order':>6} |"
              f" {'curve err':>11} {'order':>6}")
    for res in resolutions:
        area, _, _ = measure_two(two_body(0.70), res)
        _, curve, _ = measure_three(three_body(), res)
        row = {"resolution": res,
               "area_err": abs(float(area.detach()) - ea) / ea,
               "curve_err": abs(float(curve.detach()) - ec) / ec}
        if rows:
            base = math.log(res / rows[-1]["resolution"])
            row["area_order"] = math.log(rows[-1]["area_err"] / row["area_err"]) / base
            row["curve_order"] = math.log(rows[-1]["curve_err"] / row["curve_err"]) / base
        rows.append(row)
        if verbose:
            oa = f"{row['area_order']:6.2f}" if "area_order" in row else " " * 6
            oc = f"{row['curve_order']:6.2f}" if "curve_order" in row else " " * 6
            print(f"{res:>5} | {row['area_err']:>11.3e} {oa} |"
                  f" {row['curve_err']:>11.3e} {oc}")

    span = math.log(rows[-1]["resolution"] / rows[0]["resolution"])
    orders = {k.split("_")[0]: math.log(rows[0][k] / rows[-1][k]) / span
              for k in ("area_err", "curve_err")}
    if verbose:
        print(f"  fitted over the whole range: patch {orders['area']:.2f},"
              f" curve {orders['curve']:.2f}")
    return rows, orders


def partition_check(resolution: int = 48, verbose: bool = True):
    """What the output is, as opposed to what it approximates.

    Interpenetration is absent rather than small, so there is no overlap to
    report. The checkable consequences are that the complex is conforming ---
    every side of every triangle used twice except along the curves where three
    patches meet --- and that the contact geometry is a single triangulation
    whose triangles each belong to two objects at once. A representation that
    carries one surface per object holds two copies of that geometry and can
    only make them agree to a tolerance.
    """
    surf = extract(three_body(), resolution)
    topo = extract3d.check_topology(surf)
    vols = [float(surf.enclosed_volume(k).detach()) for k in (1, 2, 3)]

    tri = surf.triangle_labels
    count = lambda p: int((tri == torch.tensor(p, device=tri.device)).all(dim=1).sum())
    shared = sum(count(p) for p in OBJECT_PAIRS)
    free = sum(count((0, k)) for k in (1, 2, 3))

    out = {"volumes": vols, "shared_triangles": shared, "free_triangles": free,
           "conforming": bool(topo["ok"]), "num_triangles": int(tri.shape[0])}
    if verbose:
        print(f"\nthe partition itself, grid {resolution}^3")
        print(f"  object volumes {', '.join(f'{v:.6f}' for v in vols)}")
        print(f"  triangles on a free surface {free}, shared by two objects {shared}")
        print(f"  complex conforming and coherently oriented: {out['conforming']}")
    return out


def optimise_contact(steps: int = 140, resolution: int = 40, lr: float = 8e-3,
                     verbose: bool = True):
    """Drive three prescribed contact areas by moving and resizing the objects.

    The loss reads the three shared patches and nothing else. There is no
    penalty term, no margin and no collision query, because the partition
    cannot place two objects in the same spot to begin with; the optimizer is
    only ever asked about geometry it can see.

    The targets are measured off a reference configuration at the same
    resolution, so they are known to be reachable and the residual is the
    optimizer's rather than the parametrization's.
    """
    ref = three_body(circum=0.33, radii=[0.52, 0.46, 0.55], phase=0.21)
    with torch.no_grad():
        ref_areas, ref_curve, _ = measure_three(ref, resolution)
    targets = torch.stack([ref_areas[p] for p in OBJECT_PAIRS]).detach()

    field = three_body()  # symmetric, equal radii
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.05)
    if verbose:
        print(f"\ntargets for the three contact patches: "
              f"{', '.join(f'{float(t):.6f}' for t in targets)}")

    history = []
    for step in range(steps):
        opt.zero_grad()
        areas, curve, _ = measure_three(field, resolution)
        got = torch.stack([areas[p] for p in OBJECT_PAIRS])
        residual = (got - targets) / targets
        loss = residual.pow(2).sum()
        loss.backward()
        opt.step()
        sched.step()
        history.append({"loss": float(loss.detach()),
                        "areas": [float(v) for v in got.detach()],
                        "curve": float(curve.detach()),
                        "worst": float(residual.abs().max().detach())})
        if verbose and (step % 20 == 0 or step == steps - 1):
            h = history[-1]
            print(f"  step {step:4d}  loss {h['loss']:.3e}  worst relative miss "
                  f"{h['worst']:.2e}  patches "
                  f"{', '.join(f'{v:.5f}' for v in h['areas'])}")

    areas, curve, surf = measure_three(field, resolution)
    got = [float(areas[p].detach()) for p in OBJECT_PAIRS]
    out = {"targets": [float(t) for t in targets], "final": got,
           "worst": max(abs(g - float(t)) / float(t) for g, t in zip(got, targets)),
           "history": history, "resolution": resolution,
           "final_curve": float(curve.detach()),
           "reference_curve": float(ref_curve.detach())}
    if verbose:
        print(f"  reached the targets to {out['worst']:.2e} relative, worst of three")
    return out, field, surf


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _mesh_arrays(surface) -> dict:
    return {
        "vertices": surface.vertices.detach().cpu().numpy(),
        "triangles": surface.triangles.detach().cpu().numpy(),
        "labels": surface.triangle_labels.detach().cpu().numpy(),
        "triple_segments": surface.triple_segments.detach().cpu().numpy(),
        "triple_labels": surface.triple_labels.detach().cpu().numpy(),
        "quadruple": surface.quadruple_points().detach().cpu().numpy(),
    }


def run_all(cache: Path | None = None, force: bool = False) -> dict:
    cache = cache or CACHE
    if cache.exists() and not force:
        with open(cache, "rb") as handle:
            return pickle.load(handle)

    print("=" * 78)
    print("Contact as geometry: the shared patch, its rim, and the curve along")
    print("which three objects meet")
    print("=" * 78)

    sweep = sweep_separation()
    grad = gradient_check()
    three = three_body_exactness()
    rows, orders = convergence()
    part = partition_check()
    opt, _, surf = optimise_contact()

    data = {
        "sweep": sweep, "gradient": grad, "three": three,
        "convergence": rows, "orders": orders, "partition": part,
        "optimisation": opt,
        "start_mesh": _mesh_arrays(extract(three_body(), opt["resolution"])),
        "final_mesh": _mesh_arrays(surf),
        "exact_curve": exact_triple_curve(),
        "exact_height": exact_quadruple_height(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as handle:
        pickle.dump(data, handle)
    return data


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

FREE_COLOUR = "#93a9c4"
CONTACT_COLOUR = {(1, 2): "#55a868", (1, 3): "#dd8452", (2, 3): "#8172b3"}


def _shade(base, normals):
    import matplotlib
    import numpy as np
    light = np.array([0.4, 0.5, 0.75])
    lam = 0.58 + 0.42 * np.abs(normals @ light / np.linalg.norm(light))
    rgb = np.array(matplotlib.colors.to_rgb(base))
    return np.clip(rgb * lam[:, None], 0.0, 1.0)


def figure(data: dict | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    data = data or run_all()
    fig = plt.figure(figsize=(14.6, 8.2))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.28, 1.0], hspace=0.02,
                            wspace=0.95, left=0.045, right=0.985, top=0.885,
                            bottom=0.085)

    allv = np.concatenate([data["start_mesh"]["vertices"],
                           data["final_mesh"]["vertices"]])
    centre = 0.5 * (allv.max(0) + allv.min(0))
    lim = 0.56 * float((allv.max(0) - allv.min(0)).max())

    def add_patch(ax, mesh, pair, colour, alpha, zpos):
        sel = (mesh["labels"] == np.array(pair)).all(axis=1)
        if not sel.any():
            return
        poly = mesh["vertices"][mesh["triangles"][sel]]
        n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        col = Poly3DCollection(poly, facecolors=_shade(colour, n), alpha=alpha,
                               edgecolors="#ffffff", linewidths=0.04)
        col.set_sort_zpos(zpos)
        ax.add_collection3d(col)

    def add_curve(ax, mesh, labels, colour, width, zpos):
        seg, lab = mesh["triple_segments"], mesh["triple_labels"]
        if not len(seg):
            return
        sel = (np.sort(lab, axis=1) == np.array(sorted(labels))).all(axis=1)
        if not sel.any():
            return
        line = Line3DCollection(mesh["vertices"][seg[sel]], colors=colour,
                                linewidths=width)
        line.set_sort_zpos(zpos)
        ax.add_collection3d(line)

    # The three contact planes all contain the curve where the objects meet, so
    # they splay around it at roughly 120 degrees. Any view from near the plane
    # of centres puts one of them edge-on; looking down on the curve shows all
    # three at once.
    def finish(ax, title, subtitle, half, mid=centre, elev=36, zoom=1.72):
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, pad=-12)
        ax.view_init(elev=elev, azim=-58)

    def render_solid(ax, mesh, title, subtitle):
        for pair, colour in CONTACT_COLOUR.items():
            add_patch(ax, mesh, pair, colour, 1.0, -1.0)
        for k in (1, 2, 3):
            add_patch(ax, mesh, (0, k), FREE_COLOUR, 0.16, 1.0)
        add_curve(ax, mesh, (1, 2, 3), "#101010", 2.8, 2.0)
        finish(ax, title, subtitle, lim)

    def render_contact_only(ax, mesh, title, subtitle):
        """Just the shared geometry: the three patches, their rims, and the
        curve where all three objects meet."""
        for pair, colour in CONTACT_COLOUR.items():
            add_patch(ax, mesh, pair, colour, 0.96, -1.0)
            add_curve(ax, mesh, (0,) + pair, "#6b6b6b", 1.0, 1.5)
        add_curve(ax, mesh, (1, 2, 3), "#101010", 3.0, 2.0)
        q = mesh["quadruple"]
        if len(q):
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=46, c="#101010",
                       depthshade=False, zorder=10)
        # The contact set occupies far less room than the objects do, so it gets
        # its own extent rather than being left small inside a shared one.
        sel = np.zeros(len(mesh["labels"]), dtype=bool)
        for pair in CONTACT_COLOUR:
            sel |= (mesh["labels"] == np.array(pair)).all(axis=1)
        pts = mesh["vertices"][np.unique(mesh["triangles"][sel])]
        finish(ax, title, subtitle,
               0.62 * float((pts.max(0) - pts.min(0)).max()),
               mid=0.5 * (pts.max(0) + pts.min(0)), elev=44, zoom=1.30)

    start, final = data["start_mesh"], data["final_mesh"]
    opt, three = data["optimisation"], data["three"]
    render_solid(fig.add_subplot(grid[0, 0:2], projection="3d"), start,
                 "three objects, symmetric start",
                 "patches %.3f, %.3f, %.3f"
                 % tuple(opt["history"][0]["areas"]))
    render_solid(fig.add_subplot(grid[0, 2:4], projection="3d"), final,
                 "after optimising the contact areas",
                 "patches %.3f, %.3f, %.3f;  worst miss %.0e"
                 % (*opt["final"], opt["worst"]))
    render_contact_only(fig.add_subplot(grid[0, 4:6], projection="3d"), final,
                        "the contact set on its own",
                        "three shared patches meeting along one curve")

    # -- accuracy through the loss of contact -----------------------------
    ax = fig.add_subplot(grid[1, 0:2])
    sweep = data["sweep"]
    x = np.array([r["frac"] for r in sweep])
    fine = np.linspace(0.35, 1.05, 400)
    r = TWO_RADIUS
    ax.plot(fine, [exact_patch_area(2 * r * t) for t in fine], color="#1f4e79",
            lw=1.5, label="patch area, closed form")
    ax.plot(x, [row["area"] for row in sweep], "o", ms=5.5, color="#1f4e79",
            markerfacecolor="white", label="patch area, extracted")
    ax.plot(fine, [-exact_patch_slope(2 * r * t) for t in fine], color="#c44e52",
            lw=1.5, ls="--", label=r"$-\,d(\mathrm{area})/ds$, closed form")
    ax.plot(x, [-row["slope"] for row in sweep], "s", ms=5.0, color="#c44e52",
            markerfacecolor="white", label="extracted, by autograd")
    ax.axvline(1.0, color="#8c8c8c", ls=":", lw=1.3)
    ax.annotate("contact lost", xy=(0.988, 0.06), fontsize=8, color="#5c5c5c",
                rotation=90, ha="right", va="bottom")
    ax.set_xlabel(r"separation $s/2r$")
    ax.set_ylabel("area, and its derivative")
    ax.set_xlim(0.35, 1.06)
    ax.set_ylim(-0.05, 1.75)
    ax.legend(fontsize=7.3, loc="lower left", framealpha=0.95)
    ax.set_title("the patch and its gradient, up to onset", fontsize=10.5)

    # -- convergence -------------------------------------------------------
    ax = fig.add_subplot(grid[1, 2:4])
    rows = data["convergence"]
    res = np.array([row["resolution"] for row in rows], dtype=float)
    ax.loglog(res, [row["area_err"] for row in rows], "o-", color="#1f4e79",
              lw=1.6, ms=5, label="contact patch area (%.2f)" % data["orders"]["area"])
    ax.loglog(res, [row["curve_err"] for row in rows], "s-", color="#55a868",
              lw=1.6, ms=5, label="three-object curve (%.2f)" % data["orders"]["curve"])
    ref = rows[-1]["area_err"] * (res[-1] / res) ** 2
    ax.loglog(res, ref, ls=":", color="#8c8c8c", lw=1.3, label="second order")
    ax.set_xlabel("grid resolution")
    ax.set_ylabel("relative error against closed form")
    ax.set_xticks(res)
    ax.set_xticklabels([f"{int(v)}" for v in res])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.legend(fontsize=7.6, loc="lower left", framealpha=0.95)
    ax.set_title("both ends of the complex converge", fontsize=10.5)

    # -- optimisation ------------------------------------------------------
    ax = fig.add_subplot(grid[1, 4:6])
    hist = opt["history"]
    for i, pair in enumerate(OBJECT_PAIRS):
        colour = CONTACT_COLOUR[pair]
        ax.plot([h["areas"][i] for h in hist], color=colour, lw=1.8,
                label=r"patch $%d\,|\,%d$" % pair)
        ax.axhline(opt["targets"][i], color=colour, ls="--", lw=1.1, alpha=0.75)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("contact patch area")
    ax.legend(fontsize=7.6, loc="center right", framealpha=0.95)
    ax.set_title("three prescribed contact areas, no penalty term",
                 fontsize=10.5)

    fig.suptitle("Contact as geometry: the shared patch is in the output, and "
                 "it is differentiable", fontsize=12.5, y=0.978)
    path = FIGDIR / "contact.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")
    return str(path)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    figure(run_all())
