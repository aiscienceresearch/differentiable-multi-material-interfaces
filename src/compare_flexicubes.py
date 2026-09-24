"""Head-to-head against FlexiCubes, the differentiable extractor we claim to subsume.

The paper argues that differentiable isosurface extractors cannot represent a
multi-material junction, because the two-manifold output they guarantee is
exactly what a triple curve violates, and that ours reduces to the two-label
case when K = 2. Both halves of that are arguments, not measurements, so this
measures them against FlexiCubes [SMG+23], the strongest of that family.

Three arms.

  A. Extraction accuracy at K = 2 on fields with closed-form surfaces. Same
     scalar field, matched grids, no optimisation. Establishes whether the
     reduction to two labels is competitive or merely possible.

  B. Optimisation at K = 2 driving the same parameter tensor through both
     extractors with one loss. This is the controlled version of the question
     the paper's differentiability claim raises, so the scalar grid is the only
     thing either method is allowed to move. FlexiCubes also carries per-cube
     weights, and holding those fixed tests its gradient path rather than its
     method, so it gets a third run in which they are free too.

  C. The junction, at K = 3. FlexiCubes consumes one scalar field, so the only
     way to ask it for three regions is one-vs-rest: extract each region's
     boundary separately and hope the results agree. They cannot agree along a
     triple curve, and the arm measures by how much they fail to.

Two asymmetries, stated rather than buried. FlexiCubes runs in float32, because
its working buffers are allocated at the ambient default dtype while its
default weights are Float; everything else here runs in float64. And in arm A
its weights stay at their defaults, since a fixed field gives nothing to
optimise them against.

[SMG+23] Shen, Munkberg, Hasselgren, Yin, Wang, Chen, Fidler, Litany, Gao.
Flexible Isosurface Extraction for Gradient-Based Mesh Optimization. SIGGRAPH
2023.
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
FC_DIR = ROOT / "external" / "flexicubes" / "FlexiCubes"
if not FC_DIR.exists():
    sys.exit(f"FlexiCubes not found at {FC_DIR}.\n"
             f"Run external/flexicubes/fetch.ps1 first.")
sys.path.insert(0, str(FC_DIR))
from flexicubes import FlexiCubes  # noqa: E402

import envelope3d  # noqa: E402
import extract3d  # noqa: E402
from fields import BubbleField, MultiLabelField  # noqa: E402

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "data" / "flexicubes_comparison.json"


@contextlib.contextmanager
def float32_default():
    """FlexiCubes only runs with float32 as the ambient default; see the README."""
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------
# Shared field, so both extractors read one parameter tensor
# --------------------------------------------------------------------------
def trilinear(grid: Tensor, x: Tensor, box=BOX) -> Tensor:
    """Sample a regular scalar grid at arbitrary points, differentiably.

    Both extractors query the lattice points themselves, where this returns the
    stored value exactly with weights 1 and 0. Going through interpolation
    anyway means neither method depends on the other's node ordering, which is
    what lets a single parameter tensor drive both.
    """
    n = grid.shape[0]
    lo = torch.tensor([box[0], box[2], box[4]], dtype=x.dtype, device=x.device)
    hi = torch.tensor([box[1], box[3], box[5]], dtype=x.dtype, device=x.device)
    u = ((x - lo) / (hi - lo) * (n - 1)).clamp(0, n - 1)
    i0 = u.floor().clamp(0, n - 2).long()
    f = u - i0
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                     * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                     * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                out = out + w * grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


class GridScalarField(MultiLabelField):
    """A stored scalar grid, presented to our extractor as two logits.

    Label 1 wins where the scalar is negative, matching the convention
    FlexiCubes requires of its input, so both extractors are handed the same
    sign of the same field.
    """

    num_classes = 2
    dim = 3

    def __init__(self, grid: Tensor):
        super().__init__()
        self.grid = grid

    def logits(self, x: Tensor) -> Tensor:
        s = trilinear(self.grid, x)
        return torch.stack([s * 0.5, -s * 0.5], dim=-1)


class AnalyticField(MultiLabelField):
    """Two labels from a closed-form signed distance function."""

    num_classes = 2
    dim = 3

    def __init__(self, sdf):
        super().__init__()
        self.sdf = sdf

    def logits(self, x: Tensor) -> Tensor:
        s = self.sdf(x)
        return torch.stack([s * 0.5, -s * 0.5], dim=-1)


def sphere_sdf(radius=0.62):
    return lambda x: x.norm(dim=-1) - radius


def torus_sdf(major=0.58, minor=0.24):
    def f(x):
        q = (x[..., :2].pow(2).sum(-1).sqrt() - major)
        return torch.stack([q, x[..., 2]], dim=-1).norm(dim=-1) - minor
    return f


# --------------------------------------------------------------------------
# Running each extractor
# --------------------------------------------------------------------------
def run_flexicubes(sdf_on, resolution, weights=None, training=False):
    """Extract with FlexiCubes. `sdf_on` maps grid positions to scalar values."""
    with float32_default():
        fc = FlexiCubes(device=DEVICE)
        x, cube = fc.construct_voxel_grid(resolution)
        x = x * 2.0  # their grid spans [-0.5, 0.5]; the paper's box is [-1, 1]
        s = sdf_on(x.double()).float()
        kw = weights or {}
        v, f, ldev = fc(x, s, cube, resolution, training=training, **kw)
    return v, f, ldev


def mesh_area(v: Tensor, f: Tensor) -> float:
    t = v.double()[f.long()]
    return float(torch.linalg.cross(t[:, 1] - t[:, 0],
                                    t[:, 2] - t[:, 0]).norm(dim=-1).sum() * 0.5)


def sample_surface(v: Tensor, f: Tensor, n: int) -> Tensor:
    """Draw `n` points uniformly by area from a triangle mesh, differentiably.

    Scoring the extracted vertices directly would favour whichever method emits
    more of them, and ours emits about three times as many as FlexiCubes on the
    same grid. Sampling a fixed budget by area removes that from the comparison.
    """
    tri = v[f]
    area = torch.linalg.cross(tri[:, 1] - tri[:, 0],
                              tri[:, 2] - tri[:, 0]).norm(dim=-1) * 0.5
    idx = torch.multinomial(area.detach().clamp_min(0) + 1e-20, n, replacement=True)
    t = tri[idx]
    u, w = torch.rand(n, 1, dtype=v.dtype, device=v.device), \
        torch.rand(n, 1, dtype=v.dtype, device=v.device)
    su = u.sqrt()
    return (1 - su) * t[:, 0] + su * (1 - w) * t[:, 1] + su * w * t[:, 2]


def chamfer(p: Tensor, q: Tensor) -> Tensor:
    """Symmetric squared Chamfer distance.

    Both directions are needed. The forward term alone is minimised by a
    surface that shrinks until it has no area left to be scored, which is what
    a one-sided loss actually does here.
    """
    d = torch.cdist(p, q)
    return d.min(dim=1).values.pow(2).mean() + d.min(dim=0).values.pow(2).mean()


def ellipsoid_sdf(axes=(0.78, 0.52, 0.34)):
    """Hard-to-compute exactly; this is the standard first-order approximation."""
    a = torch.tensor(axes, dtype=torch.float64)

    def f(x):
        k0 = (x / a.to(x.device)).norm(dim=-1)
        k1 = (x / a.to(x.device).pow(2)).norm(dim=-1)
        return k0 * (k0 - 1.0) / k1.clamp_min(1e-30)
    return f


def ellipsoid_cloud(n: int, axes=(0.78, 0.52, 0.34), seed: int = 0) -> Tensor:
    """Uniform by area on the ellipsoid.

    Scaling uniform samples from the sphere crowds the flattened ends, so the
    samples are reweighted by the area element the scaling induces.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor(axes, dtype=torch.float64)
    u = torch.randn(8 * n, 3, generator=g, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    w = (u / a).norm(dim=-1) * a.prod()
    return (u * a)[torch.multinomial(w, n, replacement=True, generator=g)]


def torus_cloud(n: int, major=0.58, minor=0.24, seed: int = 0) -> Tensor:
    """Points spread uniformly by area over the target torus.

    The area element grows with `major + minor cos(phi)`, so `phi` is drawn by
    rejection rather than uniformly, which would crowd the inner rim.
    """
    g = torch.Generator().manual_seed(seed)
    phi = torch.empty(0, dtype=torch.float64)
    while phi.numel() < n:
        cand = torch.rand(4 * n, generator=g, dtype=torch.float64) * 2 * math.pi
        keep = torch.rand(4 * n, generator=g, dtype=torch.float64) < \
            (major + minor * cand.cos()) / (major + minor)
        phi = torch.cat([phi, cand[keep]])
    phi = phi[:n]
    theta = torch.rand(n, generator=g, dtype=torch.float64) * 2 * math.pi
    rad = major + minor * phi.cos()
    return torch.stack([rad * theta.cos(), rad * theta.sin(), minor * phi.sin()], -1)


def region_boundary(surface, label: int):
    """The closed, outward-oriented boundary of one region of our complex.

    `enclosed_volume` already knows how to orient these; the same rule is used
    here so the region solids the winding test sees are the ones the paper
    claims to produce.
    """
    involved = (surface.triangle_labels == label).any(dim=1)
    tris = surface.triangles[involved]
    outward = surface.triangle_labels[involved, 1] == label
    tris = torch.where(outward[:, None], tris[:, [0, 2, 1]], tris)
    return surface.vertices.detach().double(), tris.long()


def winding_inside(v: Tensor, f: Tensor, q: Tensor, chunk: int = 128) -> Tensor:
    """Generalised winding number test, |w| > 0.5 meaning inside.

    Taking the absolute value keeps the test indifferent to which way a mesh
    happens to be wound, which matters because the two methods orient their
    output by different conventions and the comparison should not turn on that.
    """
    v, q = v.double().to(DEVICE), q.double().to(DEVICE)
    tri = v[f.to(DEVICE)]
    out = torch.zeros(q.shape[0], dtype=torch.float64, device=DEVICE)
    for start in range(0, q.shape[0], chunk):
        p = q[start:start + chunk, None, :]
        a, b, c = tri[None, :, 0] - p, tri[None, :, 1] - p, tri[None, :, 2] - p
        na, nb, nc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        num = (a * torch.linalg.cross(b, c)).sum(-1)
        den = (na * nb * nc + (a * b).sum(-1) * nc
               + (a * c).sum(-1) * nb + (b * c).sum(-1) * na)
        out[start:start + chunk] = 2.0 * torch.atan2(num, den).sum(dim=1)
    return (out / (4.0 * math.pi)).abs() > 0.5


# --------------------------------------------------------------------------
# Arm A: extraction accuracy at K = 2
# --------------------------------------------------------------------------
def arm_a(resolutions=(16, 32, 48, 64)):
    print("\n" + "=" * 78)
    print("A. Extraction accuracy at K=2, fixed field, matched grids")
    print("=" * 78)
    print("   Error is distance to the analytic surface in grid cells, scored two")
    print("   ways: over points sampled uniformly by area, and over the extracted")
    print("   vertices. The sampled figure is the comparable one. Our vertices are")
    print("   by construction zeros of the interpolant along tet edges, whereas a")
    print("   dual vertex is an average of its cell's crossings and is not meant to")
    print("   lie on the surface at all, so the vertex figure flatters us.")
    print("   FlexiCubes runs float32 with default weights; ours runs float64.")

    cases = [("sphere", sphere_sdf(), 4.0 * math.pi * 0.62 ** 2),
             ("torus", torus_sdf(), 4.0 * math.pi ** 2 * 0.58 * 0.24)]
    rows = []
    for name, sdf, exact_area in cases:
        print(f"\n   {name}, exact area {exact_area:.6f}")
        print(f"   {'res':>4} {'method':>12} {'tris':>8} {'area err':>11} "
              f"{'max d/h':>9} {'rms d/h':>9} {'max vert':>9} {'rms vert':>9}")
        for res in resolutions:
            h = 2.0 / res
            score = _score_factory(sdf, h)
            surf = envelope3d.extract(AnalyticField(sdf), resolution=res, box=BOX)
            entries = [("ours", surf.vertices.detach(), surf.triangles.long(),
                        float(surf.total_area().detach()),
                        int(surf.triangles.shape[0]))]

            v, f, _ = run_flexicubes(sdf, res)
            entries.append(("flexicubes", v.detach().double(), f.long(),
                            mesh_area(v, f), int(f.shape[0])))

            for method, verts, tris, area, ntri in entries:
                rms_s, max_s = score(verts, tris)
                d = sdf(verts).abs()
                row = {"shape": name, "resolution": res, "method": method,
                       "triangles": ntri,
                       "area_rel_err": abs(area - exact_area) / exact_area,
                       "max_dist_cells": max_s,
                       "rms_dist_cells": rms_s,
                       "max_vertex_cells": float(d.max()) / h,
                       "rms_vertex_cells": float((d ** 2).mean().sqrt()) / h}
                rows.append(row)
                print(f"   {res:>4} {method:>12} {ntri:>8} "
                      f"{row['area_rel_err']:>11.3e} "
                      f"{row['max_dist_cells']:>9.4f} {row['rms_dist_cells']:>9.4f} "
                      f"{row['max_vertex_cells']:>9.4f} "
                      f"{row['rms_vertex_cells']:>9.4f}")
    return rows


# --------------------------------------------------------------------------
# Arm B: optimisation at K = 2 through one parameter tensor
# --------------------------------------------------------------------------
def _score_factory(sdf, h):
    def score(v, f):
        p = sample_surface(v.double(), f.long(), 8192).detach()
        d = sdf(p).abs()
        return float((d ** 2).mean().sqrt()) / h, float(d.max()) / h
    return score


def _settled(hist, k: int = 20) -> float:
    """Mean loss over the last few steps, so a noisy trace is not judged on one."""
    tail = hist[-k:] if hist else [float("inf")]
    return sum(tail) / len(tail)


def laplacian(grid: Tensor) -> Tensor:
    """Mean squared discrete Laplacian of the scalar grid.

    Optimising 36k free grid values against a sampled Chamfer loss is a weak
    and noisy signal, and the two extractors respond to that noise very
    differently: ours reports the argmax of whatever field it is handed, while
    a dual construction averages each cell's crossings and damps it. Leaving
    the field unregularised would therefore measure that difference rather than
    the extractors, so the penalty is swept and applied identically to both.
    """
    lap = (-6.0 * grid[1:-1, 1:-1, 1:-1]
           + grid[2:, 1:-1, 1:-1] + grid[:-2, 1:-1, 1:-1]
           + grid[1:-1, 2:, 1:-1] + grid[1:-1, :-2, 1:-1]
           + grid[1:-1, 1:-1, 2:] + grid[1:-1, 1:-1, :-2])
    return lap.pow(2).mean()


def _optimise_ours(init, cloud, sdf, resolution, steps, lr, reg, n_sample, h):
    torch.manual_seed(0)
    grid = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([grid], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        surf = envelope3d.extract(GridScalarField(grid), resolution=resolution, box=BOX)
        if surf.triangles.shape[0] == 0:
            return {"collapsed": True, "history": hist or [float("inf")]}
        fit = chamfer(sample_surface(surf.vertices, surf.triangles.long(), n_sample),
                      cloud)
        (fit + reg * laplacian(grid)).backward()
        opt.step()
        hist.append(float(fit.detach()))
    surf = envelope3d.extract(GridScalarField(grid), resolution=resolution, box=BOX)
    rms, mx = _score_factory(sdf, h)(surf.vertices.detach(), surf.triangles)
    return {"collapsed": False, "history": hist, "rms_dist_cells": rms,
            "max_dist_cells": mx, "triangles": int(surf.triangles.shape[0]),
            "closes": bool(extract3d.check_topology(surf)["ok"])}


def _optimise_fc(init, cloud, sdf, resolution, steps, lr, reg, n_sample, h, free):
    torch.manual_seed(0)
    with float32_default():
        fc = FlexiCubes(device=DEVICE)
        x, cube = fc.construct_voxel_grid(resolution)
        x = (x * 2.0).detach()
        grid = init.clone().float().to(DEVICE).requires_grad_(True)
        params, w = [grid], {}
        if free:
            nc = cube.shape[0]
            w = {"beta_fx12": torch.ones((nc, 12), device=DEVICE, requires_grad=True),
                 "alpha_fx8": torch.ones((nc, 8), device=DEVICE, requires_grad=True),
                 "gamma_f": torch.ones((nc,), device=DEVICE, requires_grad=True)}
            params += list(w.values())
        opt = torch.optim.Adam(params, lr=lr)
        cloud_d = cloud.to(DEVICE)
        hist = []
        for _ in range(steps):
            opt.zero_grad()
            s = trilinear(grid.double(), x.double()).float()
            v, f, ldev = fc(x, s, cube, resolution, training=True, **w)
            if f.shape[0] == 0:
                return {"collapsed": True, "history": hist or [float("inf")]}
            fit = chamfer(sample_surface(v.double(), f.long(), n_sample), cloud_d)
            hist.append(float(fit.detach()))
            loss = fit + reg * laplacian(grid.double())
            if free:
                # The developability regulariser their paper recommends
                # alongside free weights; omitting it would be a handicap of
                # our choosing rather than a property of the method.
                loss = loss + 0.25 * ldev.double().mean()
            loss.backward()
            opt.step()
        s = trilinear(grid.double(), x.double()).float()
        v, f, _ = fc(x, s, cube, resolution, **w)
    rms, mx = _score_factory(sdf, h)(v.detach(), f)
    return {"collapsed": False, "history": hist, "rms_dist_cells": rms,
            "max_dist_cells": mx, "triangles": int(f.shape[0]), "closes": None}


# The smoothness weight runs to 10 because our error was still falling at 1e-1,
# the old top of the grid. Stopping a sweep while the curve is still dropping
# and then reporting that nothing in the sweep closed the gap would be an
# artefact of where the sweep ended rather than a finding.
ELLIPSOID_GRID = [(lr, rg) for rg in (0.0, 1e-2, 1e-1, 1.0, 10.0)
                  for lr in (5e-3, 1e-2)]
TORUS_GRID = [(lr, rg) for rg in (1e-2, 1e-1, 1.0) for lr in (5e-3, 1e-2)]


def arm_b(resolution: int = 32, steps: int = 250, configs=None,
          n_sample: int = 2048, n_target: int = 4096, target: str = "ellipsoid"):
    """Same scalar grid, same loss, two extractors.

    Only the scalar grid is allowed to move, so this is a comparison of the two
    gradient paths and not of the two methods' full machinery. FlexiCubes also
    carries per-cube weights, and freezing them tests its gradient path rather
    than its method, so it gets a second run in which they are free and its own
    developability regulariser is switched on.

    Two targets. The ellipsoid is a pure deformation and both should reach it,
    which is what makes the precision comparable. The torus additionally
    demands a genus change, which is the case a fixed-topology mesh cannot do
    at all and an implicit one can in principle; it is reported because neither
    extractor manages it from this initialisation, and reporting only the easy
    target would leave that unsaid.

    The loss is symmetric Chamfer against a target point cloud. Accuracy is
    scored separately, as the distance from area-sampled surface points to the
    analytic target, so the number reported is not the number optimised.

    Each method sweeps learning rate and field-smoothness weight and is
    reported at its best. Shared hyperparameters look fairer than they are:
    at the largest rate our trace oscillates where FlexiCubes' is flat, and
    without a smoothness penalty our surface carries field noise that their
    dual averaging removes. Fixing either for both would report a tuning
    accident as a property of the extractor.
    """
    print("\n" + "=" * 78)
    print(f"B. Optimisation at K=2 towards the {target}:"
          f" one parameter tensor, two extractors")
    print("=" * 78)
    print(f"   Grid {resolution}^3 initialised to a sphere. Symmetric Chamfer on")
    print("   area-weighted samples; error is distance to the analytic target.")

    if target == "ellipsoid":
        sdf, cloud = ellipsoid_sdf(), ellipsoid_cloud(n_target)
    elif target == "torus":
        sdf, cloud = torus_sdf(), torus_cloud(n_target)
    else:
        raise ValueError(target)
    n = resolution + 1
    lin = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    init = torch.stack([gx, gy, gz], dim=-1).norm(dim=-1) - 0.62
    h = 2.0 / resolution
    configs = configs or (ELLIPSOID_GRID if target == "ellipsoid" else TORUS_GRID)
    runners = {
        "ours": lambda lr, rg: _optimise_ours(init, cloud, sdf, resolution, steps,
                                              lr, rg, n_sample, h),
        "flexicubes": lambda lr, rg: _optimise_fc(init, cloud, sdf, resolution, steps,
                                                  lr, rg, n_sample, h, False),
        "flexicubes+weights": lambda lr, rg: _optimise_fc(init, cloud, sdf, resolution,
                                                          steps, lr, rg, n_sample, h,
                                                          True),
    }

    results, sweep = {}, []
    for name, run in runners.items():
        best = None
        for lr, rg in configs:
            r = run(lr, rg)
            r["lr"], r["reg"], r["method"] = lr, rg, name
            tag = "collapsed" if r["collapsed"] else f"{_settled(r['history']):.3e}"
            rms = "" if r["collapsed"] else f"  rms d/h {r['rms_dist_cells']:.4f}"
            print(f"     {name:<18} lr {lr:<7.0e} reg {rg:<7.0e} settled {tag}{rms}")
            # Every setting is kept, not just the winner. The effect of the
            # smoothness weight is itself discussed in the paper, so the
            # numbers behind that discussion have to survive the run.
            sweep.append({k: v for k, v in r.items() if k != "history"})
            if not r["collapsed"] and (
                    best is None or _settled(r["history"]) < _settled(best["history"])):
                best = r
        if best is None:
            print(f"     {name}: collapsed at every setting")
            continue
        results[name] = best

    # What the smoothness penalty is worth, at the best rate for each method.
    print("\n   effect of the field smoothness penalty, at each method's best rate:")
    for name in runners:
        rows = [s for s in sweep if s["method"] == name and not s["collapsed"]
                and s["lr"] == results[name]["lr"]]
        for s in sorted(rows, key=lambda s: s["reg"]):
            print(f"     {name:<18} reg {s['reg']:<7.0e} rms d/h "
                  f"{s['rms_dist_cells']:.4f}")
    results["_sweep"] = sweep

    print(f"\n   {'method':>20} {'lr':>8} {'reg':>8} {'rms d/h':>9} {'max d/h':>9} "
          f"{'tris':>8} {'chamfer':>10} {'closes':>7}")
    for k, r in ((k, v) for k, v in results.items() if k != "_sweep"):
        print(f"   {k:>20} {r['lr']:>8.0e} {r['reg']:>8.0e} "
              f"{r['rms_dist_cells']:>9.4f} {r['max_dist_cells']:>9.4f} "
              f"{r['triangles']:>8} {_settled(r['history']):>10.3e} "
              f"{('-' if r['closes'] is None else str(r['closes'])):>7}")
    return results


# --------------------------------------------------------------------------
# Arm C: the junction at K = 3
# --------------------------------------------------------------------------
def arm_c(resolutions=(16, 24, 32, 48), samples: int = 40000):
    print("\n" + "=" * 78)
    print("C. A triple junction, at K=3")
    print("=" * 78)
    print("   Standard double bubble: two unit-radius spheres a radius apart,")
    print("   three regions meeting along a circle. FlexiCubes takes one scalar")
    print("   field, so each region is extracted one-vs-rest and the results")
    print("   are asked whether they form a partition.")

    r = 0.55
    field = BubbleField(centres=[[r / 2, 0.0, 0.0], [-r / 2, 0.0, 0.0]],
                        radii=[r, r])

    def one_vs_rest(k):
        """Negative inside region k, as FlexiCubes requires.

        This is how a K-region problem is put to a single-field extractor, and
        it is what the paper means by the two-label reduction not surviving:
        each region is found without reference to the others. The field itself
        is not a strawman --- its zero set is exactly the true boundary of
        region k --- but it has a crease along the triple curve, which is
        where `max` switches argument.
        """
        def f(x):
            lg = field.logits(x.cpu())  # the field's parameters live on the CPU
            mine = lg[:, k]
            other = torch.cat([lg[:, :k], lg[:, k + 1:]], dim=1).max(dim=1).values
            return (other - mine).to(x.device)
        return f

    torch.manual_seed(0)
    q = (torch.rand((samples, 3), dtype=torch.float64) - 0.5) * 2.0
    truth = field.labels(q)
    truth_interior = (truth == 1) | (truth == 2)

    print("\n   Overlap is the fraction of the domain two regions both claim,")
    print("   which is zero for anything that is a partition. Swept over")
    print("   resolution, because a defect that refines away is a different")
    print("   complaint from one that does not.")
    print(f"\n   {'res':>4} {'method':>12} {'tris':>8} {'overlap':>10} "
          f"{'gap':>10} {'mislabel':>10}")

    rows = []
    for resolution in resolutions:
        # `ours, one-vs-rest` is the control that decides what the FlexiCubes
        # row is evidence of. The one-vs-rest field is creased along the triple
        # curve, and a dual vertex averaging a cell's crossings is exactly the
        # operation a crease upsets --- so a failure there could be the
        # decomposition or could be FlexiCubes meeting a crease. Running our own
        # extractor on the same two derived fields, scored the same way,
        # separates them: whatever it loses is the decomposition's fault.
        for method in ("ours", "ours, one-vs-rest", "flexicubes"):
            if method == "ours":
                surf = envelope3d.extract(field, resolution=resolution, box=BOX)
                meshes = {k: region_boundary(surf, k) for k in (1, 2)}
                ntri = int(surf.triangles.shape[0])
            elif method == "ours, one-vs-rest":
                meshes, ntri = {}, 0
                for k in (1, 2):
                    sub = envelope3d.extract(AnalyticField(one_vs_rest(k)),
                                             resolution=resolution, box=BOX,
                                             perturb=1e-9)
                    meshes[k] = (sub.vertices.detach(), sub.triangles.long())
                    ntri += int(sub.triangles.shape[0])
            else:
                meshes, ntri = {}, 0
                for k in (1, 2):
                    v, f, _ = run_flexicubes(one_vs_rest(k), resolution)
                    meshes[k] = (v.detach().double(), f.long())
                    ntri += int(f.shape[0])

            inside = {k: winding_inside(*meshes[k], q).cpu() for k in (1, 2)}
            both = inside[1] & inside[2]
            claimed = inside[1] | inside[2]
            pred = torch.zeros_like(truth)
            pred[inside[2]] = 2
            pred[inside[1]] = 1  # tie-break; `both` is counted on its own below
            row = {
                "resolution": resolution, "method": method, "triangles": ntri,
                "overlap_frac": float(both.double().mean()),
                "gap_frac": float((truth_interior & ~claimed).double().mean()),
                "mislabel_frac": float((pred != truth).double().mean()),
            }
            rows.append(row)
            print(f"   {resolution:>4} {method:>18} {ntri:>8} "
                  f"{row['overlap_frac']:>10.4%} {row['gap_frac']:>10.4%} "
                  f"{row['mislabel_frac']:>10.4%}")
        print()

    def arm(name):
        return [r for r in rows if r["method"] == name]

    ours, ovr, fcx = arm("ours"), arm("ours, one-vs-rest"), arm("flexicubes")
    print(f"   Ours, as one complex, overlaps by "
          f"{max(r['overlap_frac'] for r in ours):.4%} at worst over every")
    print("   resolution: the complex is one partition, so there is nothing to")
    print("   disagree with.")
    print(f"   Ours, run one-vs-rest, overlaps by "
          f"{ovr[0]['overlap_frac']:.4%} to {ovr[-1]['overlap_frac']:.4%}, and")
    print(f"   FlexiCubes one-vs-rest by {fcx[0]['overlap_frac']:.4%} to "
          f"{fcx[-1]['overlap_frac']:.4%}.")
    print("   The control decides what the FlexiCubes row means: whatever our own")
    print("   extractor also loses when run one-vs-rest is the decomposition's")
    print("   doing and not theirs.")
    return rows


def main():
    torch.set_default_dtype(torch.float64)
    print("=" * 78)
    print(f"FlexiCubes comparison. device={DEVICE}")
    print("=" * 78)

    a = arm_a()
    b = {t: arm_b(target=t) for t in ("ellipsoid", "torus")}
    c = arm_c()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"arm_a": a, "arm_b": b, "arm_c": c}, indent=1))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
