"""Validation for the differentiable multi-label extractor.

Run with `python validate.py`. Every check prints PASS or FAIL and the script
exits non-zero if anything fails, so it doubles as a regression test without
pulling in pytest.

The checks are, in order of how much they matter:

  A  topology invariants hold on four qualitatively different fields
  B  triple junctions match closed-form circumcentres
  C  interface vertices lie on the exact interface; the polyline converges
  D  autograd gradients match central finite differences
  E  autograd gradients match derivatives of an independent closed form

D and E are the load-bearing ones. D shows the implicit-function-theorem
reattachment is self-consistent; E shows it is *right*, by comparing against a
derivative computed through a completely separate expression that never touches
the extractor.
"""

from __future__ import annotations

import math
import sys

import torch

from extract2d import CROSSING, JUNCTION, check_topology, extract, junction_angles
from fields import (
    AnnulusField,
    MultiLabelField,
    NeuralMultiLabelField,
    PowerDiagramField,
    SectorField,
    circumcentre,
)

torch.set_default_dtype(torch.float64)

BOX = (-1.0, 1.0, -1.0, 1.0)
VORONOI_SITES = torch.tensor(
    [[-0.6, -0.5], [0.55, -0.6], [0.0, 0.65], [-0.75, 0.5], [0.8, 0.3]]
)

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------------
# A. Topology invariants
# --------------------------------------------------------------------------
def check_a_topology() -> None:
    print("\nA. Topological invariants (degree 2 interior / 1 boundary / 3 junction)")
    cases = {
        "power diagram (5 sites)": PowerDiagramField(VORONOI_SITES),
        "sector fan (K=3)": SectorField(centre=(0.05, -0.1), num_classes=3),
        "sector fan (K=6)": SectorField(centre=(0.0, 0.0), num_classes=6, phase=0.3),
        "annuli (K=4)": AnnulusField(),
        "neural field (K=3)": NeuralMultiLabelField(num_classes=3, seed=0),
        "neural field (K=5)": NeuralMultiLabelField(num_classes=5, seed=3),
    }
    for name, fld in cases.items():
        for res in (32, 64):
            mesh = extract(fld, resolution=res, box=BOX)
            topo = check_topology(mesh)
            detail = (
                f"res={res} V={mesh.vertices.shape[0]} S={mesh.segments.shape[0]} "
                f"junctions={topo['num_junctions']}"
            )
            if not topo["ok"]:
                detail += f" -> {topo}"
            record(f"{name} @ res {res}", topo["ok"], detail)


# --------------------------------------------------------------------------
# B. Junction exactness against closed-form circumcentres
# --------------------------------------------------------------------------
def check_b_junction_accuracy() -> None:
    print("\nB. Triple-junction position vs closed-form circumcentre")
    fld = PowerDiagramField(VORONOI_SITES)
    for res in (32, 64, 128):
        mesh = extract(fld, resolution=res, box=BOX)
        junctions = mesh.vertices[mesh.vertex_kind == JUNCTION].detach()
        if junctions.numel() == 0:
            record(f"res {res}", False, "no junctions found")
            continue
        # Reference set: circumcentres of every site triple that is a genuine
        # Voronoi vertex, i.e. whose circumcentre is actually claimed by one of
        # its own three sites.
        refs = []
        n = VORONOI_SITES.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    c = circumcentre(
                        fld.sites[i], fld.sites[j], fld.sites[k]
                    ).detach()
                    if fld.labels(c[None])[0].item() in (i, j, k):
                        refs.append(c)
        refs_t = torch.stack(refs)
        err = torch.cdist(junctions, refs_t).min(dim=1).values.max().item()
        ok = err < 1e-12 and junctions.shape[0] == refs_t.shape[0]
        record(
            f"res {res}",
            ok,
            f"found {junctions.shape[0]}/{refs_t.shape[0]} junctions, max error {err:.3e}",
        )


# --------------------------------------------------------------------------
# C. Interface exactness and polyline convergence
# --------------------------------------------------------------------------
def check_c_interface_accuracy() -> None:
    print("\nC. Interface accuracy on curved interfaces (annuli, exact circles)")
    fld = AnnulusField()
    target_radii = torch.stack(
        [fld.exact_interface_radius(k, k + 1) for k in range(fld.num_classes - 1)]
    ).detach()

    def radial_error(points: torch.Tensor) -> tuple[float, float]:
        r = points.norm(dim=-1)
        dev = (r[:, None] - target_radii[None, :]).abs().min(dim=1).values
        return dev.max().item(), dev.pow(2).mean().sqrt().item()

    vertex_errs, midpoint_rms, midpoint_max = {}, {}, {}
    for res in (32, 64, 128, 256):
        mesh = extract(fld, resolution=res, box=BOX)
        crossings = mesh.vertices[mesh.vertex_kind == CROSSING].detach()
        vertex_errs[res] = radial_error(crossings)[0]
        seg = mesh.vertices[mesh.segments].detach()
        midpoint_max[res], midpoint_rms[res] = radial_error(seg.mean(dim=1))

    # Extracted vertices solve f_i = f_j exactly, so they sit on the true
    # interface to solver tolerance regardless of grid resolution.
    worst_vertex = max(vertex_errs.values())
    record(
        "interface vertices lie on the exact circle",
        worst_vertex < 1e-12,
        f"max radial error {worst_vertex:.3e} over res {sorted(vertex_errs)}",
    )

    # The polyline joining them is a chord approximation, so its midpoints
    # deviate at second order in the cell size. The rate is measured on the RMS
    # deviation rather than the maximum: a chord's sagitta is L^2 / 8R, and L
    # varies between h and h*sqrt(2) depending on how the interface happens to
    # cut each triangle, so the maximum carries an O(1) alignment jitter that
    # does not average out and makes the per-step rate bounce around 2.
    # The rate is a least-squares slope of log(error) against log(h) over the
    # whole refinement range. Consecutive-pair ratios are not used because the
    # crossing sets at different resolutions are not nested, so each pair
    # carries sampling noise that a fitted slope averages out.
    resolutions = sorted(midpoint_rms)
    log_h = torch.tensor([-math.log2(r) for r in resolutions])
    log_e = torch.tensor([math.log2(midpoint_rms[r]) for r in resolutions])
    rate = float(
        ((log_h - log_h.mean()) * (log_e - log_e.mean())).sum()
        / ((log_h - log_h.mean()) ** 2).sum()
    )
    pairwise = [
        math.log2(midpoint_rms[a] / midpoint_rms[b])
        for a, b in zip(resolutions[:-1], resolutions[1:])
    ]
    detail = "rms " + ", ".join(f"h/{r}:{midpoint_rms[r]:.2e}" for r in resolutions)
    detail += f" | fitted rate {rate:.2f}"
    detail += " (pairwise " + ", ".join(f"{x:.2f}" for x in pairwise) + ")"
    record(
        "polyline midpoint error converges at ~2nd order",
        1.8 < rate < 2.2,
        detail,
    )


# --------------------------------------------------------------------------
# D. Gradients vs central finite differences
# --------------------------------------------------------------------------
def _scalar_functional(field: MultiLabelField, resolution: int, weights: torch.Tensor | None):
    mesh = extract(field, resolution=resolution, box=BOX)
    if weights is None:
        gen = torch.Generator().manual_seed(1234)
        weights = torch.randn(mesh.vertices.shape, generator=gen, dtype=mesh.vertices.dtype)
    if weights.shape != mesh.vertices.shape:
        return None, weights, mesh
    return (mesh.vertices * weights).sum(), weights, mesh


def _fd_gradient_check(name: str, field: MultiLabelField, resolution: int, num_probes: int = 10,
                       h: float = 1e-6, tol: float = 1e-6) -> None:
    value, weights, mesh = _scalar_functional(field, resolution, None)
    params = [p for p in field.parameters() if p.requires_grad]
    analytic = torch.autograd.grad(value, params, allow_unused=True)

    flat_params = [(pi, p) for pi, p in enumerate(params)]
    gen = torch.Generator().manual_seed(7)
    probes = []
    for pi, p in flat_params:
        n = p.numel()
        take = min(num_probes, n)
        idx = torch.randperm(n, generator=gen)[:take]
        probes.extend((pi, int(t)) for t in idx)

    worst_rel, worst_info, skipped = 0.0, "", 0
    for pi, flat_idx in probes:
        p = params[pi]
        original = p.data.reshape(-1)[flat_idx].item()

        def evaluate(offset: float):
            p.data.reshape(-1)[flat_idx] = original + offset
            with torch.no_grad():
                _, _, m = _scalar_functional(field, resolution, weights)
                if m.vertices.shape != weights.shape:
                    return None
                return (m.vertices * weights).sum().item()

        plus, minus = evaluate(h), evaluate(-h)
        p.data.reshape(-1)[flat_idx] = original
        if plus is None or minus is None:
            # A perturbation this small flipped a grid-node label, so the
            # discrete connectivity changed and the functional is genuinely
            # discontinuous there. Not a gradient error; report it separately.
            skipped += 1
            continue
        fd = (plus - minus) / (2 * h)
        ana = analytic[pi].reshape(-1)[flat_idx].item() if analytic[pi] is not None else 0.0
        rel = abs(fd - ana) / max(abs(fd), abs(ana), 1e-8)
        if rel > worst_rel:
            worst_rel = rel
            worst_info = f"param#{pi}[{flat_idx}] fd={fd:.8f} autograd={ana:.8f}"

    detail = f"res={resolution} probes={len(probes)} worst rel err {worst_rel:.3e}"
    if skipped:
        detail += f" ({skipped} skipped: connectivity change)"
    if worst_rel > tol:
        detail += f" | {worst_info}"
    record(name, worst_rel < tol, detail)


def check_d_finite_differences() -> None:
    print("\nD. Autograd vs central finite differences on the extracted vertices")
    _fd_gradient_check("power diagram, d/d(sites, weights)", PowerDiagramField(VORONOI_SITES), 48)
    _fd_gradient_check("annuli, d/d(radii)", AnnulusField(), 48)
    _fd_gradient_check("sector fan, d/d(centre)", SectorField(centre=(0.05, -0.1)), 48)
    _fd_gradient_check(
        "neural field, d/d(all MLP weights)",
        NeuralMultiLabelField(num_classes=3, hidden=32, num_frequencies=3, seed=0),
        40,
        num_probes=4,
    )


# --------------------------------------------------------------------------
# E. Gradients vs an independent closed form
# --------------------------------------------------------------------------
def check_e_independent_gradient() -> None:
    print("\nE. Junction Jacobian vs derivative of the closed-form circumcentre")
    fld = PowerDiagramField(VORONOI_SITES)
    mesh = extract(fld, resolution=96, box=BOX)
    junction_idx = (mesh.vertex_kind == JUNCTION).nonzero(as_tuple=True)[0]

    worst = 0.0
    detail_worst = ""
    for vi in junction_idx.tolist():
        point = mesh.vertices[vi]
        with torch.no_grad():
            # Identify which three sites this junction separates.
            logits = fld.logits(point.detach()[None])[0]
            triple = torch.topk(logits, 3).indices.sort().values.tolist()

        # Path 1: through the extractor's implicit-function-theorem gradient.
        jac_extract = torch.stack(
            [
                torch.autograd.grad(point[d], fld.sites, retain_graph=True)[0]
                for d in range(2)
            ]
        )

        # Path 2: through the closed-form circumcentre, which shares no code
        # with the extractor.
        i, j, k = triple
        cc = circumcentre(fld.sites[i], fld.sites[j], fld.sites[k])
        jac_closed = torch.stack(
            [
                torch.autograd.grad(cc[d], fld.sites, retain_graph=True)[0]
                for d in range(2)
            ]
        )

        rel = (jac_extract - jac_closed).abs().max().item() / max(
            jac_closed.abs().max().item(), 1e-12
        )
        if rel > worst:
            worst = rel
            detail_worst = f"junction separating sites {triple}"

    ok = junction_idx.numel() > 0 and worst < 1e-9
    record(
        "junction Jacobian matches closed form",
        ok,
        f"{junction_idx.numel()} junctions, worst rel err {worst:.3e} ({detail_worst})",
    )


# --------------------------------------------------------------------------
# F. Junction angle measurement against known geometry
# --------------------------------------------------------------------------
def check_f_junction_angles() -> None:
    print("\nF. Junction angle measurement vs known geometry")

    # A sector fan of K=3 has three straight interfaces exactly 120 apart.
    worst = 0.0
    for centre, phase in (((0.0, 0.0), 0.0), ((0.13, -0.21), 0.7), ((-0.3, 0.25), 2.1)):
        fld = SectorField(centre=centre, num_classes=3, phase=phase)
        mesh = extract(fld, resolution=64, box=BOX)
        angles = junction_angles(fld, mesh)
        if len(angles) != 1:
            record(f"sector fan at {centre}", False, f"expected 1 junction, got {len(angles)}")
            return
        worst = max(worst, max(abs(a - 120.0) for a in angles[0]))
    record(
        "sector fan angles are 120 deg",
        worst < 1e-8,
        f"worst deviation {worst:.3e} deg over 3 configurations",
    )

    # At a Voronoi vertex the angle subtended by site i's region is exactly
    # pi minus the Delaunay triangle's angle at site i.
    fld = PowerDiagramField(VORONOI_SITES)
    mesh = extract(fld, resolution=96, box=BOX)
    measured = junction_angles(fld, mesh)
    junction_pts = mesh.vertices[mesh.vertex_kind == JUNCTION].detach()
    worst_v = 0.0
    for n in range(junction_pts.shape[0]):
        i, j, k = mesh.junction_labels[n].tolist()
        p, q, r = fld.sites[i].detach(), fld.sites[j].detach(), fld.sites[k].detach()
        expected = []
        for u, v, w in ((p, q, r), (q, p, r), (r, p, q)):
            e1, e2 = v - u, w - u
            cos = (e1 @ e2) / (e1.norm() * e2.norm())
            expected.append(180.0 - math.degrees(math.acos(float(cos.clamp(-1, 1)))))
        worst_v = max(
            worst_v,
            max(abs(a - b) for a, b in zip(sorted(measured[n]), sorted(expected))),
        )
    record(
        "Voronoi vertex angles match Delaunay complement",
        junction_pts.shape[0] > 0 and worst_v < 1e-8,
        f"{junction_pts.shape[0]} junctions, worst deviation {worst_v:.3e} deg",
    )


def main() -> int:
    print("=" * 78)
    print("Differentiable multi-label boundary extraction - validation")
    print("=" * 78)
    check_a_topology()
    check_b_junction_accuracy()
    check_c_interface_accuracy()
    check_d_finite_differences()
    check_e_independent_gradient()
    check_f_junction_angles()

    failed = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 78)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
