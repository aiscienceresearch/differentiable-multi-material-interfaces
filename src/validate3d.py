"""Validation for 3D differentiable multi-label extraction.

Run with `python validate3d.py`. Every check prints PASS or FAIL and the script
exits non-zero if anything fails.

Two extractors are under test and the comparison between them is the point.

`extract3d` reads each cell's structure off its corner labels, which is what
marching-tets-style methods do. In 2D that is almost always right, because the
triple set is isolated points. In 3D the triple set is a curve, and a face can
show three distinct corner labels while the three regions never meet inside it.
Section H measures how often: about half of all three-label faces, at a rate
that does not improve under refinement.

`envelope3d` drops the assumption. It interpolates the logits linearly and
extracts the argmax partition of the interpolant exactly, which needs no case
analysis because every region is convex. Section G shows what that buys: on a
power diagram, where every interface is planar and a correct extractor should
be exact at any resolution, the corner-label version converges at first order
while the arrangement version is exact to machine precision.

Sections A-F verify the arrangement extractor itself: exact vertices, closed
and orientable surfaces, second-order convergence on curved interfaces,
gradients against finite differences, and Plateau's law at triple curves.
"""

from __future__ import annotations

import math
import sys

import torch

import envelope3d
import extract3d
import voronoi_exact
from extract3d import (CROSSING, QUADRUPLE, TRIPLE, certify, check_topology,
                       triple_curve_dihedrals)
from fields import (AnnulusField, MultiLabelField, NeuralMultiLabelField,
                    PowerDiagramField, SectorField, simplex_directions)

torch.set_default_dtype(torch.float64)

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
SITES = torch.tensor([
    [0.35, 0.20, 0.10], [-0.30, 0.25, -0.15], [0.05, -0.40, 0.22],
    [-0.15, -0.10, 0.45], [0.10, 0.05, -0.48],
])

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def cases() -> dict[str, MultiLabelField]:
    return {
        "power diagram (K=5)": PowerDiagramField(SITES),
        "sector fan (K=4)": SectorField(centre=(0.05, -0.02, 0.03),
                                        directions=simplex_directions(3)),
        "spherical shells (K=4)": AnnulusField(dim=3),
        "neural field (K=4)": NeuralMultiLabelField(num_classes=4, dim=3, seed=0),
        "neural field (K=6)": NeuralMultiLabelField(num_classes=6, hidden=48, dim=3, seed=2),
    }


def _enclosing_cell_field() -> PowerDiagramField:
    """A power diagram whose cell 0 is a bounded polytope well inside the box."""
    torch.manual_seed(3)
    shell = torch.tensor([
        [1.0, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [1.4, 0, 0], [-1.4, 0, 0], [0, 1.4, 0],
        [0, -1.4, 0], [0, 0, 1.4], [0, 0, -1.4],
    ]) * 0.55
    shell = shell + 0.06 * torch.randn_like(shell)
    return PowerDiagramField(torch.cat([torch.tensor([[0.02, -0.01, 0.03]]), shell]))


# --------------------------------------------------------------------------
# A. Vertices solve the equations that define them
# --------------------------------------------------------------------------
PERTURB = 1e-9


def check_a_vertex_exactness() -> None:
    print("\nA. Vertices satisfy their defining equal-logit equations")
    print("   Recomputed from the interpolant, independently of the solve.")
    for name, fld in cases().items():
        surf = envelope3d.extract(fld, resolution=14, box=BOX, perturb=PERTURB)
        interpolated = _interpolated_logits_at(fld, surf, PERTURB)
        worst = 0.0
        for kind, count in ((CROSSING, 2), (TRIPLE, 3), (QUADRUPLE, 4)):
            sel = surf.vertex_kind == kind
            if not bool(sel.any()):
                continue
            vals = interpolated[sel].gather(1, surf.vertex_labels[sel][:, :count])
            worst = max(worst, float((vals.max(dim=1).values - vals.min(dim=1).values).max()))
        record(name, worst < 1e-12,
               f"{surf.vertices.shape[0]} vertices, max spread among tied logits {worst:.3e}")


def _interpolated_logits_at(fld: MultiLabelField, surf, perturb: float = 0.0) -> torch.Tensor:
    """Evaluate the linear interpolant at each extracted vertex.

    Each vertex lies in a known cell, so its barycentric coordinates give the
    interpolated logits directly. Done by nearest-cell search over tets would be
    slow; instead the vertex already records which labels should tie, and the
    interpolant is reconstructed from the enclosing tet found by rounding.
    """
    with torch.no_grad():
        node_logits = envelope3d._perturb(fld.logits(surf.nodes), perturb)
        v = surf.vertices.detach()
        # Locate each vertex in the grid by barycentric test against the tets of
        # its containing cube, which is cheap and exact for a uniform grid.
        tets = surf.tets
        corners = surf.nodes[tets]
        # Solve barycentric coordinates for every vertex against every candidate
        # tet is too costly, so use the cube index and the 6 tets inside it.
        res = surf.diagnostics["resolution"]
        step = 2.0 / res
        cell = ((v + 1.0) / step).floor().clamp(0, res - 1).long()
        cube = (cell[:, 2] * res + cell[:, 1]) * res + cell[:, 0]
        num_cubes = res ** 3
        best = torch.full((v.shape[0],), -1, dtype=torch.long)
        best_bary = torch.zeros((v.shape[0], 4), dtype=v.dtype)
        for k in range(6):
            cand = cube + k * num_cubes
            p = corners[cand]
            edges = (p[:, 1:, :] - p[:, 0:1, :]).transpose(1, 2)
            coords = torch.linalg.solve(edges, (v - p[:, 0, :])[..., None]).squeeze(-1)
            bary = torch.cat([1.0 - coords.sum(1, keepdim=True), coords], dim=1)
            ok = (bary >= -1e-9).all(dim=1) & (best < 0)
            best = torch.where(ok, cand, best)
            best_bary = torch.where(ok[:, None], bary, best_bary)
        return torch.einsum("mc,mck->mk", best_bary, node_logits[tets[best.clamp_min(0)]])


# --------------------------------------------------------------------------
# B. The extracted complex is closed and orientable
# --------------------------------------------------------------------------
def check_b_topology() -> None:
    print("\nB. Closure and orientability of the extracted complex")
    print("   Every triangle side is used twice with opposite winding, except on"
          " triple curves\n   (used once by each of three patches) and at the"
          " domain boundary.")
    for name, fld in cases().items():
        for res in (12, 20):
            surf = envelope3d.extract(fld, resolution=res, box=BOX)
            t = check_topology(surf)
            d = surf.diagnostics
            record(f"{name} @ res {res}", t["ok"],
                   f"{t['num_triangles']} triangles, {t['num_sides']} sides, "
                   f"unpaired {t['unpaired_sides']}, "
                   f"non-orientable {t['inconsistently_oriented_sides']}, "
                   f"thin patches {d['patches_with_too_few_corners']}")


# --------------------------------------------------------------------------
# C. Features the corner-label analysis cannot express
# --------------------------------------------------------------------------
def check_c_multiplicity() -> None:
    print("\nC. Features per cell that a one-per-cell case analysis cannot represent")
    total_edges = total_faces = 0
    for name, fld in cases().items():
        surf = envelope3d.extract(fld, resolution=20, box=BOX)
        d = surf.diagnostics
        total_edges += d["edges_with_multiple_crossings"]
        total_faces += d["faces_with_multiple_triple_points"]
        print(f"   {name:24s} edges with 2+ crossings {d['edges_with_multiple_crossings']:5d}, "
              f"faces with 2+ triple points {d['faces_with_multiple_triple_points']:4d}")
    record("multi-feature cells are found and represented", total_edges > 0,
           f"{total_edges} multi-crossing edges, {total_faces} multi-triple-point faces "
           f"across all fields")


# --------------------------------------------------------------------------
# D. Curved interfaces converge
# --------------------------------------------------------------------------
def check_d_convergence() -> None:
    print("\nD. Curved interfaces (spherical shells, exact radii)")
    fld = AnnulusField(dim=3)
    r = float(0.5 * (fld.radii[0] + fld.radii[1]).detach())
    exact_area = 4 * math.pi * r * r
    errors = []
    for res in (16, 24, 32, 48):
        surf = envelope3d.extract(fld, resolution=res, box=BOX)
        mask = (surf.triangle_labels == torch.tensor([0, 1])).all(dim=1)
        area = float(surf.triangle_areas()[mask].sum().detach())
        errors.append((res, abs(area - exact_area) / exact_area))
    rate = math.log(errors[0][1] / errors[-1][1]) / math.log(errors[-1][0] / errors[0][0])
    record("sphere area converges at ~2nd order", rate > 1.7,
           ", ".join(f"res {a}: {b:.2e}" for a, b in errors) + f" | fitted rate {rate:.2f}")


# --------------------------------------------------------------------------
# E. Gradients against central finite differences
# --------------------------------------------------------------------------
def _fd_gradient_check(name: str, fld: MultiLabelField, res: int, probe_limit: int = 12) -> None:
    """Compare autograd against central differences of a permutation-invariant loss.

    The loss must not depend on vertex order, because perturbing a parameter can
    reorder the extracted vertices without moving the surface. Probes where the
    vertex count changes are genuine topology changes, where a difference
    quotient means nothing, so they are skipped and counted.
    """
    def loss_of(field: MultiLabelField):
        surf = envelope3d.extract(field, resolution=res, box=BOX)
        v = surf.vertices
        w = torch.tensor([0.83, -0.41, 0.62], dtype=v.dtype)
        return (v @ w).sum() + v.pow(2).sum() + v.prod(dim=-1).sum(), v.shape[0]

    loss, base = loss_of(fld)
    loss.backward()
    worst, checked, skipped = 0.0, 0, 0
    eps = 1e-6
    for p in [q for q in fld.parameters() if q.grad is not None]:
        flat = p.detach().reshape(-1)
        for idx in range(0, flat.numel(), max(1, flat.numel() // probe_limit)):
            original = flat[idx].item()
            with torch.no_grad():
                flat[idx] = original + eps
            plus, n_plus = loss_of(fld)
            with torch.no_grad():
                flat[idx] = original - eps
            minus, n_minus = loss_of(fld)
            with torch.no_grad():
                flat[idx] = original
            if n_plus != base or n_minus != base:
                skipped += 1
                continue
            fd = (float(plus.detach()) - float(minus.detach())) / (2 * eps)
            worst = max(worst, abs(fd - float(p.grad.reshape(-1)[idx])) / max(1.0, abs(fd)))
            checked += 1
    record(name, worst < 1e-5 and checked > 0,
           f"res={res} probes={checked} (skipped {skipped}) worst rel err {worst:.3e}")


def check_e_gradients() -> None:
    print("\nE. Autograd vs central finite differences on the extracted vertices")
    _fd_gradient_check("power diagram, d/d(sites, weights)", PowerDiagramField(SITES), 12)
    _fd_gradient_check("spherical shells, d/d(radii)", AnnulusField(dim=3), 12)
    _fd_gradient_check(
        "sector fan, d/d(centre)",
        SectorField(centre=(0.05, -0.02, 0.03), directions=simplex_directions(3)), 12)
    _fd_gradient_check(
        "neural field, d/d(all MLP weights)",
        NeuralMultiLabelField(num_classes=4, hidden=32, num_frequencies=3, dim=3, seed=1), 10, 10)


# --------------------------------------------------------------------------
# F. Plateau's law at triple curves
# --------------------------------------------------------------------------
def check_f_plateau() -> None:
    print("\nF. Dihedral angles where three patches meet")
    fld = SectorField(centre=(0.05, -0.02, 0.03), directions=simplex_directions(3))
    surf = envelope3d.extract(fld, resolution=20, box=BOX)
    angles = triple_curve_dihedrals(fld, surf)
    worst = max(max(abs(a - 120.0) for a in g) for g in angles)
    record("sector-fan triple curves meet at 120 deg", worst < 1e-8,
           f"{len(angles)} segments, worst deviation {worst:.3e} deg")


# --------------------------------------------------------------------------
# G. Head to head against exactly known ground truth
# --------------------------------------------------------------------------
def check_g_exact_cell() -> None:
    print("\nG. Accuracy vs an exactly known power-diagram cell")
    print("   Every interface here is planar and every triple curve straight, so"
          " a correct\n   extractor is exact at any resolution.")
    fld = _enclosing_cell_field()
    exact = voronoi_exact.cell_geometry(fld.sites.detach(), fld.weights.detach(), 0, BOX)
    ev = float(exact["volume"])
    ea = sum(float(a) for a in exact["facet_area"].values())
    print(f"   exact cell 0: volume {ev:.12f}, area {ea:.12f}, "
          f"{len(exact['facet_area'])} facets, {exact['vertices'].shape[0]} corners")

    # The perturbation is reported separately from the method: this
    # configuration is already generic, so switching it off should leave nothing
    # but floating-point noise, and switching it on should cost exactly its own
    # magnitude and nothing more.
    runs = (("corner-label case analysis", extract3d, {}),
            ("exact arrangement", envelope3d, {"perturb": 0.0}),
            (f"exact arrangement, perturbed by {PERTURB:.0e}", envelope3d,
             {"perturb": PERTURB}))
    worst = {}
    baseline = []
    for label, module, kwargs in runs:
        print(f"   {label}")
        for res in (8, 16, 32, 48):
            surf = module.extract(fld, resolution=res, box=BOX, **kwargs)
            vol = abs(float(surf.enclosed_volume(0).detach()) - ev) / ev
            involved = (surf.triangle_labels == 0).any(dim=1)
            area = abs(float(surf.triangle_areas()[involved].sum().detach()) - ea) / ea
            print(f"     res {res:3d}  volume err {vol:.3e}  area err {area:.3e}")
            worst[label] = max(worst.get(label, 0.0), vol, area)
            if module is extract3d:
                baseline.append(vol)
    rate = math.log(baseline[0] / baseline[-1]) / math.log(48 / 8)
    print(f"   corner-label convergence rate {rate:.2f}; the arrangement needs none")
    clean, jittered = runs[1][0], runs[2][0]
    record("arrangement extractor is exact on a power diagram", worst[clean] < 1e-12,
           f"worst volume/area error {worst[clean]:.3e} over res 8-48")
    record("the symbolic perturbation costs only its own magnitude",
           worst[jittered] < 100 * PERTURB,
           f"worst volume/area error {worst[jittered]:.3e}, "
           f"{worst[jittered] / PERTURB:.0f}x the perturbation")


# --------------------------------------------------------------------------
# H. Why the corner-label analysis cannot be repaired by refining
# --------------------------------------------------------------------------
def check_h_certificate() -> None:
    print("\nH. How often the corner-label case analysis is wrong")
    fld = PowerDiagramField(SITES)
    rates = []
    for res in (16, 32, 48):
        c = certify(fld, resolution=res, box=BOX)
        rate = 100 * c["faces_without_interior_triple_point"] / max(c["three_label_faces"], 1)
        rates.append(rate)
        print(f"   res {res:3d}  three-label faces with no interior triple point "
              f"{c['faces_without_interior_triple_point']:4d}/{c['three_label_faces']:4d} "
              f"= {rate:5.1f}%  |  edges needing 2+ crossings "
              f"{100 * c['edges_with_extra_label'] / max(c['mixed_edges'], 1):5.2f}%")
    record("the corner-label failure rate does not shrink under refinement",
           min(rates) > 20.0,
           f"{rates[0]:.0f}% -> {rates[-1]:.0f}% from res 16 to 48, so refinement alone"
           f" cannot fix it")


def check_i_entity_keys() -> None:
    """Shared faces must be identified the same way on grids of any size.

    Faces were keyed by packing their three node indices in base `num_nodes`,
    which overflows int64 once `num_nodes**3` does -- at $2^{21}$ nodes, a
    $127^3$ grid. Past that, distinct faces collided, neighbouring tetrahedra
    disagreed about which face they shared, and the surface came out with
    thousands of unpaired sides. Extracting at that size costs half a minute,
    so the check instead runs both code paths on the same tetrahedra and
    requires them to agree, by telling the slow one that the grid is enormous.
    """
    print("\nI. Entity keys survive grids past the packing ceiling")
    nodes, tets = envelope3d.build_tetrahedral_grid(6, BOX)
    faces = tets[:, torch.tensor(extract3d.FACE_TABLE)]
    n = nodes.shape[0]

    fast = extract3d._unique_entities(faces, n)
    huge = 10 ** 7                      # 10^21 does not fit, forcing the fold
    slow = extract3d._unique_entities(faces, huge)

    def canonical(entities, ids, counts):
        key = {}
        for row, c in zip(entities.tolist(), counts.tolist()):
            key[tuple(sorted(row))] = int(c)
        return key, ids

    fast_map, fast_ids = canonical(fast[0], fast[1], fast[2])
    slow_map, slow_ids = canonical(slow[0], slow[1], slow[2])
    same_partition = bool((fast_ids.reshape(-1).unique(return_inverse=True)[1]
                           == slow_ids.reshape(-1).unique(return_inverse=True)[1]).all())

    record("both entity-key paths find the same shared faces",
           fast_map == slow_map and same_partition,
           f"{len(fast_map)} faces on a 6^3 grid, {extract3d._keys_fit(n, 3)} "
           f"vs {extract3d._keys_fit(huge, 3)} for the packed key")
    record("the packing ceiling is detected, not assumed",
           extract3d._keys_fit(2 ** 21, 3) and not extract3d._keys_fit(2 ** 21 + 1, 3),
           "2^21 nodes is the last cube that fits in int64")


def check_j_random_fields() -> None:
    """Closure on fields nobody chose.

    Checks A--I run on five fields selected to have closed-form answers. That
    leaves open whether closure survives on fields picked at random, so this
    sweeps forty of them and fails if any one of them fails.

    The neural seeds start at 100 to keep the sweep disjoint from the fields
    used elsewhere: seed 2 at K=6, hidden 48 is the `cases()` entry "neural
    field (K=6)" exactly, and drawing it here would make the sweep a retest.
    """
    print("\nJ. Closure and orientability on randomly sampled fields")
    print("   Forty fields drawn without inspection: eight seeds each of the"
          " neural field at\n   K=4,6 and of random power diagrams at K=4,6,9.")
    worst = []
    for k in (4, 6):
        for seed in range(100, 108):
            fld = NeuralMultiLabelField(num_classes=k, hidden=48, dim=3, seed=seed)
            t = check_topology(envelope3d.extract(fld, resolution=16, box=BOX))
            worst.append((f"neural K={k} seed={seed}", t))
    for k in (4, 6, 9):
        for seed in range(8):
            gen = torch.Generator().manual_seed(seed)
            sites = (torch.rand((k, 3), generator=gen, dtype=torch.float64) - 0.5) * 1.2
            t = check_topology(envelope3d.extract(PowerDiagramField(sites),
                                                  resolution=16, box=BOX))
            worst.append((f"power K={k} seed={seed}", t))

    bad = [name for name, t in worst if not t["ok"]]
    # An empty surface reports only `ok`, so default the side counts rather
    # than raising and losing the check.
    unpaired = sum(t.get("unpaired_sides", 0) for _, t in worst)
    misoriented = sum(t.get("inconsistently_oriented_sides", 0) for _, t in worst)
    record("closure and orientation hold on all 40 unseen fields", not bad,
           f"{len(worst)} fields, {unpaired} unpaired sides, "
           f"{misoriented} inconsistently oriented"
           + (f", failures: {bad}" if bad else ""))


def main() -> int:
    print("=" * 78)
    print("Differentiable multi-label surface extraction in 3D - validation")
    print("=" * 78)
    check_a_vertex_exactness()
    check_b_topology()
    check_c_multiplicity()
    check_d_convergence()
    check_e_gradients()
    check_f_plateau()
    check_g_exact_cell()
    check_h_certificate()
    check_i_entity_keys()
    check_j_random_fields()

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 78)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
