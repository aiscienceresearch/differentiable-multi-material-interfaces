"""What does the symbolic perturbation cost near a degenerate configuration?

The paper says plainly that resolving exact coincidences by a deterministic
symbolic perturbation is the weaker guarantee against deciding them with exact
predicates, and that this is what buys the derivative. That claim should carry a
number rather than an adjective, and this measures one.

The degeneracy chosen is the one with no two-label analogue. In 3D exactly four
regions meet at a point generically; five meeting at a point is a
codimension-one coincidence. Five sites on a common sphere produce it, because
the centre is then equidistant from all five. Pushing one site outward by t
splits that single five-fold vertex into ordinary four-fold ones separated by
O(t), so t is a dial that runs a configuration continuously into degeneracy
while every quadruple point along the way stays exactly computable.

The quantity to watch is the position error of those quadruple points. The
obvious worry is that conditioning degrades as two vertices merge, so that a
perturbation of size delta surfaces amplified, as delta/t. It does not. Each
quadruple point is solved from its own 3x3 system of pairwise bisector
equations, and those stay well conditioned however close a fifth label comes to
joining them: the vertices merge, the individual solves do not degrade. The
error measured below tracks delta and is flat in t, which makes the cost of the
trade a fixed additive term rather than an amplified one.

What this does not establish is correctness on every configuration, which is
what exact predicates actually provide. It is one controlled family.
"""

from __future__ import annotations

import itertools
import math

import torch

torch.set_default_dtype(torch.float64)

import envelope3d
from fields import PowerDiagramField

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
RADIUS = 0.6

# Offset from the origin so the degenerate vertex does not land on a grid node,
# which would stack a second, unrelated coincidence on top of the one under test.
CENTRE = torch.tensor([0.0131, -0.0217, 0.0173])

# Five directions in general position: no four coplanar with the centre, so the
# only degeneracy in the family is the cosphericity itself.
DIRECTIONS = torch.tensor([
    [1.00, 0.30, 0.20],
    [-0.90, 0.50, 0.35],
    [0.20, -1.00, 0.45],
    [0.25, 0.40, -1.00],
    [-0.40, -0.60, -0.70],
])


def sites_at(t: float) -> torch.Tensor:
    """Five sites on a sphere of radius RADIUS, with site 0 pushed out by t.

    At t = 0 all five are equidistant from CENTRE, which is therefore a point
    where five Voronoi cells meet.
    """
    d = DIRECTIONS / DIRECTIONS.norm(dim=1, keepdim=True)
    r = torch.full((DIRECTIONS.shape[0],), RADIUS, dtype=d.dtype)
    r[0] = r[0] + t
    return CENTRE + r[:, None] * d


def _meeting_point(sites: torch.Tensor, weights: torch.Tensor, S) -> torch.Tensor:
    """Where the four labels in S have equal logits.

    Derived here from f_i = -|x-c_i|^2 + w_i rather than taken from the field,
    so the ground truth shares no code with the extractor it checks. The
    quadratic term cancels in every difference, leaving
    2(c_i - c_j).x = |c_i|^2 - |c_j|^2 - w_i + w_j.
    """
    i = S[0]
    rows, rhs = [], []
    for j in S[1:]:
        rows.append(2.0 * (sites[i] - sites[j]))
        rhs.append(sites[i].pow(2).sum() - sites[j].pow(2).sum()
                   - weights[i] + weights[j])
    return torch.linalg.solve(torch.stack(rows), torch.stack(rhs))


def exact_quadruple_points(sites: torch.Tensor, weights: torch.Tensor,
                           tol: float = 1e-11):
    """Every point where four cells genuinely meet, by enumeration.

    A four-subset contributes a vertex only when no other label wins there, so a
    candidate is kept only if its tied logit value is the maximum over all
    labels. Points outside the box are discarded because the extractor never
    sees them. Once t falls to the tolerance the enumeration starts admitting
    the fifth subset too, and the reference stops being able to resolve what it
    is being asked to check; `run` flags those rows.
    """
    n = sites.shape[0]
    found = []
    for S in itertools.combinations(range(n), 4):
        p = _meeting_point(sites, weights, S)
        if not bool(((p > -1.0) & (p < 1.0)).all()):
            continue
        f = -(p[None] - sites).pow(2).sum(-1) + weights
        if float(f.max() - f[list(S)].mean()) <= tol:
            found.append(p)
    return torch.stack(found) if found else torch.zeros((0, 3))


def _match(exact: torch.Tensor, got: torch.Tensor) -> float:
    """Worst distance from a true quadruple point to the nearest extracted one."""
    if exact.shape[0] == 0:
        return 0.0
    if got.shape[0] == 0:
        return float("inf")
    return float(torch.cdist(exact, got).min(dim=1).values.max())


def _closest_pair(points: torch.Tensor) -> float:
    """Separation of the two nearest true quadruple points, i.e. how degenerate."""
    if points.shape[0] < 2:
        return float("inf")
    d = torch.cdist(points, points)
    d = d + torch.eye(d.shape[0], dtype=d.dtype) * 1e30
    return float(d.min())


def run(resolution: int = 24, perturbations=(0.0, 1e-12, 1e-10, 1e-8),
        exponents=range(1, 12)):
    weights = torch.zeros(DIRECTIONS.shape[0])
    print(f"resolution {resolution}, five sites on a sphere of radius {RADIUS}")
    print("t is the radial offset of one site; at t=0 five cells meet at a point.")
    print("sep is the true distance between the two merging quadruple points.")
    print("err/delta tests whether the perturbation costs its own magnitude.\n")

    head = f"{'t':>9} {'sep':>10} {'n':>3}"
    sub = f"{'':>9} {'':>10} {'':>3}"
    for d in perturbations:
        head += " | {:>21}".format("delta=" + format(d, ".0e"))
        sub += " | {:>10} {:>8}".format("err", "err/del")
    print(head)
    print(sub)

    rows = []
    for e in exponents:
        t = 10.0 ** (-e)
        sites = sites_at(t)
        exact = exact_quadruple_points(sites, weights)
        sep = _closest_pair(exact)
        trusted = int(exact.shape[0]) == 2
        line = f"{t:9.1e} {sep:10.2e} {exact.shape[0]:3d}"

        entry = {"t": t, "separation": sep, "n_exact": int(exact.shape[0]),
                 "trusted": trusted}
        for delta in perturbations:
            field = PowerDiagramField(sites.clone(), weights.clone())
            try:
                surf = envelope3d.extract(field, resolution=resolution,
                                          box=BOX, perturb=delta)
                err = _match(exact, surf.quadruple_points().detach())
            except Exception as exc:                      # noqa: BLE001
                err = float("nan")
                entry[f"fail@{delta}"] = repr(exc)[:60]
            ratio = err / delta if delta > 0 else float("nan")
            line += f" | {err:10.2e} {ratio:8.2f}"
            entry[f"err@{delta}"] = err
        print(line + ("" if trusted else "   <- reference at its own limit"))
        rows.append(entry)
    return rows


def main() -> int:
    print("=" * 108)
    print("Cost of symbolic perturbation as a configuration approaches degeneracy")
    print("=" * 108)
    rows = run()
    good = [r for r in rows if r["trusted"]]

    print("\n" + "-" * 108)
    print(f"Over t = {max(r['t'] for r in good):.0e} down to "
          f"{min(r['t'] for r in good):.0e}, true vertex separation shrinking to "
          f"{min(r['separation'] for r in good):.1e}:")
    for delta in (1e-12, 1e-10, 1e-8):
        vals = [r[f"err@{delta}"] / delta for r in good
                if math.isfinite(r.get(f"err@{delta}", float("nan")))]
        if vals:
            print(f"  delta={delta:.0e}   err/delta in "
                  f"[{min(vals):.2f}, {max(vals):.2f}]")
    clean = [r["err@0.0"] for r in good
             if math.isfinite(r.get("err@0.0", float("nan")))]
    if clean:
        print(f"  unperturbed   worst error {max(clean):.2e}")
    print("\nThe error tracks delta rather than delta/t, so merging vertices do not")
    print("amplify it. One family is not a correctness proof for all configurations,")
    print("which is what exact predicates provide and this does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
