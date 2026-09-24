"""Is the 120 degree partition actually the minimiser of this energy?

The Plateau experiment optimises a free multi-label field and the junction
angles move away from 120 degrees. Before blaming the optimiser, check the
premise, because there are only two possibilities:

  (a) the exactly-120 family contains a shorter equal-area partition than the
      free field found, in which case the free optimisation is in a poor local
      minimum and 120 degrees is the right target; or
  (b) it does not, in which case some non-120 configuration genuinely has less
      interface length at equal areas and the expectation is wrong.

The family of exactly-120 Y partitions is available in closed form: for a
sector field f_k(x) = <x - p, d_k> with the three directions 120 degrees apart,
the argmax regions are 120-degree wedges and the interfaces are three straight
rays from p, exactly 120 degrees apart. Letting p and the rotation phase vary
sweeps the whole family, so optimising over just those three numbers with the
same energy and the same extractor answers the question directly.

Run `python plateau_reference.py`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from extract2d import check_topology, extract, junction_angles
from fields import MultiLabelField

torch.set_default_dtype(torch.float64)
BOX = (-1.0, 1.0, -1.0, 1.0)
FAST = dict(bisection_steps=30, newton_steps=6, junction_newton_steps=20)


class RotatableSectorField(MultiLabelField):
    """Exactly-120-degree Y partition with learnable centre and rotation."""

    def __init__(self, centre=(0.0, 0.0), phase: float = 0.0):
        super().__init__()
        self.centre = nn.Parameter(torch.as_tensor(centre, dtype=torch.get_default_dtype()))
        self.phase = nn.Parameter(torch.tensor(float(phase)))
        self.num_classes = 3

    def logits(self, x: Tensor) -> Tensor:
        k = torch.arange(3, dtype=self.phase.dtype, device=self.phase.device)
        angles = self.phase + k * (2 * math.pi / 3)
        dirs = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (3, 2)
        return (x - self.centre) @ dirs.T


def soft_area_fractions(field: MultiLabelField, resolution: int = 160,
                        temperature: float = 0.02) -> Tensor:
    xs = torch.linspace(BOX[0], BOX[1], resolution)
    ys = torch.linspace(BOX[2], BOX[3], resolution)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    return torch.softmax(field.logits(pts) / temperature, dim=-1).mean(dim=0)


def hard_area_fractions(field: MultiLabelField, resolution: int = 400) -> list[float]:
    xs = torch.linspace(BOX[0], BOX[1], resolution)
    ys = torch.linspace(BOX[2], BOX[3], resolution)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    with torch.no_grad():
        labels = field.labels(pts)
    counts = torch.bincount(labels, minlength=field.num_classes).to(torch.get_default_dtype())
    return (counts / labels.numel()).tolist()


def optimise(field: MultiLabelField, steps: int = 700, lr: float = 5e-3,
             resolution: int = 72, weights=(50.0, 4000.0), label: str = "") -> dict:
    target = 1.0 / field.num_classes
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    w0, w1 = weights
    for step in range(steps):
        opt.zero_grad()
        w = w0 * (w1 / w0) ** (step / max(steps - 1, 1))
        mesh = extract(field, resolution=resolution, box=BOX, **FAST)
        length = mesh.segment_lengths().sum()
        pen = ((soft_area_fractions(field) - target) ** 2).sum()
        (length + w * pen).backward()
        opt.step()
        sched.step()
    mesh = extract(field, resolution=resolution, box=BOX, **FAST)
    angles = junction_angles(field, mesh)
    return {
        "label": label,
        "length": float(mesh.segment_lengths().sum().detach()),
        "areas": hard_area_fractions(field),
        "angles": angles[0] if angles else None,
        "topology_ok": check_topology(mesh)["ok"],
    }


def main() -> None:
    print("=" * 78)
    print("Reference: best equal-area partition inside the exactly-120-degree family")
    print("=" * 78)

    # Several starts, since the family is small but the energy is not convex.
    best = None
    for centre in ((0.0, 0.0), (0.25, -0.2), (-0.3, 0.15), (0.1, 0.4)):
        for phase in (0.0, 0.4, 0.9, 1.5):
            field = RotatableSectorField(centre=centre, phase=phase)
            out = optimise(field, label=f"centre={centre} phase={phase:.1f}")
            area_err = max(abs(a - 1 / 3) for a in out["areas"])
            if area_err > 2e-3 or not out["topology_ok"]:
                continue
            if best is None or out["length"] < best["length"]:
                best = out
    if best is None:
        print("no feasible equal-area configuration found in the 120-degree family")
        return

    print(f"\nbest 120-degree equal-area partition:")
    print(f"  from {best['label']}")
    print(f"  interface length      {best['length']:.4f}")
    print(f"  area fractions        {[round(a, 4) for a in best['areas']]}")
    print(f"  angles (sanity check) {[round(a, 2) for a in best['angles']]}")

    free_field_length = 3.2648  # neural field result from demo_optimize.py
    print(f"\nfree neural field reached {free_field_length:.4f} at angles ~[108.5, 113.4, 138.1]")
    if best["length"] < free_field_length - 1e-3:
        verdict = (
            "The 120-degree family is SHORTER, so the free optimisation is in a worse\n"
            "  local minimum and 120 degrees remains the right target on the square."
        )
    elif best["length"] > free_field_length + 1e-3:
        verdict = (
            "The 120-degree straight-arm family is LONGER, so it is not the minimiser\n"
            "  and cannot serve as the reference on a square domain.\n\n"
            "  The reason is the wall boundary condition. A minimiser must meet the\n"
            "  domain boundary at 90 degrees, and three straight arms 120 degrees apart\n"
            "  cannot all be perpendicular to the sides of a square, so the true\n"
            "  minimiser has curved arms. This rules out the straight-arm family as a\n"
            "  reference; it does not establish what the true junction angle is.\n\n"
            "  Conclusion: test the 120-degree law on a DISK instead, where radial arms\n"
            "  are perpendicular to the boundary circle, so the minimiser is exactly\n"
            "  three straight radial arms at 120 degrees with total length 3R. See the\n"
            "  disk experiment in demo_optimize.py."
        )
    else:
        verdict = "The two are within 1e-3; the comparison is inconclusive."
    print(f"\nverdict:\n  {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
