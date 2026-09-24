"""Optimising a multi-label field through the extractor.

Two experiments, both driving the field parameters with a loss that is a
function of the *extracted mesh* rather than of the field values. If gradients
did not flow through extraction, neither could work.

  1. Target matching. Fit a randomly initialised neural field so its extracted
     multi-material geometry matches a target partition, using a Chamfer
     distance on interface vertices plus a triple-junction position term.

  2. Plateau's law. Minimise total interface length subject to equal-area
     constraints. At any stationary point the three unit tangents at a triple
     junction must sum to zero, so the angles must be 120 degrees. That is a
     quantitative, theory-predicted number the optimisation either reproduces
     or does not, which makes it a real test rather than a demonstration.

Run `python demo_optimize.py`. Figures are written to ../figures.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection

from extract2d import CROSSING, JUNCTION, check_topology, extract, junction_angles
from fields import MultiLabelField, NeuralMultiLabelField, PowerDiagramField, SectorField

torch.set_default_dtype(torch.float64)

BOX = (-1.0, 1.0, -1.0, 1.0)
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
# Fewer bisection steps than the validation default: Newton polishing carries
# the accuracy, and this loop re-extracts on every optimiser step.
FAST = dict(bisection_steps=30, newton_steps=6, junction_newton_steps=20)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_mesh(ax, field: MultiLabelField, mesh, title: str, render_res: int = 300) -> None:
    xs = torch.linspace(BOX[0], BOX[1], render_res)
    ys = torch.linspace(BOX[2], BOX[3], render_res)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    with torch.no_grad():
        labels = field.labels(pts).reshape(render_res, render_res)
    ax.imshow(
        labels, origin="lower", extent=BOX, cmap="Pastel1", interpolation="nearest",
        vmin=0, vmax=8,
    )
    seg = mesh.vertices[mesh.segments].detach().numpy()
    ax.add_collection(LineCollection(seg, colors="black", linewidths=1.1))
    j = mesh.vertices[mesh.vertex_kind == JUNCTION].detach()
    if j.numel():
        ax.plot(j[:, 0], j[:, 1], "o", color="crimson", markersize=7, zorder=5)
    ax.set_xlim(BOX[0], BOX[1])
    ax.set_ylim(BOX[2], BOX[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(a, b)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def clipped_interface_length(mesh, radius: float) -> torch.Tensor:
    """Total interface length inside the disk of the given radius, exactly clipped.

    Each segment is intersected with the circle analytically rather than being
    included or dropped whole, so the energy is the true length inside the
    domain and stays differentiable in the vertex positions.
    """
    v = mesh.vertices[mesh.segments]
    p0, d = v[:, 0], v[:, 1] - v[:, 0]
    a = (d * d).sum(-1)
    b = 2.0 * (p0 * d).sum(-1)
    c = (p0 * p0).sum(-1) - radius**2

    disc = b * b - 4.0 * a * c
    # disc <= 0 means the whole line misses the disk, so the segment contributes
    # nothing; clamping keeps the sqrt finite and the interval empty.
    root = torch.sqrt(disc.clamp_min(0.0))
    a_safe = a.clamp_min(1e-30)
    s_lo = ((-b - root) / (2.0 * a_safe)).clamp(0.0, 1.0)
    s_hi = ((-b + root) / (2.0 * a_safe)).clamp(0.0, 1.0)
    inside = (s_hi - s_lo).clamp_min(0.0) * torch.where(disc > 0, 1.0, 0.0)
    return (inside * a_safe.sqrt()).sum()


def disk_area_fractions(field: MultiLabelField, radius: float, resolution: int = 200,
                        temperature: float = 0.02) -> torch.Tensor:
    """Differentiable area fractions restricted to the disk.

    Areas are a volume integral, so they come from the field on a fine grid
    rather than from the extracted mesh; only the interface energy needs the
    extractor. `temperature` -> 0 recovers the hard partition.
    """
    xs = torch.linspace(-radius, radius, resolution)
    gy, gx = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    pts = pts[pts.norm(dim=-1) <= radius]
    return torch.softmax(field.logits(pts) / temperature, dim=-1).mean(dim=0)


def hard_disk_area_fractions(field: MultiLabelField, radius: float,
                             resolution: int = 500) -> list[float]:
    """Area fractions of the actual argmax partition inside the disk.

    `disk_area_fractions` is what the optimiser sees; this is what it means.
    """
    xs = torch.linspace(-radius, radius, resolution)
    gy, gx = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    pts = pts[pts.norm(dim=-1) <= radius]
    with torch.no_grad():
        labels = field.labels(pts)
    counts = torch.bincount(labels, minlength=field.num_classes).to(torch.get_default_dtype())
    return (counts / labels.numel()).tolist()





# --------------------------------------------------------------------------
# Experiment 1: target matching
# --------------------------------------------------------------------------
def experiment_target_matching(steps: int = 260, resolution: int = 56) -> dict:
    print("\n" + "=" * 78)
    print("1. Fitting a neural field to a target partition, loss on extracted mesh only")
    print("=" * 78)

    target_field = PowerDiagramField(torch.tensor([[-0.55, -0.35], [0.6, -0.4], [0.0, 0.6]]))
    target_mesh = extract(target_field, resolution=192, box=BOX, **FAST)
    target_points = target_mesh.vertices[target_mesh.vertex_kind == CROSSING].detach()
    target_junction = target_mesh.vertices[target_mesh.vertex_kind == JUNCTION].detach()
    print(f"target: {target_points.shape[0]} interface vertices, "
          f"{target_junction.shape[0]} junction at {target_junction[0].tolist()}")

    model = NeuralMultiLabelField(num_classes=3, hidden=64, num_frequencies=3, seed=5)
    # Warm start: make the initial partition a three-region fan so that a
    # triple junction exists from the outset. This fits logits directly and is
    # the only stage that does not go through the extractor.
    warm_target = SectorField(centre=(0.0, 0.0), num_classes=3)
    warm_opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    xs = torch.rand(4096, 2) * 2 - 1
    for _ in range(400):
        warm_opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model.logits(xs), warm_target.logits(xs))
        loss.backward()
        warm_opt.step()

    init_mesh = extract(model, resolution=resolution, box=BOX, **FAST)
    print(f"after warm start: {init_mesh.diagnostics['num_junctions']} junction(s)")

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    # Re-extracting each step makes the loss piecewise smooth: it jumps slightly
    # whenever a grid node changes label. Cosine decay is what stops the run
    # from bouncing between connectivity states near the optimum.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    history = []
    for step in range(steps):
        opt.zero_grad()
        mesh = extract(model, resolution=resolution, box=BOX, **FAST)
        crossings = mesh.vertices[mesh.vertex_kind == CROSSING]
        loss = chamfer(crossings, target_points)
        junctions = mesh.vertices[mesh.vertex_kind == JUNCTION]
        if junctions.shape[0] and target_junction.shape[0]:
            loss = loss + 2.0 * torch.cdist(junctions, target_junction).min(dim=1).values.mean()
        loss.backward()
        opt.step()
        sched.step()
        history.append(float(loss.detach()))
        if step % 40 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {history[-1]:.6f}  "
                  f"junctions {mesh.diagnostics['num_junctions']}")

    final_mesh = extract(model, resolution=resolution, box=BOX, **FAST)
    final_j = final_mesh.vertices[final_mesh.vertex_kind == JUNCTION].detach()
    junction_err = (
        float(torch.cdist(final_j, target_junction).min().item()) if final_j.numel() else float("nan")
    )
    print(f"  loss {history[0]:.6f} -> {history[-1]:.6f}  "
          f"({100 * (1 - history[-1] / history[0]):.1f}% reduction)")
    print(f"  triple-junction error: {junction_err:.4f}")
    print(f"  topology still valid: {check_topology(final_mesh)['ok']}")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    plot_mesh(axs[0], target_field, target_mesh, "target partition")
    plot_mesh(axs[1], model, init_mesh, "neural field, initial")
    plot_mesh(axs[2], model, final_mesh, f"neural field, after {steps} steps")
    axs[3].semilogy(history)
    axs[3].set_xlabel("optimiser step")
    axs[3].set_ylabel("loss on extracted mesh")
    axs[3].set_title("loss (log scale)", fontsize=10)
    axs[3].grid(alpha=0.3)
    fig.suptitle(
        "Gradients flow from a loss on the extracted multi-material mesh into the field",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.abspath(os.path.join(FIGDIR, "target_matching.png"))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {path}")

    return {
        "loss_initial": history[0],
        "loss_final": history[-1],
        "junction_error": junction_err,
        "figure": path,
    }


# --------------------------------------------------------------------------
# Experiment 2: Plateau's law
# --------------------------------------------------------------------------
def experiment_plateau(steps: int = 700, resolution: int = 96, radius: float = 0.8,
                       area_weight_range: tuple[float, float] = (50.0, 4000.0)) -> dict:
    """Minimise interface length at equal areas inside a disk.

    The domain is a disk rather than a square, and that choice is the whole
    point. A minimiser has to satisfy two conditions at once: 120 degrees where
    the three interfaces meet (equal tensions balance) and 90 degrees where an
    interface meets the domain wall. On a square those conflict for straight
    arms, so the minimiser has curved arms and no clean reference value exists;
    `plateau_reference.py` confirms this numerically, finding that the best
    exactly-120-degree partition of the square is *longer* than what a free
    field achieves.

    On a disk the two conditions agree, because radial arms are automatically
    perpendicular to the boundary circle. The minimiser is therefore known in
    closed form: three straight radial arms 120 degrees apart, meeting at the
    centre, with total interface length exactly 3R. That gives three
    independent numbers to check rather than a qualitative picture.
    """
    print("\n" + "=" * 78)
    print(f"2. Minimal equal-area partition of a disk (R={radius}); the exact answer is")
    print(f"   three radial arms at 120 deg meeting at the origin, length {3 * radius:.4f}")
    print("=" * 78)

    def build_neural():
        fld = NeuralMultiLabelField(num_classes=3, hidden=64, num_frequencies=3, seed=11)
        # Warm start deliberately off-centre and rotated, so reaching the answer
        # requires real movement rather than staying where it began.
        warm = SectorField(centre=(0.3, -0.25), num_classes=3, phase=0.4)
        wopt = torch.optim.Adam(fld.parameters(), lr=3e-3)
        xs = torch.rand(4096, 2) * 2 - 1
        for _ in range(400):
            wopt.zero_grad()
            torch.nn.functional.mse_loss(fld.logits(xs), warm.logits(xs)).backward()
            wopt.step()
        return fld, 1e-3

    def build_power():
        return PowerDiagramField(
            torch.tensor([[-0.45, -0.3], [0.5, -0.45], [0.1, 0.55]])
        ), 5e-3

    runs, figs = {}, {}
    for name, builder in (
        ("power diagram (3 sites)", build_power),
        ("neural field (K=3)", build_neural),
    ):
        field, lr = builder()
        init_mesh = extract(field, resolution=resolution, box=BOX, **FAST)
        init_angles = junction_angles(field, init_mesh)
        target_area = 1.0 / field.num_classes

        opt = torch.optim.Adam(field.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
        w0, w1 = area_weight_range
        history = []
        for step in range(steps):
            opt.zero_grad()
            # Penalty continuation: a weight strong enough to pin the areas from
            # the start would swamp the length term and freeze the geometry.
            area_weight = w0 * (w1 / w0) ** (step / max(steps - 1, 1))
            mesh = extract(field, resolution=resolution, box=BOX, **FAST)
            length = clipped_interface_length(mesh, radius)
            area_pen = ((disk_area_fractions(field, radius) - target_area) ** 2).sum()
            (length + area_weight * area_pen).backward()
            opt.step()
            sched.step()
            history.append((float(length.detach()), float(area_pen.detach())))
            if step % 100 == 0 or step == steps - 1:
                print(f"  [{name}] step {step:4d}  length {history[-1][0]:.4f}  "
                      f"area penalty {history[-1][1]:.2e}  "
                      f"junctions {mesh.diagnostics['num_junctions']}")

        final_mesh = extract(field, resolution=resolution, box=BOX, **FAST)
        angles = junction_angles(field, final_mesh)
        final_areas = hard_disk_area_fractions(field, radius)
        worst_dev = (
            max(abs(a - 120.0) for tri in angles for a in tri) if angles else float("nan")
        )
        junction_pts = final_mesh.vertices[final_mesh.vertex_kind == JUNCTION].detach()
        centre_err = float(junction_pts.norm(dim=-1).min()) if junction_pts.numel() else float("nan")

        print(f"  [{name}] initial angles      {[[round(a,1) for a in t] for t in init_angles]}")
        print(f"  [{name}] final angles        {[[round(a,1) for a in t] for t in angles]}")
        print(f"  [{name}] worst dev from 120  {worst_dev:.2f} deg")
        print(f"  [{name}] length              {history[0][0]:.4f} -> {history[-1][0]:.4f} "
              f"(exact {3 * radius:.4f}, error {abs(history[-1][0] - 3 * radius):.4f})")
        print(f"  [{name}] junction from origin {centre_err:.4f}")
        print(f"  [{name}] hard area fractions  {[round(a,4) for a in final_areas]} "
              f"(target {target_area:.4f})")
        print(f"  [{name}] topology valid       {check_topology(final_mesh)['ok']}")

        runs[name] = {
            "angles": angles,
            "worst_deviation_deg": worst_dev,
            "length_final": history[-1][0],
            "length_exact": 3 * radius,
            "junction_offset": centre_err,
            "areas": final_areas,
        }
        figs[name] = (field, init_mesh, final_mesh, history)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4))
    for row, (name, (field, init_mesh, final_mesh, history)) in enumerate(figs.items()):
        for col, (mesh, label) in enumerate(
            ((init_mesh, "initial"), (final_mesh, "minimal length, equal areas"))
        ):
            plot_mesh(axs[row, col], field, mesh, f"{name}\n{label}")
            circle = plt.Circle((0, 0), radius, fill=False, color="navy", lw=1.6, ls="--")
            axs[row, col].add_patch(circle)
        axs[row, 2].plot([h[0] for h in history], color="C0")
        axs[row, 2].axhline(3 * radius, color="crimson", ls="--", lw=1.4,
                            label=f"exact minimum 3R = {3 * radius:.2f}")
        axs[row, 2].set_xlabel("optimiser step")
        axs[row, 2].set_ylabel("interface length in disk")
        ang = runs[name]["angles"]
        axs[row, 2].set_title(
            f"final angles: {[round(a, 1) for a in ang[0]]}" if ang else "no junction",
            fontsize=9,
        )
        axs[row, 2].legend(fontsize=8)
        axs[row, 2].grid(alpha=0.3)
    fig.suptitle(
        "Plateau's 120-degree law recovered by optimising a multi-label field "
        "through the extractor",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.abspath(os.path.join(FIGDIR, "plateau_disk.png"))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {path}")
    runs["figure"] = path
    return runs


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    experiment_target_matching()
    experiment_plateau()


if __name__ == "__main__":
    main()
