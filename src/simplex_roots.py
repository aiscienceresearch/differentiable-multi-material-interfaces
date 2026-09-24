"""Differentiable location of multi-label geometric primitives on simplices.

Multi-label extraction needs one geometric primitive per codimension, and they
are all the same operation at different dimensions. On an m-simplex whose m+1
corners carry m+1 distinct labels there is a point where all m+1 logits are
equal, found by solving m equations in the m affine coordinates of the simplex:

    m = 1   crossing on an edge          f_i = f_j                    (2D and 3D)
    m = 2   triple point on a face       f_i = f_j = f_k              (2D and 3D)
    m = 3   quadruple point in a tet     f_i = f_j = f_k = f_l        (3D)

Writing it once makes the recursive structure explicit: the 2D extractor uses
m = 1 and m = 2, and the 3D extractor uses the same two plus m = 3, with faces
in 3D solving exactly the problem that cells solve in 2D.

Every routine comes in two halves. `locate_*` finds the point to solver
tolerance under `no_grad`, and `reattach_*_gradient` applies one implicit
function theorem correction whose value is unchanged but whose gradient is the
implicit derivative dx/dtheta = -J^-1 dF/dtheta. Accuracy is therefore set by
the solver and differentiability by the reattachment, and neither limits the
other. Because the labels enter the residual F, gradients flow through the
label assignment rather than only through positions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fields import MultiLabelField

SINGULAR = 1e-30


def pair_diff(values: Tensor, i: Tensor, j: Tensor) -> Tensor:
    """values (M, K) -> f_i - f_j, shape (M,)."""
    return values.gather(1, i[:, None]).squeeze(1) - values.gather(1, j[:, None]).squeeze(1)


def pair_diff_grad(grads: Tensor, i: Tensor, j: Tensor) -> Tensor:
    """grads (M, K, d) -> grad f_i - grad f_j, shape (M, d)."""
    d = grads.shape[-1]
    idx_i = i[:, None, None].expand(-1, 1, d)
    idx_j = j[:, None, None].expand(-1, 1, d)
    return grads.gather(1, idx_i).squeeze(1) - grads.gather(1, idx_j).squeeze(1)


# --------------------------------------------------------------------------
# m = 1: crossings along an edge
# --------------------------------------------------------------------------
def locate_segment_crossing(field: MultiLabelField, a: Tensor, b: Tensor, i: Tensor, j: Tensor,
                            bisection_steps: int = 60, newton_steps: int = 8) -> Tensor:
    """Find t in (0,1) with f_i = f_j on the segment a -> b.

    Label i is the argmax at a and j at b, so g(0) > 0 > g(1) and the root is
    bracketed; bisection therefore cannot fail, and Newton polishes it.
    """
    with torch.no_grad():
        lo = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
        hi = torch.ones_like(lo)
        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            g = pair_diff(field.logits(a + mid[:, None] * (b - a)), i, j)
            positive = g > 0
            lo = torch.where(positive, mid, lo)
            hi = torch.where(positive, hi, mid)
        t = 0.5 * (lo + hi)

        direction = b - a
        for _ in range(newton_steps):
            x = a + t[:, None] * direction
            g = pair_diff(field.logits(x), i, j)
            dg = (pair_diff_grad(field.logit_grads(x), i, j) * direction).sum(-1)
            step = torch.where(dg.abs() > SINGULAR, g / dg, torch.zeros_like(g))
            t = (t - step).clamp(0.0, 1.0)
    return t


def reattach_segment_gradient(field: MultiLabelField, a: Tensor, b: Tensor, i: Tensor, j: Tensor,
                              t: Tensor) -> Tensor:
    """One implicit-function-theorem correction, giving dt/dtheta at fixed t."""
    direction = b - a
    x = a + t[:, None] * direction
    with torch.no_grad():
        dg = (pair_diff_grad(field.logit_grads(x), i, j) * direction).sum(-1)
    g = pair_diff(field.logits(x), i, j)  # keeps the graph to theta
    safe = torch.where(dg.abs() > SINGULAR, dg, torch.full_like(dg, SINGULAR))
    return t - g / safe


# --------------------------------------------------------------------------
# m >= 2: equal-logit points on an affine simplex
# --------------------------------------------------------------------------
def affine_point(corners: Tensor, u: Tensor) -> Tensor:
    """Map affine coordinates to a point: p0 + sum_s u_s (p_{s+1} - p0).

    corners (M, m+1, d), u (M, m) -> (M, d).
    """
    edges = corners[:, 1:, :] - corners[:, 0:1, :]  # (M, m, d)
    return corners[:, 0, :] + (u[..., None] * edges).sum(dim=1)


def affine_coords(corners: Tensor, x: Tensor) -> Tensor:
    """Inverse of `affine_point`: least-squares affine coordinates of x.

    Exact when the simplex is full-dimensional; for a lower-dimensional simplex
    embedded in R^d (a face of a tet) it returns the coordinates of the
    orthogonal projection, which is what an initial guess wants.
    """
    edges = (corners[:, 1:, :] - corners[:, 0:1, :]).transpose(1, 2)  # (M, d, m)
    rhs = (x - corners[:, 0, :])[..., None]
    gram = edges.transpose(1, 2) @ edges
    return torch.linalg.solve(gram, edges.transpose(1, 2) @ rhs).squeeze(-1)


def _residual_and_jacobian(field: MultiLabelField, corners: Tensor, u: Tensor, labels: Tensor,
                           want_jacobian: bool):
    """F_r = f_{l_r} - f_{l_{r+1}} for r < m, and dF/du."""
    m = u.shape[1]
    x = affine_point(corners, u)
    f = field.logits(x)
    F = torch.stack([pair_diff(f, labels[:, r], labels[:, r + 1]) for r in range(m)], dim=-1)
    if not want_jacobian:
        return F, None
    grads = field.logit_grads(x)
    edges = corners[:, 1:, :] - corners[:, 0:1, :]  # (M, m, d)
    rows = [
        (pair_diff_grad(grads, labels[:, r], labels[:, r + 1])[:, None, :] * edges).sum(-1)
        for r in range(m)
    ]
    return F, torch.stack(rows, dim=1)  # (M, m, m)


def _solve_safely(J: Tensor, F: Tensor) -> tuple[Tensor, Tensor]:
    """Batched solve that leaves singular systems untouched instead of blowing up."""
    det = torch.linalg.det(J)
    singular = det.abs() < SINGULAR
    eye = torch.eye(J.shape[-1], dtype=J.dtype, device=J.device).expand_as(J)
    J_safe = torch.where(singular[:, None, None], eye, J)
    step = torch.linalg.solve(J_safe, F[..., None]).squeeze(-1)
    return torch.where(singular[:, None], torch.zeros_like(step), step), singular


def clamp_to_dilated_simplex(u: Tensor, margin: float) -> Tensor:
    """Keep Newton iterates near their simplex without pinning them to it.

    A genuine equal-logit point can lie slightly outside a simplex that has
    distinct corner labels, so clamping to the simplex itself would stop the
    solver short of the true root. The iterate is confined to the simplex
    dilated by `margin` in barycentric units, which prevents divergence while
    leaving the true root reachable. Points that end up far outside are
    reported as under-resolved rather than silently moved.
    """
    alpha = 1.0 - u.sum(dim=-1, keepdim=True)
    bary = torch.cat([alpha, u], dim=-1).clamp_min(-margin)
    bary = bary / bary.sum(dim=-1, keepdim=True).clamp_min(SINGULAR)
    return bary[:, 1:]


def locate_simplex_equal_logits(field: MultiLabelField, corners: Tensor, labels: Tensor,
                                u_init: Tensor | None = None, newton_steps: int = 25,
                                margin: float = 1.0, polish_steps: int = 8) -> Tensor:
    """Newton solve for the point where all m+1 given logits are equal.

    corners (M, m+1, d), labels (M, m+1) -> affine coordinates u (M, m).

    The clamped phase keeps the iterate near its simplex so it cannot run away,
    and the polish phase then drops the clamp. Polishing matters for gradients,
    not just for accuracy: the reattachment below differentiates the residual at
    whatever point it is handed, so it returns the implicit derivative of the
    true root only if the iterate has actually reached it. An iterate resting
    against the clamp still yields the right position after one correction, but
    the derivative would be evaluated in the wrong place. Polish steps are
    accepted only when they reduce the residual, so they cannot make a
    converged solve worse.
    """
    m = labels.shape[1] - 1
    with torch.no_grad():
        if u_init is None:
            u = torch.full(
                (corners.shape[0], m), 1.0 / (m + 1), dtype=corners.dtype, device=corners.device
            )
        else:
            u = u_init.clone()
        for _ in range(newton_steps):
            F, J = _residual_and_jacobian(field, corners, u, labels, want_jacobian=True)
            step, _ = _solve_safely(J, F)
            u = clamp_to_dilated_simplex(u - step, margin)
        for _ in range(polish_steps):
            F, J = _residual_and_jacobian(field, corners, u, labels, want_jacobian=True)
            step, _ = _solve_safely(J, F)
            candidate = u - step
            F_new, _ = _residual_and_jacobian(
                field, corners, candidate, labels, want_jacobian=False
            )
            better = F_new.norm(dim=-1) < F.norm(dim=-1)
            u = torch.where(better[:, None], candidate, u)
    return u


def reattach_simplex_gradient(field: MultiLabelField, corners: Tensor, u: Tensor,
                              labels: Tensor) -> Tensor:
    """Vector implicit-function-theorem correction: du/dtheta = -J^-1 dF/dtheta."""
    with torch.no_grad():
        _, J = _residual_and_jacobian(field, corners, u, labels, want_jacobian=True)
        det = torch.linalg.det(J)
        singular = det.abs() < SINGULAR
        eye = torch.eye(J.shape[-1], dtype=J.dtype, device=J.device).expand_as(J)
        J = torch.where(singular[:, None, None], eye, J)
    F, _ = _residual_and_jacobian(field, corners, u, labels, want_jacobian=False)
    correction = torch.linalg.solve(J, F[..., None]).squeeze(-1)
    correction = torch.where(singular[:, None], torch.zeros_like(correction), correction)
    return u - correction


def simplex_residual_norm(field: MultiLabelField, corners: Tensor, u: Tensor,
                          labels: Tensor) -> Tensor:
    """||F|| at the given affine coordinates, for diagnostics."""
    with torch.no_grad():
        F, _ = _residual_and_jacobian(field, corners, u, labels, want_jacobian=False)
    return F.norm(dim=-1)


def barycentric_overshoot(u: Tensor) -> Tensor:
    """How far outside its simplex a point sits, in barycentric units (0 = inside)."""
    alpha = 1.0 - u.sum(dim=-1, keepdim=True)
    bary = torch.cat([alpha, u], dim=-1)
    return (-bary.min(dim=-1).values).clamp_min(0.0)
