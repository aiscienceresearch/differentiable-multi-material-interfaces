"""Multi-label implicit fields.

A multi-label field maps x in R^d to K logits. The represented partition is
    R_k = { x : f_k(x) > f_j(x) for all j != k },
so interfaces are the sets where the top two logits are equal, triple junctions
where the top three are, and in 3D quadruple points where the top four are.
This is the softmax/argmax partition used by multi-class neural fields, and it
is exactly the structure a K-channel classifier head produces.

Every field here is written for general d so that the 2D and 3D extractors can
be validated against the same ground truth rather than against two independent
implementations. Closed-form ground truth is what makes accuracy and gradients
checkable:

- PowerDiagramField with unit weights is the Voronoi diagram of its sites, so
  interfaces are perpendicular bisectors, and the point where d+1 labels meet
  is the circumcentre of the corresponding d+1 sites.
- SectorField is an exact fan of flat sectors meeting at a prescribed point.
- AnnulusField has curved interfaces at exactly known radii, which is what a
  convergence study needs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MultiLabelField(nn.Module):
    """Base class. Subclasses implement `logits` and declare `dim`."""

    num_classes: int
    dim: int = 2

    def logits(self, x: Tensor) -> Tensor:
        """x: (N, d) -> (N, K)."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.logits(x)

    def labels(self, x: Tensor) -> Tensor:
        """Hard label assignment, (N,) long."""
        return self.logits(x).argmax(dim=-1)

    def logit_grads(self, x: Tensor) -> Tensor:
        """Spatial gradients d f_k / d x, returned as (N, K, d).

        Uses autograd so subclasses need only define `logits`. Points are
        independent, so one backward pass per class recovers the full per-point
        Jacobian; K is small in every case we care about.

        The result is always detached from the field parameters. Callers use it
        only for Jacobians in the implicit-function-theorem corrections, which
        require the detached value by construction, and `enable_grad` lets it
        be called from inside a `no_grad` region.
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            f = self.logits(x)
            grads = []
            for k in range(f.shape[-1]):
                (g,) = torch.autograd.grad(
                    f[:, k].sum(), x, retain_graph=(k + 1 < f.shape[-1])
                )
                grads.append(g.detach())
        return torch.stack(grads, dim=1)


class PowerDiagramField(MultiLabelField):
    """f_k(x) = -||x - c_k||^2 + w_k.

    With w = 0 the argmax partition is the Voronoi diagram of the sites, whose
    geometry we know exactly. Sites and weights are nn.Parameters so gradients
    can be pushed into them from a loss on the extracted mesh.
    """

    def __init__(self, sites: Tensor, weights: Tensor | None = None):
        super().__init__()
        sites = torch.as_tensor(sites, dtype=torch.get_default_dtype())
        self.sites = nn.Parameter(sites.clone())
        if weights is None:
            weights = torch.zeros(sites.shape[0], dtype=sites.dtype)
        self.weights = nn.Parameter(torch.as_tensor(weights, dtype=sites.dtype).clone())
        self.num_classes = sites.shape[0]
        self.dim = sites.shape[1]

    def logits(self, x: Tensor) -> Tensor:
        d2 = (x[:, None, :] - self.sites[None, :, :]).pow(2).sum(-1)
        return -d2 + self.weights[None, :]

    def exact_equidistant_point(self, indices) -> Tensor:
        """Exact point where the given d+1 classes all meet.

        Each equality f_i = f_j is linear in x, so d+1 labels give a d x d
        linear system. With unit weights the solution is the circumcentre of
        the d+1 sites: the triple point in 2D, the quadruple point in 3D.
        """
        c, w = self.sites, self.weights
        i = indices[0]
        rows, rhs = [], []
        for j in indices[1:]:
            rows.append(2.0 * (c[j] - c[i]))
            rhs.append(c[j].pow(2).sum() - c[i].pow(2).sum() + w[i] - w[j])
        return torch.linalg.solve(torch.stack(rows), torch.stack(rhs))

    def exact_triple_line(self, i: int, j: int, k: int) -> tuple[Tensor, Tensor]:
        """Point and unit direction of the 3D line where classes i, j, k meet.

        Three labels in 3D leave one degree of freedom, so the triple set is a
        straight line: the intersection of two bisector planes. Returned as the
        minimum-norm point on the line plus the direction spanning its null
        space, which is what a point-to-line distance check needs.
        """
        c, w = self.sites, self.weights
        A = torch.stack([2.0 * (c[j] - c[i]), 2.0 * (c[k] - c[i])])
        rhs = torch.stack(
            [
                c[j].pow(2).sum() - c[i].pow(2).sum() + w[i] - w[j],
                c[k].pow(2).sum() - c[i].pow(2).sum() + w[i] - w[k],
            ]
        )
        point = torch.linalg.pinv(A) @ rhs
        direction = torch.linalg.cross(A[0], A[1])
        return point, direction / direction.norm()

    def exact_interface_normal(self, i: int, j: int) -> tuple[Tensor, Tensor]:
        """Unit normal n and offset b of the i|j interface line {x : n.x = b}."""
        c, w = self.sites, self.weights
        n = 2.0 * (c[j] - c[i])
        b = c[j].pow(2).sum() - c[i].pow(2).sum() + w[i] - w[j]
        norm = n.norm()
        return n / norm, b / norm


class BubbleField(MultiLabelField):
    """f_0(x) = 0 and f_k(x) = -||x - c_k||^2 + r_k^2 for k >= 1.

    Holding one channel constant makes each interior channel's interface with
    it the sphere ||x - c_k|| = r_k, while interior channels still meet each
    other on planes, exactly as in a power diagram. The family is therefore the
    smallest one that contains the standard double bubble: two spheres of equal
    radius R whose centres are R apart, separated by the flat disk their
    bisector cuts. That is what makes it usable as an optimization target ---
    the minimiser is inside the family, so a failure to reach it is a failure
    of the extractor's gradients and not of the parametrization.

    Radii are stored as logarithms so they stay positive; a negative squared
    radius would delete a region and strand the optimizer with no gradient.
    """

    def __init__(self, centres: Tensor, radii: Tensor):
        super().__init__()
        centres = torch.as_tensor(centres, dtype=torch.get_default_dtype())
        radii = torch.as_tensor(radii, dtype=centres.dtype)
        self.centres = nn.Parameter(centres.clone())
        self.log_radii = nn.Parameter(radii.clone().log())
        self.num_classes = centres.shape[0] + 1
        self.dim = centres.shape[1]

    def radii(self) -> Tensor:
        return self.log_radii.exp()

    def separation(self) -> Tensor:
        """Distance between the two interior centres, which Plateau's condition ties
        to the radii: the three normals have equal length exactly when
        r_1 = r_2 = ||c_1 - c_2||."""
        return (self.centres[0] - self.centres[1]).norm()

    def logits(self, x: Tensor) -> Tensor:
        d2 = (x[:, None, :] - self.centres[None, :, :]).pow(2).sum(-1)
        interior = -d2 + self.radii().pow(2)[None, :]
        return torch.cat([x.new_zeros(x.shape[0], 1), interior], dim=1)


def double_bubble_reference(radius: float) -> dict:
    """Closed-form standard double bubble of two equal volumes.

    Two spheres of radius R centred a distance R apart, cut by the bisecting
    plane. The three surfaces meet at 120 degrees along the circle where the
    spheres cross, which in this family is the statement r_1 = r_2 = ||c_1-c_2||.
    Each lobe is the spherical cap of height 3R/2, so its volume is
    pi h^2 (3R - h) / 3 = 9 pi R^3 / 8 and its curved area 2 pi R h = 3 pi R^2;
    the flat wall has radius sqrt(3) R / 2.
    """
    r = float(radius)
    return {
        "radius": r,
        "separation": r,
        "volume_each": 9.0 * math.pi * r ** 3 / 8.0,
        "cap_area_each": 3.0 * math.pi * r ** 2,
        "wall_radius": math.sqrt(3.0) * r / 2.0,
        "wall_area": 3.0 * math.pi * r ** 2 / 4.0,
        "total_area": 27.0 * math.pi * r ** 2 / 4.0,
    }


class AnnulusField(MultiLabelField):
    """f_k(x) = -(||x|| - r_k)^2, giving concentric annuli.

    The argmax assigns each point to the nearest target radius, so the k|k+1
    interface is exactly the circle of radius (r_k + r_{k+1}) / 2. Curved
    interfaces with closed-form ground truth are what make a polyline
    convergence study meaningful: the Voronoi field has straight interfaces, so
    the extracted polyline there is exact and measures nothing.
    """

    def __init__(self, radii=(0.15, 0.45, 0.75, 1.05), dim: int = 2):
        super().__init__()
        radii_t = torch.as_tensor(radii, dtype=torch.get_default_dtype())
        self.radii = nn.Parameter(radii_t.clone())
        self.num_classes = radii_t.numel()
        self.dim = dim

    def logits(self, x: Tensor) -> Tensor:
        r = x.norm(dim=-1, keepdim=True)
        return -(r - self.radii[None, :]).pow(2)

    def exact_interface_radius(self, i: int, j: int) -> Tensor:
        return 0.5 * (self.radii[i] + self.radii[j])


def simplex_directions(dim: int, dtype: torch.dtype | None = None) -> Tensor:
    """d+1 unit vectors in R^d that sum to zero, pointing at a regular simplex.

    Used as sector directions: because no direction is dominated by the others,
    all d+1 sectors meet at a single point, which is the configuration a 2D
    triple junction and a 3D quadruple point need for exact ground truth.
    """
    dtype = dtype or torch.get_default_dtype()
    n = dim + 1
    centred = torch.eye(n, dtype=dtype) - 1.0 / n
    basis, _ = torch.linalg.qr(centred)  # first `dim` columns span the sum-zero hyperplane
    dirs = centred @ basis[:, :dim]
    return dirs / dirs.norm(dim=-1, keepdim=True)


class SectorField(MultiLabelField):
    """f_k(x) = <x - centre, d_k>, an exact fan of K flat sectors about `centre`.

    In 2D the interfaces are rays and the sectors meet at one triple junction.
    In 3D they are half-planes meeting along triple lines that in turn meet at
    one quadruple point, so the same field exercises every codimension.
    """

    def __init__(self, centre: Tensor | tuple[float, ...] = (0.0, 0.0), num_classes: int = 3,
                 phase: float = 0.0, directions: Tensor | None = None):
        super().__init__()
        centre = torch.as_tensor(centre, dtype=torch.get_default_dtype())
        self.centre = nn.Parameter(centre.clone())
        if directions is None:
            if centre.numel() != 2:
                raise ValueError("give explicit `directions` for fields outside 2D")
            angles = (
                torch.arange(num_classes, dtype=centre.dtype) * (2 * math.pi / num_classes) + phase
            )
            directions = torch.stack([angles.cos(), angles.sin()], dim=-1)
        directions = torch.as_tensor(directions, dtype=centre.dtype)
        self.register_buffer("dirs", directions / directions.norm(dim=-1, keepdim=True))
        self.num_classes = directions.shape[0]
        self.dim = directions.shape[1]

    def logits(self, x: Tensor) -> Tensor:
        return (x - self.centre) @ self.dirs.T


class NeuralMultiLabelField(MultiLabelField):
    """MLP with Fourier features, emitting K logits.

    Fourier features are used rather than raw coordinates because the extractor
    relies on the logits being smooth enough for Newton steps to converge; a
    ReLU MLP on raw coordinates has distributional curvature at its creases.
    """

    def __init__(self, num_classes: int = 3, hidden: int = 64, num_frequencies: int = 4,
                 depth: int = 3, seed: int | None = 0, dim: int = 2):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        freqs = 2.0 ** torch.arange(num_frequencies, dtype=torch.get_default_dtype())
        self.register_buffer("freqs", freqs)
        in_dim = dim + 2 * dim * num_frequencies
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.dim = dim

    def features(self, x: Tensor) -> Tensor:
        proj = (x[..., None] * self.freqs).flatten(1)  # (N, d * F)
        return torch.cat([x, proj.sin(), proj.cos()], dim=-1)

    def logits(self, x: Tensor) -> Tensor:
        return self.net(self.features(x))


def circumcentre(p: Tensor, q: Tensor, r: Tensor) -> Tensor:
    """Circumcentre of three 2D points, in closed form.

    Kept separate from PowerDiagramField so that validation can compare the
    extractor's autograd gradients against derivatives of an independent
    analytic expression rather than against another path through the same code.
    """
    ax, ay = p[0], p[1]
    bx, by = q[0], q[1]
    cx, cy = r[0], r[1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return torch.stack([ux, uy])
