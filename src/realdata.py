"""Extraction from a real segmentation network's output on a clinical CT.

The field here is not analytic and not ours: it is the per-class softmax volume
produced by TotalSegmentator's Couinaud liver-segment model on a thoracoabdominal
CT. Eight segments tile one organ, so segment-segment-segment curves and
segment-segment-background curves are anatomy rather than contrivance.

Two things are worth stating about what is being extracted. The field is a
probability vector per voxel, and the object we extract is the argmax partition
of its trilinear interpolant -- which is exactly the space-partitioning model
Hege et al. (1997) wrote down and then approximated by subdividing every cell
into 6^3 sub-cells. We solve it instead. And because only differences of the
channels matter, using probabilities rather than the logits behind them costs
nothing: softmax shifts every channel at a node by the same constant, which
cancels in every comparison the method makes.

The extraction grid is aligned so that its nodes coincide with voxel centres,
so no interpolation happens before the one the method is defined by.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fields import MultiLabelField

ROOT = Path(__file__).resolve().parent.parent
PROBS = ROOT / "data" / "probs_liver.npz"
META = ROOT / "data" / "probs_liver.pkl"


class VoxelProbabilityField(MultiLabelField):
    """Trilinear interpolant of a per-class probability volume, in millimetres.

    `block` is (K, nz, ny, nx) and `origin`/`spacing` place voxel (0,0,0) at
    `origin` millimetres with the given per-axis step. Sampling is exact at
    voxel centres, so when the extraction grid is aligned to them the node
    values are the network's own output rather than a resampling of it.
    """

    dim = 3

    def __init__(self, block: np.ndarray, origin, spacing):
        super().__init__()
        self.num_classes = int(block.shape[0])
        self.register_buffer("vol", torch.as_tensor(block, dtype=torch.get_default_dtype())[None])
        self.register_buffer("origin", torch.as_tensor(origin, dtype=torch.get_default_dtype()))
        self.register_buffer("spacing", torch.as_tensor(spacing, dtype=torch.get_default_dtype()))
        self.register_buffer("size", torch.tensor(block.shape[1:][::-1], dtype=torch.long))

    def logits(self, x: Tensor) -> Tensor:
        # x is (N, 3) in millimetres, ordered (x, y, z); the volume is (z, y, x).
        idx = (x - self.origin) / self.spacing
        norm = 2.0 * idx / (self.size.to(x.dtype) - 1.0) - 1.0
        grid = norm.reshape(1, 1, 1, -1, 3)
        # Casting the volume per call would copy hundreds of megabytes when the
        # whole organ is loaded rather than a crop.
        vol = self.vol if self.vol.dtype == x.dtype else self.vol.to(x.dtype)
        out = F.grid_sample(vol, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.reshape(self.num_classes, -1).T.contiguous()


def load_block(origin_vox, width: int):
    """Crop a (width)^3 voxel block and return the field plus its physical box."""
    probs = np.load(PROBS)["probabilities"]
    with open(META, "rb") as handle:
        meta = pickle.load(handle)
    sz, sy, sx = (float(v) for v in meta["spacing"])  # array axes are (z, y, x)

    z0, y0, x0 = origin_vox
    block = probs[:, z0:z0 + width, y0:y0 + width, x0:x0 + width]
    block = np.ascontiguousarray(block)

    # Put the block centre at the origin so the geometry is centred in the box.
    span = np.array([(width - 1) * sx, (width - 1) * sy, (width - 1) * sz])
    corner = -0.5 * span
    field = VoxelProbabilityField(block, corner, (sx, sy, sz))
    box = (corner[0], corner[0] + span[0],
           corner[1], corner[1] + span[1],
           corner[2], corner[2] + span[2])
    return field, box, block, (sx, sy, sz)


def load_organ(margin: int = 3):
    """The whole labelled organ, on the box that bounds it with a background margin.

    Unlike `load_block` this does not align grid nodes to voxel centres --- it
    cannot, since one uniform resolution cannot match three different voxel
    counts --- so the field is genuinely resampled here. That is fine for an
    export, and is why the paper's numbers come from `load_block` instead. The
    margin guarantees every labelled region is surrounded by background, so the
    per-region surfaces close rather than being cut by the box.
    """
    probs = np.load(PROBS)["probabilities"]
    with open(META, "rb") as handle:
        meta = pickle.load(handle)
    spacing_zyx = np.array([float(v) for v in meta["spacing"]])

    occupied = np.argwhere(probs.argmax(0) > 0)
    lo = np.maximum(occupied.min(0) - margin, 0)
    hi = np.minimum(occupied.max(0) + margin, np.array(probs.shape[1:]) - 1)

    field = VoxelProbabilityField(probs, (0.0, 0.0, 0.0), spacing_zyx[::-1].copy())
    # Physical bounds of the bounding box, as (x, y, z).
    lo_mm = (lo * spacing_zyx)[::-1].copy()
    hi_mm = (hi * spacing_zyx)[::-1].copy()
    box = (lo_mm[0], hi_mm[0], lo_mm[1], hi_mm[1], lo_mm[2], hi_mm[2])
    return field, box, tuple(int(v) for v in hi - lo + 1)


def interface_distance(field, surf, box, resolution, points: Tensor,
                       perturb: float = 0.0) -> Tensor:
    """Distance from each point to the interface of the two leading channels.

    The probability gap between the top two channels is not interpretable on its
    own, since it depends on how sharp the network's transition is. Dividing by
    the gradient of that gap -- which the P1 interpolant supplies exactly, being
    affine on each tetrahedron -- converts it into millimetres.
    """
    values, tet, _ = _locate(field, surf, box, resolution, points, perturb)
    order = values.topk(2, dim=1).indices
    gap = values.gather(1, order[:, :1]).squeeze(1) - values.gather(1, order[:, 1:2]).squeeze(1)

    with torch.no_grad():
        node_logits = envelope3d_perturb(field, surf, perturb)
        corners = surf.nodes[surf.tets[tet]]                    # (N, 4, 3)
        vals = node_logits[surf.tets[tet]]                      # (N, 4, K)
        g = vals.gather(2, order[:, None, :1].expand(-1, 4, 1)).squeeze(2) \
            - vals.gather(2, order[:, None, 1:2].expand(-1, 4, 1)).squeeze(2)
        edges = (corners[:, 1:, :] - corners[:, 0:1, :])        # (N, 3, 3)
        rhs = (g[:, 1:] - g[:, 0:1])[..., None]                 # (N, 3, 1)
        grad = torch.linalg.solve(edges, rhs).squeeze(-1)       # (N, 3)
        return gap / grad.norm(dim=1).clamp_min(1e-30)


def envelope3d_perturb(field, surf, perturb: float) -> Tensor:
    import envelope3d
    return envelope3d._perturb(field.logits(surf.nodes), perturb)


def _locate(field, surf, box, resolution, points: Tensor, perturb: float):
    """Interpolated values, containing tet index, and barycentric coordinates."""
    with torch.no_grad():
        node_logits = envelope3d_perturb(field, surf, perturb)
        lo = torch.tensor([box[0], box[2], box[4]], dtype=points.dtype)
        hi = torch.tensor([box[1], box[3], box[5]], dtype=points.dtype)
        step = (hi - lo) / resolution

        cell = ((points - lo) / step).floor().long().clamp_(0, resolution - 1)
        cube = (cell[:, 2] * resolution + cell[:, 1]) * resolution + cell[:, 0]
        num_cubes = resolution ** 3

        corners = surf.nodes[surf.tets]
        best = torch.zeros((points.shape[0],), dtype=torch.long)
        found = torch.zeros((points.shape[0],), dtype=torch.bool)
        bary = torch.zeros((points.shape[0], 4), dtype=points.dtype)
        for k in range(6):
            cand = cube + k * num_cubes
            p = corners[cand]
            edges = (p[:, 1:, :] - p[:, 0:1, :]).transpose(1, 2)
            coords = torch.linalg.solve(edges, (points - p[:, 0, :])[..., None]).squeeze(-1)
            b = torch.cat([1.0 - coords.sum(1, keepdim=True), coords], dim=1)
            ok = (b >= -1e-9).all(dim=1) & ~found
            best = torch.where(ok, cand, best)
            bary = torch.where(ok[:, None], b, bary)
            found |= ok

        vals = node_logits[surf.tets[best]]
        values = (bary[:, :, None] * vals).sum(dim=1)
        return values, best, found


def interpolated_logits_at(field, surf, box, resolution, points: Tensor,
                           perturb: float = 0.0) -> Tensor:
    """Evaluate the P1 interpolant the method is defined on, at arbitrary points.

    validate3d has the same helper but hard-codes the [-1, 1]^3 box, and the
    real-data box is neither cubic nor centred on unit extents. Point location
    is still cheap: the containing cube follows from flooring, and only its six
    tets need a barycentric test.
    """
    import envelope3d

    with torch.no_grad():
        node_logits = envelope3d._perturb(field.logits(surf.nodes), perturb)
        lo = torch.tensor([box[0], box[2], box[4]], dtype=points.dtype)
        hi = torch.tensor([box[1], box[3], box[5]], dtype=points.dtype)
        step = (hi - lo) / resolution

        cell = ((points - lo) / step).floor().long().clamp_(0, resolution - 1)
        cube = (cell[:, 2] * resolution + cell[:, 1]) * resolution + cell[:, 0]
        num_cubes = resolution ** 3

        corners = surf.nodes[surf.tets]
        best = torch.full((points.shape[0],), -1, dtype=torch.long)
        best_bary = torch.zeros((points.shape[0], 4), dtype=points.dtype)
        for k in range(6):
            cand = cube + k * num_cubes
            p = corners[cand]
            edges = (p[:, 1:, :] - p[:, 0:1, :]).transpose(1, 2)
            coords = torch.linalg.solve(edges, (points - p[:, 0, :])[..., None]).squeeze(-1)
            bary = torch.cat([1.0 - coords.sum(1, keepdim=True), coords], dim=1)
            ok = (bary >= -1e-9).all(dim=1) & (best < 0)
            best = torch.where(ok, cand, best)
            best_bary = torch.where(ok[:, None], bary, best_bary)

        found = best >= 0
        out = torch.full((points.shape[0], node_logits.shape[1]),
                         float("nan"), dtype=points.dtype)
        tet_nodes = surf.tets[best[found]]
        vals = node_logits[tet_nodes]                      # (M, 4, K)
        out[found] = (best_bary[found][:, :, None] * vals).sum(dim=1)
        return out


def top_two_margin(values: Tensor) -> Tensor:
    """Gap between the best and second-best channel; zero exactly on an interface."""
    top = values.topk(2, dim=1).values
    return top[:, 0] - top[:, 1]


def check_nodes_are_voxels(field, box, resolution, block) -> float:
    """The grid nodes should land on voxel centres, so sampling must be exact."""
    import envelope3d

    nodes, _ = envelope3d.build_tetrahedral_grid(resolution, box)
    sampled = field.logits(nodes)
    truth = torch.as_tensor(block, dtype=sampled.dtype)
    truth = truth.reshape(block.shape[0], -1).T  # (z,y,x) flattened matches node order
    return float((sampled - truth).abs().max())
