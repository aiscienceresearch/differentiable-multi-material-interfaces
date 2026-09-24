"""Exact geometry of a 3D power diagram cell clipped to a box.

A power-diagram cell is the intersection of half-spaces, so its vertices,
facet areas and volume are computable in closed form. That gives the 3D
extractor ground truth for every level of its output at once: cell volume
tests the interface surfaces, facet areas test the individual label pairs,
facet edges test the triple curves, and facet corners test the quadruple
points. Nothing here shares code with the extractor, so agreement is
meaningful.

Vertices are enumerated by intersecting every triple of bounding planes and
keeping the points that satisfy all the half-spaces. Cells have around a dozen
planes, so the cubic enumeration is far cheaper than the extraction it checks.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor

TOL = 1e-9


def cell_planes(sites: Tensor, weights: Tensor, k: int,
                box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)):
    """Half-spaces {x : n.x <= b} bounding cell k, plus the label facing each one.

    The power cell of k is where -|x-c_k|^2 + w_k is largest, and the pairwise
    condition against site j rearranges to 2(c_j - c_k).x <= |c_j|^2 - |c_k|^2
    + w_k - w_j, which is linear in x.
    """
    normals, offsets, owners = [], [], []
    for j in range(sites.shape[0]):
        if j == k:
            continue
        normals.append(2.0 * (sites[j] - sites[k]))
        offsets.append(
            sites[j].pow(2).sum() - sites[k].pow(2).sum() + weights[k] - weights[j]
        )
        owners.append(j)
    for axis in range(3):
        lo, hi = box[2 * axis], box[2 * axis + 1]
        e = torch.zeros(3, dtype=sites.dtype)
        e[axis] = 1.0
        normals.append(e.clone())
        offsets.append(torch.as_tensor(hi, dtype=sites.dtype))
        owners.append(-1)
        normals.append(-e)
        offsets.append(torch.as_tensor(-lo, dtype=sites.dtype))
        owners.append(-1)
    return torch.stack(normals), torch.stack(offsets), owners


def cell_vertices(normals: Tensor, offsets: Tensor) -> Tensor:
    """Corners of the polytope: triple-plane intersections that satisfy every plane."""
    pts = []
    n = normals.shape[0]
    for a, b, c in itertools.combinations(range(n), 3):
        A = torch.stack([normals[a], normals[b], normals[c]])
        if A.det().abs() < 1e-12:
            continue
        x = torch.linalg.solve(A, torch.stack([offsets[a], offsets[b], offsets[c]]))
        if bool(((normals @ x - offsets) <= TOL).all()):
            pts.append(x)
    if not pts:
        return torch.zeros((0, 3), dtype=normals.dtype)
    pts_t = torch.stack(pts)
    # Merge duplicates produced by degenerate corners where >3 planes meet.
    keep = []
    for i in range(pts_t.shape[0]):
        if all((pts_t[i] - pts_t[j]).norm() > 1e-8 for j in keep):
            keep.append(i)
    return pts_t[keep]


def _polygon_area(points: Tensor, normal: Tensor) -> Tensor:
    """Area of a convex polygon given its unordered coplanar vertices."""
    if points.shape[0] < 3:
        return torch.zeros((), dtype=points.dtype)
    centre = points.mean(dim=0)
    axis = normal / normal.norm()
    ref = points[0] - centre
    ref = ref - (ref @ axis) * axis
    ref = ref / ref.norm()
    perp = torch.linalg.cross(axis, ref)
    rel = points - centre
    angle = torch.atan2(rel @ perp, rel @ ref)
    ordered = points[angle.argsort()]
    fan = torch.linalg.cross(
        ordered[1:-1] - ordered[0], ordered[2:] - ordered[0]
    )
    return 0.5 * fan.norm(dim=-1).sum()


def cell_geometry(sites: Tensor, weights: Tensor, k: int,
                  box=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)):
    """Exact volume of cell k, its facet areas by neighbour, and whether it is interior."""
    normals, offsets, owners = cell_planes(sites, weights, k, box)
    verts = cell_vertices(normals, offsets)
    facet_area: dict[int, Tensor] = {}
    volume = torch.zeros((), dtype=sites.dtype)
    touches_box = False
    for p in range(normals.shape[0]):
        on_plane = (normals[p] @ verts.T - offsets[p]).abs() < 1e-7
        if int(on_plane.sum()) < 3:
            continue
        area = _polygon_area(verts[on_plane], normals[p])
        if owners[p] < 0:
            touches_box = touches_box or float(area) > 1e-12
        else:
            facet_area[owners[p]] = area
        # Cone from the site to each facet; the site is inside the cell.
        height = (offsets[p] - normals[p] @ sites[k]) / normals[p].norm()
        volume = volume + area * height / 3.0
    return {
        "volume": volume,
        "facet_area": facet_area,
        "vertices": verts,
        "interior": not touches_box,
    }
