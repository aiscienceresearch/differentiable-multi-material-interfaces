"""Compare against vtkSurfaceNets3D on the same real segmentation.

SurfaceNets is the right baseline rather than per-label marching cubes: it is a
single multi-label pass, it shares points and cells between adjacent regions,
and it leaves no gaps. Its own documentation states the one thing it cannot do:
"In the presence of locally non-manifold configurations, points may be
selectively duplicated (initially coincident) in order to avoid creating
non-manifold edges/vertices. After smoothing, duplicated points may diverge."

That is measurable. Running the filter twice on identical input, once with
smoothing off and once on, leaves the point ordering unchanged, so the groups
of duplicated points can be identified from the unsmoothed output (where they
are coincident) and their separation read off the smoothed one. Our triple-curve
vertices are single shared indices, so the same quantity is identically zero.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def _image(labels: np.ndarray, spacing) -> vtk.vtkImageData:
    """Wrap a (nz, ny, nx) label block as vtkImageData with (x, y, z) spacing."""
    nz, ny, nx = labels.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*spacing)
    # Match realdata.load_block: the block centre sits at the origin.
    img.SetOrigin(-0.5 * (nx - 1) * spacing[0],
                  -0.5 * (ny - 1) * spacing[1],
                  -0.5 * (nz - 1) * spacing[2])
    flat = np.ascontiguousarray(labels.reshape(-1).astype(np.int16))
    arr = numpy_to_vtk(flat, deep=True)
    arr.SetName("labels")
    img.GetPointData().SetScalars(arr)
    return img


def _run(img, present, smoothing: bool):
    net = vtk.vtkSurfaceNets3D()
    net.SetInputData(img)
    net.SetBackgroundLabel(0)
    net.SetNumberOfLabels(len(present))
    for i, value in enumerate(present):
        net.SetLabel(i, float(value))
    # Quads, not triangles: the triangulation picks a diagonal from the geometry,
    # so smoothing would change the connectivity and defeat the index matching.
    net.SetOutputMeshTypeToQuads()
    if smoothing:
        net.SmoothingOn()
    else:
        net.SmoothingOff()
    net.Update()
    out = net.GetOutput()
    points = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    cells = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).astype(np.int64)
    return points, cells, out


def boundary_labels(labels: np.ndarray, spacing) -> np.ndarray:
    """The (label0, label1) pair the filter assigns to each output cell.

    SurfaceNets tags its cells with the pair they separate, the same way we tag
    patches, so the two meshes can be coloured by the same key.
    """
    present = sorted(int(v) for v in np.unique(labels) if v != 0)
    _, _, out = _run(_image(labels, spacing), present, smoothing=True)
    arr = out.GetCellData().GetArray("BoundaryLabels")
    if arr is None:
        return np.zeros((out.GetNumberOfCells(), 2), dtype=np.int64)
    return vtk_to_numpy(arr).astype(np.int64).reshape(-1, 2)


def surface_and_duplicates(labels: np.ndarray, spacing):
    """Run the filter twice and pair up the points it duplicated.

    With smoothing off the duplicates are still coincident, which is how they
    can be found at all; with it on they have moved. The point ordering is
    identical between the runs, so the groups transfer by index.
    """
    present = sorted(int(v) for v in np.unique(labels) if v != 0)
    img = _image(labels, spacing)

    raw, cells_raw, _ = _run(img, present, smoothing=False)
    smooth, cells_smooth, _ = _run(img, present, smoothing=True)

    if raw.shape != smooth.shape or not np.array_equal(cells_raw, cells_smooth):
        raise RuntimeError("smoothing changed the mesh combinatorics; "
                           "the two runs are not comparable")

    keys = np.round(raw / 1e-9).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    inverse = inverse.reshape(-1)

    groups = []
    for g in np.flatnonzero(counts > 1):
        members = np.flatnonzero(inverse == g)
        p = smooth[members]
        groups.append({
            "members": members,
            "origin": raw[members[0]],
            "spread": float(np.linalg.norm(p[:, None] - p[None], axis=-1).max()),
        })
    groups.sort(key=lambda d: -d["spread"])
    quads = cells_raw.reshape(-1, 4)
    return raw, smooth, quads, groups


def duplicated_point_divergence(labels: np.ndarray, spacing):
    """Identify SurfaceNets' duplicated points and measure how far they separate."""
    raw, _, quads, groups = surface_and_duplicates(labels, spacing)
    spreads = np.array([g["spread"] for g in groups]) if groups else np.zeros(0)
    return {
        "points": int(raw.shape[0]),
        "quads": int(quads.shape[0]),
        "duplicated_groups": len(groups),
        "duplicated_points": int(sum(len(g["members"]) for g in groups)),
        "max_divergence_mm": float(spreads.max()) if spreads.size else 0.0,
        "mean_divergence_mm": float(spreads.mean()) if spreads.size else 0.0,
        "voxel_mm": float(min(spacing)),
    }
