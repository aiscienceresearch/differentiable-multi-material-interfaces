"""Write an extracted complex to Wavefront OBJ, for Blender and friends.

Two products, because they answer different questions.

`interfaces.obj` is the complex as extracted: one shared vertex block, one
named group and material per label pair, and the triple curves as OBJ line
elements. Shared vertices survive the round trip, so running Blender's "merge
by distance" on it does nothing -- which is Proposition 2 made checkable by
someone who never reads the paper. It is non-manifold along the triple curves,
by construction and not by accident.

`regions/region_k.obj` is one closed, outward-oriented surface per region. Each
is a genuine watertight solid boundary, which is what a modelling or printing
workflow usually wants, at the cost of storing shared interfaces twice.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

PALETTE = [
    (0.23, 0.43, 0.65), (0.88, 0.51, 0.17), (0.23, 0.57, 0.23),
    (0.75, 0.24, 0.24), (0.48, 0.44, 0.69), (0.52, 0.36, 0.33),
    (0.84, 0.52, 0.74), (0.43, 0.43, 0.43), (0.77, 0.64, 0.22),
    (0.23, 0.65, 0.73), (0.18, 0.29, 0.49), (0.60, 0.73, 0.35),
]


def _pair_colour(index: int) -> tuple:
    return PALETTE[index % len(PALETTE)]


def recentre(verts: np.ndarray, centre: bool):
    """Shift the bounding-box centre to the origin, and say by how much.

    The extracted coordinates are millimetres in the scanner's own frame, which
    puts the liver about 120 units from the world origin. Blender opens looking
    at the origin, so an uncentred import is invisible until you press Home.
    The offset is written into the file header so the scan frame is recoverable.
    """
    if not centre or not len(verts):
        return verts, np.zeros(3)
    offset = 0.5 * (verts.max(0) + verts.min(0))
    return verts - offset, offset


def write_interfaces(path, surf, label_name=None, centre=True) -> dict:
    """The complex with shared vertices, grouped and coloured by label pair."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = label_name or (lambda k: str(k))

    verts = surf.vertices.detach().cpu().numpy()
    tris = surf.triangles.detach().cpu().numpy()
    pairs = surf.triangle_labels.detach().cpu().numpy()
    curves = surf.triple_segments.detach().cpu().numpy()
    verts, offset = recentre(verts, centre)

    unique = sorted({tuple(p) for p in pairs.tolist()})
    mtl = path.with_suffix(".mtl")

    with open(mtl, "w", encoding="utf-8") as f:
        f.write("# one material per label pair\n")
        for i, (a, b) in enumerate(unique):
            r, g, bl = _pair_colour(i)
            f.write(f"\nnewmtl pair_{a}_{b}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {bl:.4f}\n")
            f.write("Ka 0.0 0.0 0.0\nKs 0.05 0.05 0.05\nNs 16\nd 1.0\nillum 2\n")
        f.write("\nnewmtl triple_curves\nKd 0.05 0.05 0.05\nillum 1\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Exact argmax partition of a multi-label field.\n")
        f.write("# Vertices are shared between patches: merging by distance is a no-op.\n")
        f.write("# Non-manifold along the triple curves, by construction.\n")
        f.write("# Units are millimetres. Add the offset below to return to the\n")
        f.write("# scanner's frame: x y z = "
                f"{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        f.write(f"mtllib {mtl.name}\n\n")
        for x, y, z in verts:
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")

        for a, b in unique:
            sel = (pairs[:, 0] == a) & (pairs[:, 1] == b)
            f.write(f"\no patch_{name(a)}_{name(b)}\n")
            f.write(f"g patch_{name(a)}_{name(b)}\n")
            f.write(f"usemtl pair_{a}_{b}\n")
            for t in tris[sel] + 1:
                f.write(f"f {t[0]} {t[1]} {t[2]}\n")

        if len(curves):
            f.write("\no triple_curves\ng triple_curves\nusemtl triple_curves\n")
            for u, v in curves + 1:
                f.write(f"l {u} {v}\n")

    return {"file": str(path), "vertices": len(verts), "triangles": len(tris),
            "patches": len(unique), "curve_edges": len(curves)}


def write_regions(directory, surf, label_name=None, centre=True) -> list:
    """One closed, outward-oriented surface per region.

    Every region is shifted by the same offset as `write_interfaces` uses, so
    the files still register with each other when loaded together.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = label_name or (lambda k: str(k))

    verts = surf.vertices.detach().cpu().numpy()
    verts, offset = recentre(verts, centre)
    tris = surf.triangles.detach().cpu()
    pairs = surf.triangle_labels.detach().cpu()

    written = []
    for label in sorted(set(pairs.reshape(-1).tolist())):
        involved = (pairs == label).any(dim=1)
        if not bool(involved.any()):
            continue
        # Triangles are wound low label to high, so a region that is the higher
        # of its pair sees them inside out.
        t = tris[involved]
        flip = pairs[involved, 1] == label
        t = torch.where(flip[:, None], t[:, [0, 2, 1]], t).numpy()

        used, remap = np.unique(t, return_inverse=True)
        remap = remap.reshape(t.shape)
        out = directory / f"region_{name(label)}.obj"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# closed boundary of region {name(label)}, normals outward\n")
            f.write("# millimetres; scanner-frame offset x y z = "
                    f"{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
            f.write(f"o region_{name(label)}\n")
            for x, y, z in verts[used]:
                f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for a, b, c in remap + 1:
                f.write(f"f {a} {b} {c}\n")
        written.append({"file": str(out), "label": int(label),
                        "vertices": int(len(used)), "triangles": int(len(t))})
    return written


def check_obj(path) -> dict:
    """Re-read the file and count what is actually in it."""
    v = f = l = 0
    groups, materials = set(), set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            head = line.split(" ", 1)[0]
            v += head == "v"
            f += head == "f"
            l += head == "l"
            if head == "g":
                groups.add(line.strip())
            if head == "usemtl":
                materials.add(line.strip())
    return {"v": v, "f": f, "l": l, "groups": len(groups),
            "materials": len(materials)}


def export_realdata(out="export/liver"):
    """Export the clinical example of Section 'A field nobody designed for us'."""
    import envelope3d
    import realdata
    from realdata_experiment import ORIGIN_VOX, PERTURB, RESOLUTION, WIDTH

    torch.set_default_dtype(torch.float64)
    field, box, _, _ = realdata.load_block(ORIGIN_VOX, WIDTH)
    surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=PERTURB)

    root = Path(__file__).resolve().parent.parent / out
    names = {0: "outside"}
    label_name = lambda k: names.get(k, f"segment{k}")

    summary = write_interfaces(root / "interfaces.obj", surf, label_name)
    regions = write_regions(root / "regions", surf, label_name)
    return surf, summary, regions


def export_organ(resolution: int = 128, out="export/liver_whole"):
    """The whole labelled organ, so each segment closes into its own solid.

    The $49^3$ crop of the paper cuts every region it contains, which is fine
    for measuring but gives open surfaces. Over the organ's own bounding box
    with a background margin, only the regions that run off the edge of the
    scan stay open.
    """
    import envelope3d
    import extract3d
    import realdata

    torch.set_default_dtype(torch.float64)
    field, box, size = realdata.load_organ()
    surf = envelope3d.extract(field, resolution=resolution, box=box, perturb=1e-9)
    report = extract3d.check_topology(surf)

    root = Path(__file__).resolve().parent.parent / out
    names = {0: "outside"}
    label_name = lambda k: names.get(k, f"segment{k}")
    summary = write_interfaces(root / "interfaces.obj", surf, label_name)
    regions = write_regions(root / "regions", surf, label_name)
    return surf, summary, regions, report, size


if __name__ == "__main__":
    surf, summary, regions = export_realdata()
    print("interfaces.obj:", summary)
    print("  re-read:", check_obj(summary["file"]))
    print(f"regions: {len(regions)} closed surfaces")
    for r in regions:
        print("   %-46s %6d triangles" % (os.path.basename(r["file"]), r["triangles"]))
