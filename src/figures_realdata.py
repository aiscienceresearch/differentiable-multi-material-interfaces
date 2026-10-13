"""The real-data figure: a clinical segmentation field, extracted and compared.

    realdata_liver.png    input, output, and the two methods at one junction
    realdata_liver_io.png input and output only, laid out for the main paper

The figure has to answer three questions in order: what went in, what came out,
and what is different about the junctions. Panel (a) is the data itself, a slice
of the label field with the regions named in place, so the anatomy does not have
to be inferred from a 3D shard. Panel (b) is the extracted complex, every patch
coloured by the pair of segments it separates. Panels (c) and (d) are the same
junction under both methods, same camera, same box, and coloured by the same
key -- SurfaceNets tags its cells with a label pair exactly as we tag patches.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import baseline_surfacenets as bl
import envelope3d
import extract3d
import realdata
from figures3d import FIGDIR
from realdata_experiment import ORIGIN_VOX, PERTURB, RESOLUTION, WIDTH

torch.set_default_dtype(torch.float64)

LIGHT = np.array([0.4, 0.5, 0.75])
REGION = {0: "#ebebee", 4: "#7ba3d0", 5: "#7cc196", 6: "#e08b8e",
          7: "#a99bd4", 8: "#f0ae7c"}
NAME = {0: "outside\nthe liver", 4: "segment 4", 5: "segment 5",
        6: "segment 6", 7: "segment 7", 8: "segment 8"}
WALL = ["#3a6ea5", "#e1812c", "#3a923a", "#c03d3e", "#7b6fb0",
        "#845b53", "#d684bd", "#6d6d6d", "#c4a437", "#3aa6b9", "#2f4b7c"]
OUTER = "#d0d0d6"
ELEV, AZIM = 20, -58
IO_ELEV, IO_AZIM = 55, -60


def _shade(poly, colours, floor=0.58):
    n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    lam = floor + (1.0 - floor) * np.abs(n @ LIGHT / np.linalg.norm(LIGHT))
    return np.clip(np.asarray(colours) * lam[:, None], 0, 1)


def _poly(ax, poly, colours, alpha, edge="none", lw=0.0):
    col = Poly3DCollection(poly, facecolors=_shade(poly, colours), alpha=alpha,
                           edgecolors=edge, linewidths=lw)
    col.set_sort_zpos(0)
    ax.add_collection3d(col)


def _frame(ax, pts=None, bounds=None, zoom=1.0):
    if bounds is not None:
        lo, hi = bounds
    else:
        p = np.asarray(pts, dtype=float).reshape(-1, 3)
        lo, hi = p.min(0), p.max(0)
        pad = 0.03 * (hi - lo).max()
        lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple((hi - lo) / (hi - lo).max()), zoom=zoom)
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)


def _caption(fig, ax, head, body):
    """Heading and explanation above the panel, in figure coordinates.

    Body is anchored just above the panel and grows upward; the heading sits
    a fixed gap above a three-line body. Neither can grow into the other, and
    the body cannot grow down into the panel (where the callouts live).
    """
    bb = ax.get_position()
    x = 0.5 * (bb.x0 + bb.x1)
    body_y = bb.y1 + 0.042
    fig.text(x, body_y, body, fontsize=9.0, color="#333333",
             ha="center", va="bottom", linespacing=1.55)
    fig.text(x, body_y + 0.076, head, fontsize=11.5, fontweight="bold",
             ha="center", va="bottom")


def _callout(fig, ax, text, color, *, x_frac=1.005, y_frac=0.99):
    """Annotation box inside a panel, placed in figure coordinates.

    text2D on a 3D axes is sensitive to the projection and was ending up
    against the panel caption; figure coordinates keep a stable gap. The mesh
    runs the full height of the panel down its middle, so there is no sky above
    it to sit in, but the shared zoom box leaves the right-hand third of both
    (c) and (d) empty. The box is flush with the right edge of that column, so
    the text has to be wrapped narrow enough to stay clear of the mesh.
    """
    bb = ax.get_position()
    fig.text(bb.x0 + x_frac * (bb.x1 - bb.x0),
             bb.y0 + y_frac * (bb.y1 - bb.y0),
             text, fontsize=8.6, fontweight="bold", color=color,
             ha="right", va="top", multialignment="left", linespacing=1.50,
             bbox=dict(boxstyle="round,pad=0.50", fc="white",
                       ec=color, lw=0.7, alpha=0.94),
             zorder=20)


def _outside_legend(fig, ax, handles, *, ncol=None, title=None, fontsize=9.4,
                    drop=0.007):
    """Legend in rows centred just below the panel, outside the axes.

    An in-axes legend sits on the mesh whichever corner it is given, and on a
    3D axes the collections draw over it in any case. Rows under the axes
    clear the geometry without leaving a block of dead space beside it. Each
    call must `add_artist` the result, or a later fig.legend would replace it.
    """
    bb = ax.get_position()
    leg = fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5 * (bb.x0 + bb.x1), bb.y0 - drop),
        bbox_transform=fig.transFigure,
        fontsize=fontsize, ncol=ncol or len(handles), columnspacing=1.2,
        framealpha=0.96, borderpad=0.45, handlelength=1.2,
        handletextpad=0.45, labelspacing=0.35, title=title,
        title_fontsize=(fontsize + 0.2) if title else fontsize,
    )
    if title is not None:
        leg._legend_box.sep = 6
    fig.add_artist(leg)
    return leg


def trilinear_interface_distance(field, points, chunk: int = 20000):
    """Distance in mm from each point to the interface of the trilinear field.

    Neither method is built on this interpolant: ours is defined on the P1
    interpolant of the same nodal values and SurfaceNets is defined on nothing
    continuous at all. That is the reason to use it. Since the trilinear field
    is not affine, the margin over the gradient of the margin is a first-order
    distance rather than an exact one, which is all that is needed to compare
    two meshes a fraction of a voxel apart.
    """
    out = []
    for p in torch.as_tensor(points).split(chunk):
        p = p.clone().requires_grad_(True)
        values = field.logits(p)
        order = values.topk(2, dim=1).indices
        gap = (values.gather(1, order[:, :1]) - values.gather(1, order[:, 1:2])).squeeze(1)
        grad, = torch.autograd.grad(gap.sum(), p)
        out.append((gap / grad.norm(dim=1).clamp_min(1e-30)).abs().detach())
    return torch.cat(out).numpy()


def figure_realdata_accuracy(path: str = "realdata_accuracy.png") -> None:
    field, box, block, spacing = realdata.load_block(ORIGIN_VOX, WIDTH)
    surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=PERTURB)
    labels = block.argmax(0).astype(np.int16)
    _, smooth, quads, _ = bl.surface_and_duplicates(labels, spacing)

    vx = float(min(spacing))
    lo = np.array([box[0], box[2], box[4]])
    hi = np.array([box[1], box[3], box[5]])

    def interior(points):
        """A cut through a region is not an interface, so the crop wall is out."""
        return ((points > lo + vx) & (points < hi - vx)).all(axis=-1)

    ours_v = surf.vertices.detach().cpu().numpy()
    ours_f = surf.triangles.detach().cpu().numpy()
    ours_pairs = np.sort(surf.triangle_labels.detach().cpu().numpy(), axis=1)
    sn_pairs = np.sort(bl.boundary_labels(labels, spacing), axis=1)

    # Only the segment-to-segment walls. The liver's outer shell is opaque and
    # would hide every junction behind it, and it is the one part of the
    # surface where the two methods have the least to disagree about.
    faces = [("ours", ours_v, ours_f[ours_pairs[:, 0] != 0]),
             ("vtkSurfaceNets3D", smooth, quads[sn_pairs[:, 0] != 0])]

    shown, stats = [], {}
    for name, verts, cells in faces:
        dist = trilinear_interface_distance(field, verts)
        keep = interior(verts[cells]).all(axis=1)
        shown.append((name, verts[cells][keep], dist[cells[keep]].mean(axis=1)))
        stats[name] = dist[np.unique(cells[keep])]
        print("   %-18s %6d faces, median %.4f vox, p90 %.4f vox"
              % (name, int(keep.sum()), np.median(stats[name]) / vx,
                 np.quantile(stats[name], 0.9) / vx))

    # The two large walls are named in place, as in panel (b) of the main
    # figure; the small ones run along the seam between them.
    wall = ours_f[ours_pairs[:, 0] != 0]
    wall_pairs = ours_pairs[ours_pairs[:, 0] != 0]
    keep = interior(ours_v[wall]).all(axis=1)
    big = sorted(np.unique(wall_pairs[keep], axis=0),
                 key=lambda p: -int((keep & (wall_pairs == p).all(1)).sum()))[:2]
    tags = [(f"{p[0]}\u2009|\u2009{p[1]}",
             np.median(ours_v[wall[keep & (wall_pairs == p).all(1)]].reshape(-1, 3), axis=0))
            for p in big]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "offby", ["#dfe4ea", "#ffd75e", "#f2971f", "#e02b1d", "#7d0018"])
    norm = matplotlib.colors.Normalize(0.0, 0.5)

    fig = plt.figure(figsize=(13.6, 5.4))
    gs3d = fig.add_gridspec(1, 2, left=0.005, right=0.645, top=0.79, bottom=0.14,
                            wspace=0.01)
    gsp = fig.add_gridspec(1, 1, left=0.735, right=0.985, top=0.79, bottom=0.23)
    bounds = None

    for i, (name, poly, err) in enumerate(shown):
        ax = fig.add_subplot(gs3d[0, i], projection="3d")
        colours = cmap(norm(err / vx))[:, :3]
        col = Poly3DCollection(poly, facecolors=_shade(poly, colours, floor=0.76),
                               edgecolors="none", linewidths=0.0)
        col.set_sort_zpos(0)
        ax.add_collection3d(col)
        if bounds is None:
            p = poly.reshape(-1, 3)
            pad = 0.03 * (p.max(0) - p.min(0)).max()
            bounds = (p.min(0) - pad, p.max(0) + pad)
        _frame(ax, bounds=bounds, zoom=1.30)
        ax.view_init(elev=IO_ELEV, azim=IO_AZIM)
        for text, at in tags:
            ax.text(*at, text, fontsize=8.6, fontweight="bold", ha="center",
                    va="center", zorder=100,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#555555",
                              lw=0.8, alpha=0.9))
        d = stats[name]
        _caption(fig, ax, f"({'ab'[i]})  {name}",
                 f"median {np.median(d) / vx:#.2g} voxels ({np.median(d):#.2g} mm),\n"
                 f"90th percentile {np.quantile(d, 0.9) / vx:.3f} voxels")

    cax = fig.add_axes((0.20, 0.085, 0.26, 0.024))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", extend="max")
    bar.set_label(f"distance to the trilinear interface, in voxels "
                  f"(1 voxel = {vx:.2f} mm)", fontsize=8.5)
    bar.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gsp[0, 0])
    for name, colour in (("ours", "#4c72b0"), ("vtkSurfaceNets3D", "#dd8452")):
        d = np.sort(stats[name]) / vx
        ax.plot(np.maximum(d, 1e-5), np.linspace(0, 100, d.size), lw=2.0,
                color=colour, label=name)
        ax.plot([np.median(d)], [50], "o", color=colour, ms=6, zorder=5)
    ax.axhline(50, color="#999999", lw=0.9, ls="--")
    ax.text(2.6, 51.5, "median", fontsize=8, color="#666666", ha="right", va="bottom")
    on_edge = float((stats["ours"] / vx < 1e-5).mean())
    ax.annotate(f"{100 * on_edge:.0f}% of our vertices lie on grid\n"
                f"edges parallel to an axis, where\n"
                f"the two interpolants agree;\n"
                f"they are drawn at $10^{{-5}}$",
                xy=(1.15e-5, 100 * on_edge), xytext=(1.6e-5, 6.0),
                fontsize=7.8, color="#2f4f80", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#2f4f80", lw=0.8))
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 3.0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("distance to the trilinear interface, in voxels")
    ax.set_ylabel("vertices within this distance (%)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    _caption(fig, ax, "(c)  every vertex of (a) and (b)",
             "share of the vertices that lie closer\n"
             "than the distance on the horizontal axis")

    out = os.path.join(FIGDIR, path)
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print("wrote", out)


def figure_realdata(path: str = "realdata_liver.png") -> None:
    field, box, block, spacing = realdata.load_block(ORIGIN_VOX, WIDTH)
    surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=PERTURB)
    report = extract3d.check_topology(surf)

    labels = block.argmax(0).astype(np.int16)
    raw, smooth, quads, groups = bl.surface_and_duplicates(labels, spacing)
    sn_pairs = bl.boundary_labels(labels, spacing)
    sn_pairs.sort(axis=1)

    verts = surf.vertices.detach().cpu().numpy()
    tris = surf.triangles.detach().cpu().numpy()
    pairs = surf.triangle_labels.detach().cpu().numpy()
    seg = surf.triple_segments.detach().cpu().numpy()
    poly = verts[tris]

    allpairs = sorted({tuple(p) for p in pairs.tolist()})
    internal = [p for p in allpairs if p[0] != 0]
    colour = {p: (OUTER if p[0] == 0 else WALL[internal.index(p) % len(WALL)])
              for p in allpairs}
    rgb = lambda ps: np.array([matplotlib.colors.to_rgb(
        colour.get(tuple(p), OUTER)) for p in ps])

    def swatches(present):
        """One row per internal wall, and a single row for the outer shell.

        Every `0 | k` patch is drawn in the same grey, so listing them
        separately puts identical swatches on consecutive rows and reads as a
        fault. They are collapsed the way panel (b) collapses them.
        """
        rows = [Patch(facecolor=colour.get(p, OUTER), edgecolor="#555555",
                      label=f"{p[0]}\u2009|\u2009{p[1]}")
                for p in present if p[0] != 0]
        if any(p[0] == 0 for p in present):
            rows.append(Patch(facecolor=OUTER, edgecolor="#999999",
                              label="0\u2009|\u2009k  (outer)"))
        return rows

    worst = groups[0]
    centre, vx = worst["origin"], float(min(spacing))
    radius = 2.6 * vx
    inside = lambda p: (np.linalg.norm(p - centre, axis=-1) <= radius).all(axis=-1)

    keep = inside(poly)
    qp = smooth[quads]
    keepq = inside(qp)
    local = np.concatenate([poly[keep].reshape(-1, 3), qp[keepq].reshape(-1, 3)])
    half = 0.54 * float((local.max(0) - local.min(0)).max())
    mid = 0.5 * (local.max(0) + local.min(0))
    zoombox = (mid - half, mid + half)

    fig = plt.figure(figsize=(9.9, 8.8))
    # The bottom margin is the strip the (c)/(d) legends sit in, outside the
    # axes, so they cannot cover the junction mesh.
    # The junction panels carry the finding, so the bottom row is given the
    # larger share of the height.
    gs = fig.add_gridspec(2, 2, left=0.02, right=0.98, top=0.755, bottom=0.105,
                          wspace=0.06, hspace=1.15, height_ratios=[1.0, 1.35])

    # ------------------------------------------------------------------ (a)
    ax = fig.add_subplot(gs[0, 0])
    lo = np.array([box[0], box[2], box[4]])
    kz = int(np.clip(round((centre[2] - lo[2]) / spacing[2]), 0, labels.shape[0] - 1))
    sl = labels[kz]
    present = [v for v in REGION if (sl == v).any()]
    lut = {v: i for i, v in enumerate(present)}
    table = np.array([matplotlib.colors.to_rgb(REGION[v]) for v in present])
    ax.imshow(table[np.vectorize(lut.get)(sl)], origin="lower",
              interpolation="nearest", extent=[box[0], box[1], box[2], box[3]])

    # Name each region where it sits, rather than in a legend off to one side.
    ys, xs = np.mgrid[0:sl.shape[0], 0:sl.shape[1]]
    for v in present:
        m = sl == v
        if m.sum() < 40:
            continue
        cy = box[2] + (ys[m].mean() + 0.5) * spacing[1]
        cx = box[0] + (xs[m].mean() + 0.5) * spacing[0]
        ax.text(cx, cy, NAME[v], ha="center", va="center", fontsize=9.2,
                color="#1a1a1a", linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.62))

    ax.add_patch(Rectangle((mid[0] - half, mid[1] - half), 2 * half, 2 * half,
                           fill=False, ec="#d62728", lw=2.0))
    ax.annotate("panels (c), (d)",
                (mid[0] + half, mid[1]),
                textcoords="offset points", xytext=(10, 0), fontsize=9.0,
                color="#d62728", fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.2,
                                shrinkA=0, shrinkB=2))
    ax.plot([box[0] + 2.0, box[0] + 7.0], [box[2] + 1.8] * 2, "k-", lw=2.6)
    ax.text(box[0] + 4.5, box[2] + 2.6, "5 mm", ha="center", fontsize=8.6)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    _caption(fig, ax, "(a)  what goes in",
             "one slice of the network's per-voxel class field.\n"
             "The extractor reads the probabilities behind these labels,\n"
             "sampled at voxel centers with no resampling.")

    # ------------------------------------------------------------------ (b)
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    ext = pairs[:, 0] == 0
    _poly(ax, poly[ext], rgb(pairs[ext]), alpha=0.11)
    _poly(ax, poly[~ext], rgb(pairs[~ext]), alpha=0.99)
    if len(seg):
        ax.add_collection3d(Line3DCollection(verts[seg], colors="#101010", linewidths=1.4))
    if surf.num_quadruple_points:
        q = surf.quadruple_points().detach().cpu().numpy()
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=54, c="#ffd400", edgecolors="k",
                   linewidths=0.6, depthshade=False)
    _frame(ax, pts=poly.reshape(-1, 3), zoom=1.44)
    _caption(fig, ax, "(b)  what comes out",
             f"{tris.shape[0]} triangles in {len(allpairs)} patches, each one tagged with the\n"
             f"pair of segments it separates. Watertight: {report['ok']}.")
    _outside_legend(
        fig, ax,
        swatches(allpairs)
        + [Line2D([], [], color="#101010", lw=1.8, label="triple curve"),
           Line2D([], [], color="none", marker="o", mfc="#ffd400", mec="k",
                  ms=7, label="quadruple pt")],
        ncol=5, title="patch = segment pair", fontsize=8.6)

    # ------------------------------------------------------------------ (c)
    ax = fig.add_subplot(gs[1, 0], projection="3d")
    _poly(ax, poly[keep], rgb(pairs[keep]), alpha=0.62, edge="#2a2a2a", lw=0.35)
    local_seg = seg[inside(verts[seg])] if len(seg) else seg
    if len(local_seg):
        ax.add_collection3d(Line3DCollection(verts[local_seg], colors="#101010",
                                             linewidths=4.5))
        p = verts[np.unique(local_seg)]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=58, c="#00c000", edgecolors="k",
                   linewidths=0.8, depthshade=False)
        _callout(fig, ax,
                 "the patches meet\n"
                 "along this curve,\n"
                 "and every vertex\n"
                 "on it is one index\n"
                 "shared by all\n"
                 "three of them",
                 "#00650b", x_frac=1.15)
    here = sorted({tuple(p) for p in pairs[keep].tolist()})
    _outside_legend(
        fig, ax,
        swatches(here)
        + [Line2D([], [], color="#101010", lw=3.0, label="triple curve"),
           Line2D([], [], color="none", marker="o", mfc="#00c000", mec="k",
                  ms=7, label="shared vertex")],
        ncol=3, fontsize=8.8)
    _frame(ax, bounds=zoombox, zoom=1.85)
    _caption(fig, ax, "(c)  ours, inside the red box",
             "the patches end on a single shared vertex index,\n"
             "so their separation is zero by construction, not by tolerance.")

    # ------------------------------------------------------------------ (d)
    ax = fig.add_subplot(gs[1, 1], projection="3d")
    if keepq.any():
        _poly(ax, qp[keepq], rgb(sn_pairs[keepq]), alpha=0.62, edge="#2a2a2a", lw=0.35)
    d = smooth[worst["members"]]
    ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=125, c="#e8000b", edgecolors="k",
               linewidths=0.9, depthshade=False)
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            ax.plot(*zip(d[i], d[j]), color="#e8000b", lw=2.2, ls=":")
    _callout(fig, ax,
             f"{len(d)} copies of what\n"
             f"should be one\n"
             f"vertex, pulled\n"
             f"{worst['spread']:.2f} mm apart\n"
             f"({worst['spread'] / vx:.2f} voxels) by\n"
             f"the smoothing",
             "#a00008")
    hereq = sorted({tuple(p) for p in sn_pairs[keepq].tolist()}) if keepq.any() else []
    _outside_legend(
        fig, ax,
        swatches(hereq)
        + [Line2D([], [], color="none", marker="o", mfc="#e8000b", mec="k",
                  ms=8, label="duplicated vertex")],
        ncol=3, fontsize=8.8)
    _frame(ax, bounds=zoombox, zoom=1.85)
    _caption(fig, ax, "(d)  vtkSurfaceNets3D",
             "same box, same camera. It duplicates the vertex to keep the\n"
             "mesh manifold, and its smoothing pass then pulls the copies\n"
             "apart.")

    out = os.path.join(FIGDIR, path)
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print("wrote", out)


def _box_edges(lo, hi):
    """The twelve edges of the axis-aligned box [lo, hi], as segment pairs."""
    corners = np.array([[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    return np.array([[corners[a], corners[b]]
                     for a in range(8) for b in range(a + 1, 8)
                     if np.count_nonzero(corners[a] != corners[b]) == 1])


def figure_realdata_io(path: str = "realdata_liver_io.png") -> None:
    """Input and output of the clinical extraction, as a short full-width strip.

    Panels (a) and (b) of `figure_realdata`, redrawn for the main paper, where
    the figure has to fit in a third of the text width in height. The slice in
    (a) is drawn as large as that allows. Panel (b) carries the outline of the
    crop, the plane of (a) and the junction box, so the 3D view can be read
    against the slice; the camera keeps the orientation of (a), x to the right
    and y going back, and looks down far enough that the wall behind the front
    one is not hidden. The caption describes the panels, so they carry only
    their letters and the height goes to the panels themselves.
    """
    field, box, block, spacing = realdata.load_block(ORIGIN_VOX, WIDTH)
    surf = envelope3d.extract(field, resolution=RESOLUTION, box=box, perturb=PERTURB)

    labels = block.argmax(0).astype(np.int16)
    _, smooth, quads, groups = bl.surface_and_duplicates(labels, spacing)

    verts = surf.vertices.detach().cpu().numpy()
    tris = surf.triangles.detach().cpu().numpy()
    pairs = surf.triangle_labels.detach().cpu().numpy()
    seg = surf.triple_segments.detach().cpu().numpy()
    poly = verts[tris]

    allpairs = sorted({tuple(p) for p in pairs.tolist()})
    internal = [p for p in allpairs if p[0] != 0]
    colour = {p: (OUTER if p[0] == 0 else WALL[internal.index(p) % len(WALL)])
              for p in allpairs}
    rgb = lambda ps: np.array([matplotlib.colors.to_rgb(
        colour.get(tuple(p), OUTER)) for p in ps])

    # The same junction box as panels (c) and (d) of `figure_realdata`.
    worst = groups[0]
    centre, vx = worst["origin"], float(min(spacing))
    radius = 2.6 * vx
    inside = lambda p: (np.linalg.norm(p - centre, axis=-1) <= radius).all(axis=-1)
    qp = smooth[quads]
    local = np.concatenate([poly[inside(poly)].reshape(-1, 3),
                            qp[inside(qp)].reshape(-1, 3)])
    half = 0.54 * float((local.max(0) - local.min(0)).max())
    mid = 0.5 * (local.max(0) + local.min(0))

    lo = np.array([box[0], box[2], box[4]])
    hi = np.array([box[1], box[3], box[5]])
    kz = int(np.clip(round((centre[2] - lo[2]) / spacing[2]), 0, labels.shape[0] - 1))
    z_slice = lo[2] + kz * spacing[2]

    fig = plt.figure(figsize=(8.4, 2.8))
    SLICE, PLANE, JUNCTION = "#333333", "#333333", "#d62728"

    def letter(x, text):
        fig.text(x, 0.90, text, fontsize=11.0, fontweight="bold",
                 ha="left", va="top")

    # ------------------------------------------------------------------ (a)
    ax = fig.add_axes((0.055, 0.11, 0.24, 0.79))
    sl = labels[kz]
    present = [v for v in REGION if (sl == v).any()]
    lut = {v: i for i, v in enumerate(present)}
    table = np.array([matplotlib.colors.to_rgb(REGION[v]) for v in present])
    ax.imshow(table[np.vectorize(lut.get)(sl)], origin="lower",
              interpolation="nearest", extent=[box[0], box[1], box[2], box[3]])
    ys, xs = np.mgrid[0:sl.shape[0], 0:sl.shape[1]]
    for v in present:
        m = sl == v
        if m.sum() < 40:
            continue
        cy = box[2] + (ys[m].mean() + 0.5) * spacing[1]
        cx = box[0] + (xs[m].mean() + 0.5) * spacing[0]
        ax.text(cx, cy, NAME[v], ha="center", va="center", fontsize=9.0,
                color="#1a1a1a", linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.62))
    ax.add_patch(Rectangle((mid[0] - half, mid[1] - half), 2 * half, 2 * half,
                           fill=False, ec=JUNCTION, lw=1.8))
    ax.annotate("junction,\nFig. S13", (mid[0] + half, mid[1] - half),
                textcoords="offset points", xytext=(9, -8), fontsize=8.4,
                color=JUNCTION, fontweight="bold", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
                arrowprops=dict(arrowstyle="-", color=JUNCTION, lw=1.1,
                                shrinkA=0, shrinkB=1))
    ax.plot([box[0] + 2.0, box[0] + 7.0], [box[2] + 1.8] * 2, "k-", lw=2.4)
    ax.text(box[0] + 4.5, box[2] + 2.6, "5 mm", ha="center", fontsize=8.4)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(SLICE)
        s.set_linestyle((0, (4, 2.5)))
        s.set_linewidth(1.0)
    letter(0.014, "(a)")

    # ------------------------------------------------------------------ (b)
    ax = fig.add_axes((0.315, -0.05, 0.35, 1.10), projection="3d")
    ext = pairs[:, 0] == 0
    _poly(ax, poly[ext], rgb(pairs[ext]), alpha=0.16)
    _poly(ax, poly[~ext], rgb(pairs[~ext]), alpha=0.99)
    if len(seg):
        ax.add_collection3d(Line3DCollection(verts[seg], colors="#101010", linewidths=1.2))
    if surf.num_quadruple_points:
        q = surf.quadruple_points().detach().cpu().numpy()
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=34, c="#ffd400", edgecolors="k",
                   linewidths=0.6, depthshade=False)
    ax.add_collection3d(Line3DCollection(_box_edges(lo, hi), colors="#9a9a9a",
                                         linewidths=0.7))
    plane = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]],
                      [lo[0], hi[1]], [lo[0], lo[1]]])
    ax.plot(plane[:, 0], plane[:, 1], np.full(5, z_slice), color=PLANE, lw=1.0,
            ls=(0, (4, 2.5)))
    jlo = np.array([mid[0] - half, mid[1] - half, mid[2] - half])
    jhi = jlo + 2 * half
    ax.add_collection3d(Line3DCollection(_box_edges(jlo, jhi), colors=JUNCTION,
                                         linewidths=1.3))
    # Name the two large walls in place; the legend covers the small ones.
    for p in sorted(internal, key=lambda p: -int((pairs == p).all(1).sum()))[:2]:
        at = np.median(poly[(pairs == p).all(1)].reshape(-1, 3), axis=0)
        ax.text(*at, f"{p[0]}\u2009|\u2009{p[1]}", fontsize=8.6, fontweight="bold",
                ha="center", va="center", zorder=100,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=colour[p],
                          lw=1.0, alpha=0.9))
    _frame(ax, bounds=(lo, hi), zoom=0.9)
    ax.view_init(elev=IO_ELEV, azim=IO_AZIM)
    letter(0.335, "(b)")

    rows = [Patch(facecolor=colour[p], edgecolor="#555555",
                  label=f"{p[0]}\u2009|\u2009{p[1]}") for p in internal]
    rows.append(Patch(facecolor=OUTER, edgecolor="#999999",
                      label="0\u2009|\u2009k  (liver surface)"))
    rows += [Line2D([], [], color="#101010", lw=1.6, label="triple curve"),
             Line2D([], [], color="none", marker="o", mfc="#ffd400", mec="k",
                    ms=6.5, label="quadruple point"),
             Line2D([], [], color=PLANE, lw=1.0, ls=(0, (4, 2.5)), label="plane of (a)"),
             Line2D([], [], color=JUNCTION, lw=1.3, label="junction box")]
    fig.legend(handles=rows, loc="center left", bbox_to_anchor=(0.675, 0.5),
               bbox_transform=fig.transFigure, ncol=2, fontsize=8.4,
               title="each patch is tagged with the\npair of segments it separates",
               title_fontsize=8.6, framealpha=0.96, borderpad=0.5,
               handlelength=1.4, handletextpad=0.5, labelspacing=0.45,
               columnspacing=1.2)

    out = os.path.join(FIGDIR, path)
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    figure_realdata()
    figure_realdata_io()
    figure_realdata_accuracy()
