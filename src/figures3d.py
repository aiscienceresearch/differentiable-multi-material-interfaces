"""Figures for the 3D multi-label extractor.

Each figure is meant to carry one claim of the paper on its own, so that the
text can point at it rather than describe it:

    arrangement_3d.png        what the extractor produces
    exploded_regions.png      the output is a set of closed labelled regions
    tetrahedra.png            the background grid and the solves it induces
    corner_label_failure.png  why the obvious per-cell case analysis fails
    multi_feature_cells.png   the configurations that case analysis cannot name
    triple_curves_120.png     triple curves meet at Plateau's angle
    convergence.png           exact on power diagrams, 2nd order on curved ones
    interface_accuracy.png    how far each method's surface is from the truth
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import envelope3d
import extract3d
import voronoi_exact
from fields import (AnnulusField, MultiLabelField, PowerDiagramField,
                    SectorField, simplex_directions)

torch.set_default_dtype(torch.float64)

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
SITES = torch.tensor([
    [0.35, 0.20, 0.10], [-0.30, 0.25, -0.15], [0.05, -0.40, 0.22],
    [-0.15, -0.10, 0.45], [0.10, 0.05, -0.48],
])
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
           "#937860", "#da8bc3", "#8c8c8c"]



# Reuse the validation suite's field so that every number quoted in a figure
# is the same number the checks report.
from validate3d import _enclosing_cell_field as enclosing_cell_field


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def _style(ax, title: str, subtitle: str = "", lim=1.0, fit=None) -> None:
    """Frame the axes on the geometry actually drawn, keeping it cubic.

    A 3D axis padded to the whole domain wastes most of the panel, since an
    interface complex occupies only the middle of its box.
    """
    if fit is not None:
        p = np.asarray(fit, dtype=float).reshape(-1, 3)
        centre = 0.5 * (p.max(0) + p.min(0))
        lim = 0.55 * float((p.max(0) - p.min(0)).max())
    else:
        centre = np.zeros(3)
    ax.set_xlim(centre[0] - lim, centre[0] + lim)
    ax.set_ylim(centre[1] - lim, centre[1] + lim)
    ax.set_zlim(centre[2] - lim, centre[2] + lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=10, pad=-2)
    ax.view_init(elev=22, azim=-52)


def drawn_points(surf, keep=None) -> np.ndarray:
    tris = surf.triangles.detach().cpu().numpy()
    if keep is not None:
        tris = tris[np.asarray(keep)]
    return surf.vertices.detach().cpu().numpy()[tris].reshape(-1, 3)


def draw_patches(ax, surf, colour_by="low", alpha=0.9, keep=None,
                 highlight=None, shade=True, edge="#ffffff", lw=0.12):
    """Fill every extracted triangle, coloured by the labels it separates."""
    tris = surf.triangles.detach().cpu().numpy()
    verts = surf.vertices.detach().cpu().numpy()
    pairs = surf.triangle_labels.detach().cpu().numpy()
    sel = np.ones(len(tris), dtype=bool) if keep is None else np.asarray(keep)
    tris, pairs = tris[sel], pairs[sel]
    poly = verts[tris]

    which = pairs[:, 0] if colour_by == "low" else pairs[:, 1]
    colours = np.array([matplotlib.colors.to_rgb(PALETTE[i % len(PALETTE)])
                        for i in which])
    if shade:
        # Cheap Lambert term so that coplanar patches stay distinguishable.
        n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        light = np.array([0.4, 0.5, 0.75])
        lam = 0.55 + 0.45 * np.abs(n @ light / np.linalg.norm(light))
        colours = np.clip(colours * lam[:, None], 0, 1)
    if highlight is not None:
        hot = np.asarray(highlight)[sel]
        colours[hot] = matplotlib.colors.to_rgb("#e8000b")

    col = Poly3DCollection(poly, facecolors=colours, alpha=alpha,
                           edgecolors=edge, linewidths=lw)
    col.set_sort_zpos(0)
    ax.add_collection3d(col)
    return col


def draw_curves(ax, surf, colour="#101010", lw=1.9, points=True):
    """Triple curves as lines, quadruple points as dots."""
    seg = surf.triple_segments.detach().cpu().numpy()
    verts = surf.vertices.detach().cpu().numpy()
    if len(seg):
        ax.add_collection3d(Line3DCollection(verts[seg], colors=colour, linewidths=lw))
    if points and surf.num_quadruple_points:
        q = surf.quadruple_points().detach().cpu().numpy()
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=26, c="#ffd400",
                   edgecolors="#101010", linewidths=0.6, depthshade=False, zorder=5)
    return seg.shape[0]


def region_triangles(surf, label: int):
    """Triangles bounding one region, wound so their normals point out of it."""
    involved = (surf.triangle_labels == label).any(dim=1)
    tris = surf.triangles[involved]
    flip = surf.triangle_labels[involved, 1] == label
    tris = torch.where(flip[:, None], tris[:, [0, 2, 1]], tris)
    return tris, involved


# --------------------------------------------------------------------------
# 1. What the extractor produces
# --------------------------------------------------------------------------
def figure_arrangement() -> str:
    fld = PowerDiagramField(SITES)
    fig = plt.figure(figsize=(13.0, 4.6))
    for i, res in enumerate((8, 16, 32)):
        surf = envelope3d.extract(fld, resolution=res, box=BOX)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        draw_patches(ax, surf, alpha=0.97, lw=0.0 if res > 16 else 0.15)
        ncurve = draw_curves(ax, surf)
        _style(ax, f"grid {res}x{res}x{res}",
               f"{surf.triangles.shape[0]} triangles, {ncurve} triple segments, "
               f"{surf.num_quadruple_points} quadruple points",
               fit=drawn_points(surf))

    fig.suptitle("Interfaces of a 5-region power diagram, coloured by the label pair they "
                 "separate.  Triple curves black, quadruple points yellow.\n"
                 "The geometry is piecewise planar, so it is recovered exactly at every "
                 "resolution; only the triangle count changes.", fontsize=10.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, "arrangement_3d.png")


# --------------------------------------------------------------------------
# 2. The output is a set of closed labelled regions
# --------------------------------------------------------------------------
def figure_exploded() -> str:
    fig = plt.figure(figsize=(11.5, 5.0))

    # Left: one region against the polytope computed analytically from the sites.
    fld = enclosing_cell_field()
    surf = envelope3d.extract(fld, resolution=20, box=BOX)
    exact = voronoi_exact.cell_geometry(fld.sites.detach(), fld.weights.detach(), 0, BOX)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _, involved = region_triangles(surf, 0)
    draw_patches(ax, surf, keep=involved.cpu().numpy(), alpha=1.0, edge="#404040", lw=0.25)
    corners = exact["vertices"].detach().cpu().numpy()
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=30, c="#ffd400",
               edgecolors="#101010", linewidths=0.7, depthshade=False, zorder=6)
    vol = float(surf.enclosed_volume(0).detach())
    ev = float(exact["volume"])
    _style(ax, "one region, with its exact corners overlaid",
           f"volume {vol:.10f}\nexact  {ev:.10f}   (relative error "
           f"{abs(vol - ev) / ev:.0e})", fit=drawn_points(surf, involved.cpu().numpy()))

    # Right: every region of a 5-label diagram, pulled off the common centre.
    fld = PowerDiagramField(SITES)
    surf = envelope3d.extract(fld, resolution=16, box=BOX)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    verts = surf.vertices.detach()
    light = np.array([0.4, 0.5, 0.75])
    light = light / np.linalg.norm(light)
    shown = []
    for label in range(int(surf.node_labels.max()) + 1):
        tris, _ = region_triangles(surf, label)
        if tris.numel() == 0:
            continue
        p = verts[tris]
        centre = p.reshape(-1, 3).mean(0)
        poly = (p + 0.5 * centre / centre.norm().clamp_min(1e-9)).cpu().numpy()
        shown.append(poly.reshape(-1, 3))
        n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        lam = 0.55 + 0.45 * np.abs(n @ light)
        base = np.array(matplotlib.colors.to_rgb(PALETTE[label % len(PALETTE)]))
        ax.add_collection3d(Poly3DCollection(
            poly, facecolors=np.clip(base * lam[:, None], 0, 1),
            edgecolors="#ffffff", linewidths=0.08))
    _style(ax, "the five regions of the diagram, pulled apart",
           "each is its own oriented surface, and neighbours share\n"
           "their interface vertex for vertex, so they reassemble exactly",
           fit=np.concatenate(shown))

    fig.suptitle("The output is a set of labelled regions, not a single surface: every "
                 "region is closed and coherently oriented,\nand an interface is stored "
                 "once and referenced by both regions that own it.", fontsize=10.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _save(fig, "exploded_regions.png")


# --------------------------------------------------------------------------
# 3. Why per-cell case analysis fails
# --------------------------------------------------------------------------
def figure_failure() -> str:
    fld = enclosing_cell_field()
    res = 16
    old = extract3d.extract(fld, resolution=res, box=BOX)
    new = envelope3d.extract(fld, resolution=res, box=BOX)
    exact = voronoi_exact.cell_geometry(fld.sites.detach(), fld.weights.detach(), 0, BOX)
    ev, ea = float(exact["volume"]), sum(float(a) for a in exact["facet_area"].values())

    fig = plt.figure(figsize=(13.2, 5.0))
    for i, (label, surf) in enumerate((("corner-label case analysis", old),
                                       ("exact arrangement", new))):
        _, involved = region_triangles(surf, 0)
        fitted = drawn_points(surf, involved.cpu().numpy())
        # A triangle is suspect when it was solved outside the cell that claimed
        # it, which is exactly the situation the case analysis cannot detect.
        over = surf.vertex_overshoot[surf.triangles].max(dim=1).values > 1e-9
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        # White rather than grey: the Lambert term takes some facets darker
        # than a mid-grey edge, and the edges disappear there.
        draw_patches(ax, surf, keep=involved.cpu().numpy(),
                     highlight=over.cpu().numpy(), alpha=1.0, edge="#ffffff", lw=0.3)
        vol = abs(float(surf.enclosed_volume(0).detach()) - ev) / ev
        area = abs(float(surf.triangle_areas()[involved].sum().detach()) - ea) / ea
        bad = int(over[involved].sum())
        _style(ax, label,
               f"volume error {vol:.2e},  area error {area:.2e}\n"
               f"{bad} triangles built from out-of-cell solves", fit=fitted)

    # Both fields, because the table in the paper reports the five-site
    # diagram while the panels beside this one show the enclosing-cell field.
    # Plotting one and tabulating the other reads as a contradiction.
    ax = fig.add_subplot(1, 3, 3)
    xs = [8, 16, 32, 48]
    for name, f, colour, style in (
            ("the field at left", fld, "#c44e52", "o-"),
            ("the five-site diagram\ntabulated in the paper",
             PowerDiagramField(SITES), "#4c72b0", "s--")):
        cert = [extract3d.certify(f, resolution=r, box=BOX) for r in xs]
        frac = [100.0 * c["faces_without_interior_triple_point"] /
                max(c["three_label_faces"], 1) for c in cert]
        ax.plot(xs, frac, style, color=colour, lw=2, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0, 100)
    ax.set_xlabel("grid resolution")
    ax.set_ylabel("percent of affected cells")
    ax.set_title("refinement does not remove it", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.annotate("a triple curve is 1-dimensional, so the faces it\n"
                "merely passes near grow as fast as the faces\n"
                "it actually crosses: refinement cannot help",
                xy=(0.5, 0.06), xycoords="axes fraction", fontsize=8,
                ha="center", color="#444444")

    fig.suptitle("Reading each cell's structure off its corner labels is an assumption, and "
                 "in 3D it is wrong for a large fraction of three-label faces.\nRed "
                 "triangles were assembled from equal-logit solves that landed outside the "
                 "cell claiming them; they cluster along the triple curves.",
                 fontsize=10.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, "corner_label_failure.png")


# --------------------------------------------------------------------------
# 4. Configurations a one-per-cell analysis cannot name
# --------------------------------------------------------------------------
def _grid_logits(fld: MultiLabelField, res: int):
    nodes, tets = extract3d.build_tetrahedral_grid(res, BOX)
    with torch.no_grad():
        return nodes, tets, fld.logits(nodes)


def figure_multi_feature() -> str:
    """The two cell configurations a one-feature-per-cell analysis cannot name.

    Both are read straight off the interpolant, so the panels are statements
    about the input rather than about any particular extractor.
    """
    fld = PowerDiagramField(SITES)
    res = 20
    nodes, tets, L = _grid_logits(fld, res)
    edge_table = torch.tensor(extract3d.EDGE_TABLE)
    face_table = torch.tensor(extract3d.FACE_TABLE)
    edge_nodes, *_ = extract3d._unique_entities(tets[:, edge_table], nodes.shape[0])
    face_nodes, *_ = extract3d._unique_entities(tets[:, face_table], nodes.shape[0])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # ---- an edge whose argmax changes label twice ------------------------
    lo, hi = L[edge_nodes[:, 0]], L[edge_nodes[:, 1]]
    ts = torch.linspace(0.0, 1.0, 129)
    along = lo[None] + ts[:, None, None] * (hi - lo)[None]       # (T, E, K)
    arg = along.argmax(dim=2)                                    # (T, E)
    seen = torch.zeros((edge_nodes.shape[0], L.shape[1]), dtype=torch.bool)
    seen.scatter_(1, arg.T, True)
    triple_edges = (seen.sum(dim=1) >= 3).nonzero().reshape(-1)
    # Prefer the edge whose middle interval is widest, so the figure is legible.
    widths = []
    for e in triple_edges.tolist():
        col = arg[:, e]
        widths.append(min(float((col == v).double().mean()) for v in col.unique().tolist()))
    e = int(triple_edges[int(np.argmax(widths))])

    labels_here = seen[e].nonzero().reshape(-1).tolist()
    t = ts.numpy()
    for k in labels_here:
        ax0.plot(t, along[:, e, k].numpy(), lw=2.0,
                 color=PALETTE[k % len(PALETTE)], label=f"label {k}")
    winner = arg[:, e].numpy()
    ymin, ymax = ax0.get_ylim()
    for k in labels_here:
        ax0.fill_between(t, ymin, ymax, where=(winner == k), alpha=0.13,
                         color=PALETTE[k % len(PALETTE)], linewidth=0)
    switches = np.nonzero(np.diff(winner))[0]
    runs = [int(winner[0])] + [int(winner[s + 1]) for s in switches]
    for s in switches:
        x = 0.5 * (t[s] + t[s + 1])
        ax0.axvline(x, color="#101010", ls="--", lw=1.2)
        ax0.plot([x], [along[s, e, winner[s]].item()], "o", color="#101010", ms=6,
                 zorder=6)
    for k, (a, b) in zip(runs, zip([0.0] + list(t[switches + 1]),
                                   list(t[switches]) + [1.0])):
        ax0.annotate(f"{k} wins", (0.5 * (a + b), ymax), textcoords="offset points",
                     xytext=(0, -13), ha="center", fontsize=9, weight="bold",
                     color=PALETTE[k % len(PALETTE)])
    ax0.set_ylim(ymin, ymax)
    ax0.set_xlim(0, 1)
    ax0.set_xlabel("position along one grid edge")
    ax0.set_ylabel("interpolated logit")
    ax0.set_title(f"a grid edge carrying {len(switches)} crossings\n"
                  f"the argmax runs {' then '.join(str(r) for r in runs)}, so label "
                  f"{runs[1]} separates the endpoints", fontsize=10)
    ax0.grid(alpha=0.3)
    ax0.annotate("placing one crossing per mixed edge cannot represent this,\n"
                 "whatever the case table says",
                 xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8.5, color="#333333")

    # ---- a face with three corner labels but no interior triple point ----
    fl = L[face_nodes]                                           # (F, 3, K)
    corner = fl.argmax(dim=2)                                    # (F, 3)
    distinct = ((corner[:, 0] != corner[:, 1]) & (corner[:, 1] != corner[:, 2])
                & (corner[:, 0] != corner[:, 2])).nonzero().reshape(-1)
    coords, ok = envelope3d._solve_equal_logits(fl[distinct], corner[distinct])
    bary = torch.cat([1.0 - coords.sum(1, keepdim=True), coords], dim=1)
    # Candidates: the tie lies outside the face, but not so far outside that a
    # figure has to be drawn at the wrong scale to show it.
    depth = bary.amin(dim=1)
    cand = (ok & (depth < -1e-12) & (depth > -0.9)).nonzero().reshape(-1)
    # Among those, prefer the face where all three regions are substantial, so
    # that the picture shows a genuine three-label face rather than a sliver.
    step = 1.0 / 12
    grid = torch.tensor([[i * step, j * step, 1 - (i + j) * step]
                         for i in range(13) for j in range(13 - i)])
    sampled = torch.einsum("sc,fck->sfk", grid, fl[distinct[cand]]).argmax(dim=2)
    share = torch.stack([(sampled == corner[distinct[cand], c]).double().mean(dim=0)
                         for c in range(3)])
    row = int(cand[int(share.amin(dim=0).argmax())])
    f = int(distinct[row])

    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2]])
    tie = bary[row].numpy() @ ref
    pad = 0.14
    x0 = min(ref[:, 0].min(), tie[0]) - pad
    x1 = max(ref[:, 0].max(), tie[0]) + pad
    y0 = min(ref[:, 1].min(), tie[1]) - pad
    y1 = max(ref[:, 1].max(), tie[1]) + pad
    n = 420
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    # Cartesian -> barycentric on the reference triangle.
    M = np.array([[ref[1, 0] - ref[0, 0], ref[2, 0] - ref[0, 0]],
                  [ref[1, 1] - ref[0, 1], ref[2, 1] - ref[0, 1]]])
    uv = np.linalg.solve(M, np.stack([gx.ravel() - ref[0, 0], gy.ravel() - ref[0, 1]]))
    b = np.stack([1.0 - uv[0] - uv[1], uv[0], uv[1]])            # (3, n*n)
    field_vals = b.T @ fl[f].numpy()                             # (n*n, K)
    win = field_vals.argmax(axis=1).reshape(n, n)
    cmap = matplotlib.colors.ListedColormap(
        [PALETTE[k % len(PALETTE)] for k in range(L.shape[1])])
    inside_tri = (b.min(axis=0) >= 0).reshape(n, n)
    # Outside the face, show the continuation faintly: it is where the case
    # analysis would have put a vertex.
    ax1.imshow(np.where(inside_tri, win, np.nan), origin="lower",
               extent=(x0, x1, y0, y1), cmap=cmap, vmin=-0.5, vmax=L.shape[1] - 0.5,
               alpha=0.9, interpolation="nearest")
    ax1.imshow(np.where(inside_tri, np.nan, win), origin="lower",
               extent=(x0, x1, y0, y1), cmap=cmap, vmin=-0.5, vmax=L.shape[1] - 0.5,
               alpha=0.22, interpolation="nearest")
    ax1.plot(*np.append(ref, ref[:1], axis=0).T, color="#101010", lw=2.0)
    offsets = ((-6, -16), (6, -16), (0, 12))
    for i, k in enumerate(corner[f].tolist()):
        ax1.annotate(f"label {k}", ref[i], textcoords="offset points",
                     xytext=offsets[i], ha="center", fontsize=10,
                     color=PALETTE[k % len(PALETTE)], weight="bold")
    ax1.plot(*tie, "x", color="#e8000b", ms=14, mew=3.0)
    ax1.annotate("the three labels do tie,\nbut they tie out here",
                 tie, textcoords="offset points", xytext=(-13, -6), ha="right",
                 fontsize=9, color="#e8000b", weight="bold")
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y0, y1)
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    ax1.set_title("a grid face carrying three labels that never meet inside it\n"
                  "the argmax partition has no interior vertex, so this face owes\n"
                  "the surface no triple point at all", fontsize=10)

    fig.suptitle("Two configurations that occur in ordinary inputs and that no "
                 "one-feature-per-cell case table can express.\nBoth are properties of "
                 "the interpolant itself, read off before any extraction takes place.",
                 fontsize=10.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, "multi_feature_cells.png")


# --------------------------------------------------------------------------
# 5. Plateau's angle along triple curves
# --------------------------------------------------------------------------
def figure_plateau() -> str:
    fld = SectorField(centre=(0.05, -0.02, 0.03), directions=simplex_directions(3))
    surf = envelope3d.extract(fld, resolution=16, box=BOX)
    angles = [a for row in extract3d.triple_curve_dihedrals(fld, surf) for a in row]

    fig = plt.figure(figsize=(11.5, 5.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw_patches(ax, surf, alpha=1.0)
    ncurve = draw_curves(ax, surf, lw=2.4)
    _style(ax, "four regions meeting at a point",
           f"{ncurve} triple segments along 6 curves", fit=drawn_points(surf))

    ax = fig.add_subplot(1, 2, 2)
    dev = [a - 120.0 for a in angles]
    ax.hist(dev, bins=31, color="#4c72b0", edgecolor="white")
    ax.axvline(0.0, color="#c44e52", lw=1.6)
    ax.set_xlabel("dihedral angle minus 120 degrees")
    ax.set_ylabel("count")
    ax.set_title(f"{len(angles)} measured dihedral angles\n"
                 f"worst deviation {max(abs(d) for d in dev):.2e} degrees", fontsize=10)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3)

    fig.suptitle("Along every triple curve the three patches meet at Plateau's angle to "
                 "machine precision.\nNothing imposes it; it follows from extracting the "
                 "interfaces as level sets of logit differences.", fontsize=10.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _save(fig, "triple_curves_120.png")


# --------------------------------------------------------------------------
# 6. Accuracy
# --------------------------------------------------------------------------
def figure_convergence() -> str:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left: exact on a power diagram, where the true answer is known in closed form.
    fld = enclosing_cell_field()
    exact = voronoi_exact.cell_geometry(fld.sites.detach(), fld.weights.detach(), 0, BOX)
    ev = float(exact["volume"])
    res = [8, 16, 32, 48]
    curves = {
        "corner-label case analysis": (extract3d, {}, "#c44e52", "o-"),
        "exact arrangement": (envelope3d, {"perturb": 0.0}, "#4c72b0", "s-"),
        "exact arrangement, perturbed 1e-9": (envelope3d, {"perturb": 1e-9}, "#55a868", "^--"),
    }
    for label, (module, kwargs, colour, style) in curves.items():
        err = []
        for r in res:
            s = module.extract(fld, resolution=r, box=BOX, **kwargs)
            err.append(max(abs(float(s.enclosed_volume(0).detach()) - ev) / ev, 1e-17))
        ax0.plot(res, err, style, color=colour, lw=1.8, ms=5, label=label)
    ax0.set_xscale("log", base=2)
    ax0.set_yscale("log")
    ax0.set_xticks(res)
    ax0.set_xticklabels([str(r) for r in res])
    ax0.set_xlabel("grid resolution")
    ax0.set_ylabel("relative volume error")
    ax0.set_title("piecewise-planar geometry: exact, not convergent", fontsize=10)
    ax0.grid(alpha=0.3, which="both")
    ax0.set_ylim(1e-18, 1e1)
    ax0.legend(fontsize=8, loc="center left")
    ax0.annotate("exact to the last bit at every resolution;\nthe dashed line is the cost "
                 "of the symbolic\nperturbation and nothing more",
                 xy=(0.04, 0.16), xycoords="axes fraction", fontsize=8, color="#444444")

    # Right: second order on curved interfaces, where exactness is impossible.
    shells = AnnulusField(dim=3)
    radii = shells.radii.detach()
    target = float(4.0 * math.pi * (0.5 * (radii[0] + radii[1])) ** 2)
    res2 = [16, 24, 32, 48]
    err2 = []
    for r in res2:
        s = envelope3d.extract(shells, resolution=r, box=BOX)
        pair = (s.triangle_labels == torch.tensor([0, 1])).all(dim=1)
        err2.append(abs(float(s.triangle_areas()[pair].sum().detach()) - target) / target)
    slope = np.polyfit(np.log(res2), np.log(err2), 1)[0]
    ax1.plot(res2, err2, "o-", color="#4c72b0", lw=1.8, ms=5, label="sphere area error")
    ref = [err2[0] * (res2[0] / r) ** 2 for r in res2]
    ax1.plot(res2, ref, "--", color="#8c8c8c", lw=1.4, label="second order reference")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(res2)
    ax1.set_xticklabels([str(r) for r in res2])
    ax1.set_xlabel("grid resolution")
    ax1.set_ylabel("relative area error")
    ax1.set_title(f"curved geometry: second order (fitted rate {-slope:.2f})", fontsize=10)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    fig.suptitle("Accuracy against closed-form ground truth: an exactly known power-diagram "
                 "cell, and a sphere of known radius.", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save(fig, "convergence.png")


# --------------------------------------------------------------------------
# 7. How far the surface actually is from the interface it claims
# --------------------------------------------------------------------------
def true_interface_distance(fld: PowerDiagramField, points) -> torch.Tensor:
    """Exact distance from each point to the boundary of the argmax partition.

    Every pairwise difference f_i - f_j of a power diagram is affine, so the
    margin between the two winning logits divided by the gradient of their
    difference is not a linearised estimate: it is the distance to the bisector
    the point is closest to crossing. A point on the interface returns zero.
    """
    sites, weights = fld.sites.detach(), fld.weights.detach()
    p = torch.as_tensor(points, dtype=sites.dtype)
    f = -(p[:, None, :] - sites[None]).pow(2).sum(-1) + weights[None]
    top, idx = f.topk(2, dim=1)
    sep = 2.0 * (sites[idx[:, 0]] - sites[idx[:, 1]]).norm(dim=1)
    return ((top[:, 0] - top[:, 1]) / sep).abs()


def _surfacenets_region(fld: PowerDiagramField, res: int, label: int = 0):
    """vtkSurfaceNets3D on the argmax of the same field, sampled on the same grid.

    The filter takes a label map, so the field has to be thresholded first.
    That is not a limitation of the filter but of its input: once the logits
    are gone the interface position is no longer in the data.
    """
    import baseline_surfacenets as bl

    xs = torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res)
    gz, gy, gx = torch.meshgrid(xs, xs, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    with torch.no_grad():
        lab = fld.logits(pts).argmax(1).reshape(res, res, res)
    lab = lab.numpy().astype(np.int16) + 1  # 0 is the filter's background
    spacing = (2.0 / res,) * 3
    _, smooth, quads, _ = bl.surface_and_duplicates(lab, spacing)
    pairs = bl.boundary_labels(lab, spacing)
    return smooth, quads[(pairs == label + 1).any(axis=1)]


def _region_faces(surf, label: int = 0):
    """Vertices and the triangles bounding one region, as plain arrays."""
    tris, _ = region_triangles(surf, label)
    return surf.vertices.detach().cpu().numpy(), tris.cpu().numpy()


def figure_interface_accuracy() -> str:
    fld = enclosing_cell_field()
    res_show = 16
    resolutions = [8, 16, 32, 48]
    methods = [
        ("exact arrangement", "#4c72b0",
         lambda r: _region_faces(envelope3d.extract(fld, resolution=r, box=BOX, perturb=0.0))),
        ("corner-label case analysis", "#c44e52",
         lambda r: _region_faces(extract3d.extract(fld, resolution=r, box=BOX))),
        ("vtkSurfaceNets3D on the labels", "#dd8452",
         lambda r: _surfacenets_region(fld, r)),
    ]

    built, errors = {}, {}
    for name, _, build in methods:
        built[name] = {r: build(r) for r in resolutions}
        errors[name] = {r: true_interface_distance(fld, v[np.unique(f)]).numpy()
                        for r, (v, f) in built[name].items()}

    fig = plt.figure(figsize=(13.2, 9.6))
    gs_top = fig.add_gridspec(1, 3, top=0.885, bottom=0.45, left=0.02, right=0.98,
                              wspace=0.02)
    gs_bot = fig.add_gridspec(1, 3, top=0.345, bottom=0.075, left=0.055, right=0.975,
                              wspace=0.30)

    # Two tones rather than one ramp: a vertex is either on the interface to
    # machine precision or it is not, and the difference between 1e-17 and
    # 1e-2 is not something a continuous colour scale can show.
    h = 2.0 / res_show
    exact_grey = np.array(matplotlib.colors.to_rgb("#b8bec6"))
    # A ramp that is saturated at its low end too: the small errors have to be
    # as visible as the large ones, and a dark low end reads as a hole.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "offby", ["#ffd75e", "#f2971f", "#e02b1d", "#7d0018"])
    norm = matplotlib.colors.Normalize(0.0, 0.5)
    light = np.array([0.4, 0.5, 0.75])
    light = light / np.linalg.norm(light)
    common = np.concatenate([built[n][res_show][0][built[n][res_show][1]].reshape(-1, 3)
                             for n, _, _ in methods])

    for i, (name, _, _) in enumerate(methods):
        verts, faces = built[name][res_show]
        poly = verts[faces]
        face_err = true_interface_distance(fld, verts).numpy()[faces].mean(axis=1)
        colours = np.tile(exact_grey, (faces.shape[0], 1))
        hot = face_err > 1e-9
        colours[hot] = cmap(norm(face_err[hot] / h))[:, :3]
        n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        colours = np.clip(colours * (0.80 + 0.20 * np.abs(n @ light))[:, None], 0, 1)

        ax = fig.add_subplot(gs_top[0, i], projection="3d")
        ax.add_collection3d(Poly3DCollection(
            poly, facecolors=colours, edgecolors="#ffffff", linewidths=0.18, alpha=1.0))
        e = errors[name][res_show]
        _style(ax, name,
               f"worst vertex {e.max():.1e} = {e.max() / h:.2f} grid cells\n"
               f"{int((e > 1e-9).sum())} of {e.size} vertices off the interface",
               fit=common)

    swatch = fig.add_axes((0.145, 0.408, 0.020, 0.016))
    swatch.set_xticks([])
    swatch.set_yticks([])
    swatch.set_facecolor(exact_grey)
    fig.text(0.173, 0.416, "on the interface to machine precision", fontsize=8.5,
             va="center")
    cax = fig.add_axes((0.565, 0.408, 0.26, 0.016))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", extend="max")
    bar.set_label("off the interface, in grid cells", fontsize=8.5)
    bar.ax.tick_params(labelsize=8)

    # Worst and typical vertex, against refinement.
    ax = fig.add_subplot(gs_bot[0, :2])
    for name, colour, _ in methods:
        worst = [max(float(errors[name][r].max()), 1e-17) for r in resolutions]
        mid = [max(float(np.median(errors[name][r])), 1e-17) for r in resolutions]
        ax.plot(resolutions, worst, "o-", color=colour, lw=1.9, ms=5, label=f"{name}: worst")
        ax.plot(resolutions, mid, "^:", color=colour, lw=1.5, ms=5, alpha=0.75,
                label=f"{name}: median")
    ref = [0.2 * (8.0 / r) for r in resolutions]
    ax.plot(resolutions, ref, "--", color="#8c8c8c", lw=1.3, label="first order reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(resolutions)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.set_xlabel("grid resolution")
    ax.set_ylabel("distance from the true interface")
    ax.set_ylim(1e-18, 1e0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="center left", ncol=2, framealpha=0.95)
    ax.set_title("the worst vertex, and the typical one", fontsize=10)
    ax.annotate("the worst case-analysis vertex does not converge:\n"
                "0.37 grid cells at resolution 8, 1.93 at resolution 48",
                xy=(0.985, 0.76), xycoords="axes fraction", fontsize=8,
                ha="right", va="top", color="#444444")

    # The whole distribution, not just its ends.
    ax = fig.add_subplot(gs_bot[0, 2])
    bins = np.logspace(-18, 0, 55)
    for name, colour, _ in methods:
        e = np.maximum(errors[name][resolutions[-1]], 1e-18)
        ax.hist(e, bins=bins, histtype="step", lw=1.8, color=colour, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance from the true interface")
    ax.set_ylabel("vertices")
    ax.set_title(f"every vertex, grid {resolutions[-1]}", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(0.7, 1e5)
    ax.legend(fontsize=7, loc="upper center", framealpha=0.95)

    fig.suptitle(
        "Distance from the extracted surface to the interface it claims to represent, on a "
        "power-diagram cell whose geometry is known in closed form.\n"
        "All three read the same field on the same grid; the third is handed its argmax, "
        "because a label map is what that filter consumes.",
        fontsize=10.5, y=0.977)
    return _save(fig, "interface_accuracy.png")


# --------------------------------------------------------------------------
# The background grid and the ladder of solves it induces
# --------------------------------------------------------------------------
def _kuhn_tets(origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """The six tetrahedra of one cube, in the order `build_tetrahedral_grid` emits.

    Each is the monotone corner path 000 -> e_p0 -> e_p0 + e_p1 -> 111, so the
    six orderings of the three axis steps give the six tets and all of them
    share the 0->7 main diagonal.
    """
    from itertools import permutations

    def corner(bits: int) -> np.ndarray:
        return np.array(origin, dtype=float) + np.array(
            [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1], dtype=float)

    out = []
    for p0, p1, _ in permutations(range(3)):
        b1 = 1 << p0
        out.append(np.stack([corner(0), corner(b1), corner(b1 | (1 << p1)), corner(7)]))
    return np.stack(out)


TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _draw_tet(ax, verts, colour, alpha=0.5, edge="#303030", lw=0.9, shade=True):
    poly = np.stack([verts[list(f)] for f in TET_FACES])
    face = np.array(matplotlib.colors.to_rgb(colour))
    if shade:
        n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        light = np.array([0.4, 0.5, 0.75])
        lam = 0.62 + 0.38 * np.abs(n @ light / np.linalg.norm(light))
        face = np.clip(face[None] * lam[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(poly, facecolors=face, alpha=alpha,
                                         edgecolors=edge, linewidths=lw))


def _cube_wire(ax, origin=(0.0, 0.0, 0.0), colour="#606060", lw=1.0, ls="-"):
    o = np.array(origin, dtype=float)
    pts = np.array([[b & 1, (b >> 1) & 1, (b >> 2) & 1] for b in range(8)], dtype=float) + o
    segs = [(a, b) for a in range(8) for b in range(a + 1, 8)
            if bin(a ^ b).count("1") == 1]
    ax.add_collection3d(Line3DCollection([pts[list(s)] for s in segs], colors=colour,
                                         linewidths=lw, linestyles=ls))


def _barycentric(tet: np.ndarray, pts: np.ndarray) -> np.ndarray:
    m = np.stack([tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]], axis=1)
    lam = np.linalg.solve(m, (pts - tet[0]).T).T
    return np.concatenate([1.0 - lam.sum(1, keepdims=True), lam], axis=1)


def figure_tetrahedra() -> str:
    fig = plt.figure(figsize=(13.4, 5.3))
    grid = fig.add_gridspec(1, 3, wspace=0.02, left=0.01, right=0.99,
                            top=0.80, bottom=0.02)

    # (a) one cube, pulled apart into its six tetrahedra.
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    tets = _kuhn_tets()
    centre = np.full(3, 0.5)
    for i, tet in enumerate(tets):
        moved = tet + 1.1 * (tet.mean(0) - centre)
        _draw_tet(ax, moved, PALETTE[i], alpha=0.75)
        # verts 0 and 3 are corner(0) and corner(7): the diagonal every tet shares.
        ax.add_collection3d(Line3DCollection([moved[[0, 3]]], colors="#101010",
                                             linewidths=2.0))
    _cube_wire(ax, colour="#9a9a9a", lw=0.9, ls=":")
    _style_plain(ax, "one cube becomes six tetrahedra",
                 "the six orders of taking the three axis steps;\n"
                 "every one of them contains the main diagonal (black)",
                 lim=1.25, centre=centre,
                 # Looking almost straight down the main diagonal, so the six
                 # tets fan around it and can actually be counted.
                 view=(33.0, 42.0), zoom=1.45)

    # (b) why that is conforming: neighbours induce the same diagonal.
    ax = fig.add_subplot(grid[0, 1], projection="3d")
    shared = np.array([[1.0, 0, 0], [1.0, 1, 0], [1.0, 0, 1], [1.0, 1, 1]])
    for origin, colour in (((0.0, 0, 0), "#4c72b0"), ((1.0, 0, 0), "#dd8452")):
        _cube_wire(ax, origin, colour=colour, lw=1.3)
    for tri, colour in ((shared[[0, 1, 3]], "#55a868"), (shared[[0, 2, 3]], "#c44e52")):
        ax.add_collection3d(Poly3DCollection([tri], facecolors=colour, alpha=0.55,
                                             edgecolors="#202020", linewidths=1.0))
    ax.add_collection3d(Line3DCollection(
        [np.array([[1.0, 0, 0], [1.0, 1, 1]])], colors="#101010", linewidths=2.6))
    _style_plain(ax, "the grid is conforming by construction",
                 "the tets on each side of a shared face cut it\n"
                 "along the same diagonal, so there is nothing to weld",
                 lim=1.35, centre=np.array([1.0, 0.5, 0.5]))

    # (c) the ladder of solves, inside a single tetrahedron.
    ax = fig.add_subplot(grid[0, 2], projection="3d")
    tet = tets[0]
    seed = np.array([0.78, 0.46, 0.24])
    dirs = simplex_directions(3).numpy()
    fld = PowerDiagramField(torch.tensor(seed[None] + 0.62 * dirs))
    surf = envelope3d.extract(fld, resolution=1, box=(0.0, 1, 0, 1, 0, 1), perturb=1e-9)

    verts = surf.vertices.detach().numpy()
    tris = surf.triangles.detach().numpy()
    pairs = surf.triangle_labels.detach().numpy()
    inside = (_barycentric(tet, verts[tris].mean(1)) > -1e-9).all(axis=1)
    tris, pairs = tris[inside], pairs[inside]

    _draw_tet(ax, tet, "#d8d8d8", alpha=0.10, edge="#585858", lw=1.5, shade=False)
    poly = verts[tris]
    cols = np.array([matplotlib.colors.to_rgb(PALETTE[(p[0] + 2 * p[1]) % len(PALETTE)])
                     for p in pairs])
    n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    light = np.array([0.4, 0.5, 0.75])
    lam = 0.6 + 0.4 * np.abs(n @ light / np.linalg.norm(light))
    patches = Poly3DCollection(poly, facecolors=np.clip(cols * lam[:, None], 0, 1),
                               alpha=0.82, edgecolors="#ffffff", linewidths=0.25)
    # Pin the patches to the back of the depth sort so the marked corners,
    # which are the point of the panel, are not swallowed by them.
    patches.set_sort_zpos(-10.0)
    ax.add_collection3d(patches)

    used = np.unique(tris)
    kinds = surf.vertex_kind.detach().numpy()[used]
    for kind, colour, marker, size, name in (
        (extract3d.CROSSING, "#1f4e79", "o", 42, "2 labels tie on an edge  ($1\\times1$)"),
        (extract3d.TRIPLE, "#e8a33d", "s", 58, "3 labels tie on a face  ($2\\times2$)"),
        (extract3d.QUADRUPLE, "#e8000b", "*", 260, "4 labels tie inside  ($3\\times3$)"),
    ):
        p = verts[used[kinds == kind]]
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=size, c=colour, marker=marker,
                       edgecolors="#101010", linewidths=0.7, depthshade=False,
                       zorder=6, label=name)
    ax.legend(fontsize=8.2, loc="lower center", framealpha=0.94,
              bbox_to_anchor=(0.5, -0.02))
    _style_plain(ax, "inside one tetrahedron, every corner is a small solve",
                 "each logit is affine here, so a tie of $m$ labels\n"
                 "is $m-1$ linear equations and nothing else",
                 lim=0.8, centre=tet.mean(0))

    fig.suptitle("The background grid, and why the arrangement inside a cell is linear algebra",
                 fontsize=12.6, y=0.975)
    return _save(fig, "tetrahedra.png")


def _style_plain(ax, title: str, subtitle: str, lim: float, centre,
                 view=(20.0, -58.0), zoom=1.5) -> None:
    centre = np.asarray(centre, dtype=float)
    ax.set_xlim(centre[0] - lim, centre[0] + lim)
    ax.set_ylim(centre[1] - lim, centre[1] + lim)
    ax.set_zlim(centre[2] - lim, centre[2] + lim)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_axis_off()
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.6, pad=-8)
    ax.view_init(elev=view[0], azim=view[1])


def _save(fig, name: str) -> str:
    path = os.path.join(FIGDIR, name)
    # 220 keeps every figure above 300 dpi at the widths the paper places them
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"  {name}")
    return path


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    print("writing figures to", FIGDIR)
    for fn in (figure_tetrahedra, figure_arrangement, figure_exploded, figure_failure,
               figure_multi_feature, figure_plateau, figure_convergence,
               figure_interface_accuracy):
        fn()


if __name__ == "__main__":
    main()
