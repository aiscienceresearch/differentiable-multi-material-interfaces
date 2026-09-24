"""Where do the unpaired sides come from?"""
import torch

import envelope3d
import extract3d
from fields import PowerDiagramField

torch.set_default_dtype(torch.float64)

torch.manual_seed(3)
shell = torch.tensor([
    [1.0, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    [1.4, 0, 0], [-1.4, 0, 0], [0, 1.4, 0], [0, -1.4, 0], [0, 0, 1.4], [0, 0, -1.4],
]) * 0.55
shell = shell + 0.06 * torch.randn_like(shell)
fld = PowerDiagramField(torch.cat([torch.tensor([[0.02, -0.01, 0.03]]), shell]))

surf = envelope3d.extract(fld, resolution=16)
nv = surf.vertices.shape[0]
K = int(surf.node_labels.max()) + 1

# Patch sizes: a polygon needs at least three corners.
group_key = surf.diagnostics
tris = surf.triangles
u = tris[:, [0, 1, 2]].reshape(-1)
v = tris[:, [1, 2, 0]].reshape(-1)
pair = surf.triangle_labels.repeat_interleave(3, dim=0)
key = extract3d._edge_pair_keys(u, v, pair, nv, K)
uniq, inverse = torch.unique(key, return_inverse=True)
count = torch.bincount(inverse, minlength=uniq.numel())

curve = surf.triple_segments
cp = surf.triple_labels[:, torch.tensor(((0, 1), (0, 2), (1, 2)))].reshape(-1, 2)
curve_key = extract3d._edge_pair_keys(
    curve[:, 0].repeat_interleave(3), curve[:, 1].repeat_interleave(3), cp, nv, K)
bnd_key = extract3d._edge_pair_keys(
    surf.boundary_segments[:, 0], surf.boundary_segments[:, 1],
    surf.boundary_segment_labels, nv, K)
allowed = torch.cat([curve_key, bnd_key])
bad = ((count == 1) & ~torch.isin(uniq, allowed)).nonzero(as_tuple=True)[0]
print(f"{bad.numel()} unpaired sides of {uniq.numel()}")

side_tri = torch.arange(len(tris)).repeat_interleave(3)
rows = torch.isin(inverse, bad)
kinds = surf.vertex_kind[torch.stack([u[rows], v[rows]], -1)]
print("  endpoint kinds (0=cross,1=triple,2=quad):",
      torch.bincount(kinds.reshape(-1), minlength=3).tolist())
onb = surf.vertex_on_domain_boundary[torch.stack([u[rows], v[rows]], -1)].all(1)
print(f"  both endpoints on the domain boundary: {int(onb.sum())} of {int(rows.sum())}")

# Are these sides in fact triple-curve segments that got dropped?
both_tp = (kinds > 0).all(dim=1)
print(f"  sides joining two triple/quadruple points: {int(both_tp.sum())}")
d = surf.diagnostics
print(f"  triple points {d['num_triple_points']}, quadruple {d['num_quadruple_points']}, "
      f"triple segments recorded {d['num_triple_segments']}")
