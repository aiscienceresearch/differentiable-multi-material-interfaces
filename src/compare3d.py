"""Corner-label case analysis vs the exact arrangement, against exact ground truth."""
import torch

import envelope3d
import extract3d
import voronoi_exact
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

exact = voronoi_exact.cell_geometry(fld.sites.detach(), fld.weights.detach(), 0)
ev = float(exact["volume"])
ea = sum(float(a) for a in exact["facet_area"].values())
print(f"exact cell 0: volume {ev:.12f}  area {ea:.12f}  "
      f"{len(exact['facet_area'])} facets, {exact['vertices'].shape[0]} corners\n")

for name, module in (("corner-label case analysis", extract3d),
                     ("exact arrangement", envelope3d)):
    print(name)
    for res in (8, 16, 32, 48):
        surf = module.extract(fld, resolution=res)
        vol = float(surf.enclosed_volume(0).detach())
        involved = (surf.triangle_labels == 0).any(dim=1)
        area = float(surf.triangle_areas()[involved].sum().detach())
        t = extract3d.check_topology(surf)
        print(f"  res {res:3d}  volume err {abs(vol - ev) / ev:.3e}  "
              f"area err {abs(area - ea) / ea:.3e}  "
              f"unpaired {t['unpaired_sides']:4d}  "
              f"non-orientable {t['inconsistently_oriented_sides']:5d}  "
              f"quadruple pts {surf.diagnostics['num_quadruple_points']}")
    print()
