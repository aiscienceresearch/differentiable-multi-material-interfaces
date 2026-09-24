# Exported meshes

Regenerate with `python src/export_mesh.py` (the crop) and
`python -c "import export_mesh; export_mesh.export_organ(128)"` (the organ).
Both need `data/probs_liver.npz`, which `tools/fetch_realdata.ps1` produces.

## Opening these in Blender

Units are millimetres, and the meshes are centred on their own bounding box so
they land at the world origin. The scanner-frame offset is in each file's
header comment if you need the original coordinates back.

The whole liver is still about 210 units across, and Blender treats one unit as
a metre, so on import it sits mostly outside the default view. Press **Home**
in the viewport to frame everything. To make the readouts say millimetres
rather than metres, set Scene Properties → Units → Unit System to Metric and
Length to Millimeters.

## `liver/` — the 49³ crop used in the paper

The region the paper measures. Every region in it is cut by the crop, so the
per-region surfaces are open.

## `liver_whole/` — the whole organ at resolution 128

All eight Couinaud segments plus background, 561k triangles, 25 patches, 5533
triple-curve edges. Segments 3, 5 and 6 are closed watertight solids; the rest
are open only where the liver runs off the top of the scan.

## What is in each file

`interfaces.obj` is the complex as extracted. One shared vertex block, one
object and material per label pair (`patch_segment5_segment6` and so on), and
the triple curves as OBJ line elements in their own object. Two things are
worth trying on it in Blender:

- Select all and **Merge by Distance**. It removes nothing. The patches already
  share vertex indices rather than merely coinciding, which is the conformity
  guarantee the paper proves, and no welding step was involved in producing it.
- Look at the `triple_curves` object with **edge select**. Those 5533 edges are
  where three segments meet. They are output, not something recovered
  afterwards by intersecting surfaces.

The mesh is deliberately non-manifold along those curves: three faces share
each of those edges. Blender will report this under Mesh Statistics, and it is
correct rather than a defect. Tools that insist on a two-manifold input want
the per-region files instead.

`regions/region_*.obj` is one outward-oriented surface per region, suitable for
boolean operations, printing, or simulation setup. Interfaces appear twice
here, once in each of the two regions that share them, which is the price of
each file being a standalone solid.
