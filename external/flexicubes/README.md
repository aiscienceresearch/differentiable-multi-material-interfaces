# FlexiCubes

The differentiable extractor the paper compares against at `K=2`, from
Shen et al., *Flexible Isosurface Extraction for Gradient-Based Mesh
Optimization*, SIGGRAPH 2023. Run `fetch.ps1` to check it out at the pinned
commit `4cc7d6c`; `src/compare_flexicubes.py` expects it there.

Nothing is vendored. The upstream is Apache 2.0, so it could be, but the
comparison is easier to trust if the code it runs against is fetched from
source at a stated commit.

## Dependencies

`flexicubes.py` imports only `torch` and its
own `tables.py`, so the extractor runs without nvdiffrast or kaolin, which are
dependencies of their rendering examples that the comparison does not use.

## Precision and weights

**It runs in float32.** Its working buffers are allocated at the ambient
default dtype while its default weight tensors are `float`, so the two only
agree under `torch.float32`. Everything else in this repository runs in
float64. The comparison therefore runs each method in the precision it was
designed for and says so. Running ours in float32 would understate it, for
the reasons in the `patches_with_too_few_corners` note in
`src/envelope3d.py`.

**The weights are left at their defaults.** FlexiCubes carries per-cube
`alpha`, `beta` and `gamma` weights and supports deforming the grid, and
optimizing them is the substance of their contribution. For the fixed-field
accuracy test there is nothing to optimize them against, so they stay at their
defaults, which is also how the method would be used to extract a known field.
In the optimization arm they are free parameters and are optimized, because
holding them fixed there would handicap the method rather than test it.
