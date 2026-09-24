# FlexiCubes

The differentiable extractor the paper compares against at `K=2`, from
Shen et al., *Flexible Isosurface Extraction for Gradient-Based Mesh
Optimization*, SIGGRAPH 2023. Run `fetch.ps1` to check it out at the pinned
commit `4cc7d6c`; `src/compare_flexicubes.py` expects it there.

Nothing is vendored. The upstream is Apache 2.0, so it could be, but the
comparison is easier to trust if the code it runs against is fetched from
source at a stated commit.

## What it needs

Less than its examples suggest. `flexicubes.py` imports only `torch` and its
own `tables.py`, so the extractor runs without nvdiffrast or kaolin — those are
dependencies of their rendering examples, which the comparison does not use.

## Two things to know before reading the numbers

**It runs in float32.** Its working buffers are allocated at the ambient
default dtype while its default weight tensors are `float`, so the two only
agree under `torch.float32`. Everything else in this repository runs in
float64. The comparison therefore runs each method in the precision it was
designed for and says so, rather than dragging ours down to float32 — which
would understate it, for the reasons in the `patches_with_too_few_corners`
note in `src/envelope3d.py`.

**The weights are left at their defaults.** FlexiCubes carries per-cube
`alpha`, `beta` and `gamma` weights and supports deforming the grid, and
optimising them is the substance of their contribution. For the fixed-field
accuracy test there is nothing to optimise them against, so they stay at their
defaults, which is also how you would use the method to extract a known field.
In the optimisation arm they are free parameters and are optimised, because
holding them fixed there would handicap the method rather than test it.
