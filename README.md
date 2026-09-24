# Exact Differentiable Extraction of Multi-Material Interfaces

Code for the paper of the same name. Given a field of $K$ logits on a
tetrahedral grid, this extracts the exact argmax partition of the field's
piecewise-linear interpolant — triple curves, quadruple points and all — in a
form that reverse-mode autograd can differentiate.

The point is the combination. The exact partition is already computable, by
[Du et al. 2022](https://doi.org/10.1145/3528223.3530176), but their
construction encodes vertices implicitly and resolves them with exact
predicates, so no derivative survives it. Here each vertex is instead the
solution of a $1\times1$, $2\times2$ or $3\times3$ linear system in the node
logits, which makes its position an explicit differentiable function of the
field. The trade is exact predicates for a symbolic perturbation, and
`src/compare_du.py` measures what that costs by running both implementations
on identical tetrahedra.

## Layout

| Path | What it is |
| --- | --- |
| `src/envelope3d.py` | The extractor. The argmax arrangement of the P1 interpolant, in 3D. |
| `src/extract3d.py` | The corner-label case analysis, kept as the baseline the paper argues against. |
| `src/extract2d.py` | The 2D instantiation. |
| `src/fields.py` | Test fields: power diagrams, spherical shells, sector fans, neural fields. |
| `src/validate3d.py`, `src/validate.py` | The automated checks cited in the paper. |
| `src/scaling.py` | Wall time and peak memory against resolution and against K. |
| `src/compare_du.py` | Head-to-head against the reference implementation of Du et al. |
| `src/compare_flexicubes.py` | Head-to-head against FlexiCubes at K=2, and one-vs-rest at K=3. |
| `src/double_bubble.py`, `src/contact.py` | The two optimisation studies. |
| `src/ablation_double_bubble.py` | Four-arm ablation isolating the combinatorics from the solver, plus an initialisation sweep. |
| `src/realdata_experiment.py` | Extraction from a segmentation network's probability volume. |
| `src/check_claims.py` | Re-checks the numbers quoted in the paper against `data/`, and its citations against its bibliography. |
| `external/du2022/` | Container recipe and loader patch for building Du et al.'s code. |
| `external/flexicubes/` | Fetch script for FlexiCubes, pinned to the commit compared against. |
| `paper/` | LaTeX source for the manuscript and its supplement. |
| `paper/make_submission.py` | Bundles the CGF source archive: main.tex with figures flattened, the Eurographics class and the bibliography. |

## Reproducing

Everything runs on CPU in float64.

```bash
pip install torch numpy matplotlib
python src/validate3d.py      # the 3D checks
python src/validate.py        # the 2D checks
```

Both print a `[PASS]`/`[FAIL]` line per check and exit non-zero on any failure.
`data/validate3d_run.txt` and `data/validate_run.txt` are transcripts of a
clean run, so the count of checks quoted in the paper can be verified without
running anything. `data/degeneracy_run.txt` is the same for the perturbation
cost measurement.

```bash
python src/check_claims.py    # the paper's numbers against data/
```

This reads the committed JSON rather than re-running anything, so it is
instant, and it exits non-zero if a number in the manuscript has drifted from
the result it was taken from. It also checks that every citation resolves and
every bibliography entry is used, which needs `paper/*.aux`, so build the
paper first if you want that half.

### The clinical experiment

The probability volume is not redistributed here. `tools/fetch_realdata.ps1`
retrieves it from its original source. This study also compares against
`vtkSurfaceNets3D`, so it needs VTK:

```bash
pip install vtk
python src/realdata_experiment.py
```

### The comparison against Du et al.

This one needs Docker, because it builds and runs their C++ implementation.

```bash
docker build -t du2022:latest external/du2022
python src/compare_du.py
```

`data/du_comparison_run.txt` is the transcript of one such run, so the table in
the supplement can be checked without building the container. Counts and vertex
gaps are deterministic; the timings are wall-clock and move between runs.

### The comparison against FlexiCubes

FlexiCubes is fetched rather than vendored. Its extractor imports only `torch`,
so none of the rendering dependencies in its examples are needed.

```bash
pwsh external/flexicubes/fetch.ps1
python src/compare_flexicubes.py
```

Three arms: extraction accuracy at `K=2`, an optimisation driving one parameter
tensor through both extractors, and a `K=3` junction where FlexiCubes has to be
run one-vs-rest. The third arm also runs our extractor through the same
one-vs-rest decomposition, as a control separating the decomposition from
FlexiCubes' vertex placement. `data/flexicubes_comparison_run.txt` is the
transcript. Note
that FlexiCubes runs in float32 and everything else here runs in float64; the
reason, and why we did not equalise it, is in `external/flexicubes/README.md`.

Two arms reproduce exactly and one does not. Extraction and the junction sweep
are deterministic for both methods; the FlexiCubes optimisation is not, because
its scatter assembly accumulates in float32 in an order the GPU does not fix.
Expect its rows in the optimisation table to move in the second decimal on a
re-run. Ours are deterministic throughout.

### The double-bubble ablation

`extract3d` differs from `envelope3d` in four ways at once, so substituting it
cannot say which difference costs the optimisation anything. This runs two
intermediate arms that change the combinatorics alone --- one keeping our
perturbation, one dropping it --- and then re-runs all four from five different
initialisations, to test whether the wrong answer is a point or a basin.

```bash
python src/ablation_double_bubble.py
```

It takes upwards of an hour. Results are written to
`data/ablation_double_bubble.json` after every arm and the script skips arms
already present there, so an interrupted run can be resumed by invoking it
again. `data/ablation_double_bubble_run.txt` is the transcript. Both are
deterministic.

`external/du2022/Dockerfile` builds their published code unmodified except for
two things, both of which are input-side and neither of which touches their
arrangement algorithm:

- Compiler flags that force-include headers libstdc++ no longer supplies
  transitively, and relax warnings-as-errors for a deprecated call in a pinned
  dependency. Their code is from 2022 and does not otherwise build on a current
  toolchain.
- `sampled-values.patch`, twenty-nine lines adding a branch to their function
  loader so it can read per-vertex values. Their loader only evaluates analytic
  primitives, so without this there is no way to hand them a field with no
  closed form. The branch fills the same matrix column every existing branch
  fills. `compare_du.py` includes a control case that sends one field down both
  the analytic and the sampled path and requires the same answer.

The harness passes our tetrahedra to their tool through its `tetMeshFile`
input, so both read the same cells and any disagreement is attributable to the
algorithm rather than the mesh.

## Licensing

Code is MIT (`LICENSE`). The manuscript and figures under `paper/` and
`figures/` are CC BY 4.0, matching the open-access licence CGF publishes under.

`external/du2022/sampled-values.patch` is a modification of
[implicit_functions](https://github.com/duxingyi-charles/implicit_functions)
and remains under that project's license; it is distributed here as a patch
rather than as modified sources for exactly that reason.
