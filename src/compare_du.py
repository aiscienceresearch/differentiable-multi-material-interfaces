"""Head-to-head against the reference implementation of Du et al.

Their method and ours solve the same combinatorial problem -- the argmax
partition of a piecewise-linear multi-material field over a tetrahedral mesh --
by opposite routes. They use exact predicates and a precomputed lookup table
over the simplicial arrangement, which makes the combinatorics provably correct
but opaque to differentiation. We solve each vertex as a small linear system in
the node logits, which is what makes the result differentiable.

If both are correct, they must agree exactly, and that agreement is the point of
this script: it is an independent check on our extractor by a published
implementation whose correctness is established separately from ours.

Making the comparison mean something requires removing every difference that is
not the algorithm. Two matter.

The mesh.
    Their tool normally generates its own grid, and its tetrahedralisation is
    not ours. A disagreement would then say nothing, so we export our own
    tetrahedra through their ``tetMeshFile`` input and both sides read the same
    cells.

The field.
    Their loader evaluates analytic primitives at the tet vertices and has no
    input path for arbitrary per-vertex values. Two kinds of case follow from
    that. The analytic ones are framed in their language -- the spheres and
    planes their tool accepts -- with `ReferenceSpecField` mirroring those
    formulas so our extractor sees bit-identical node logits. The rest are the
    fields the paper actually cares about, which have no closed form, and reach
    their tool through a 29-line addition to its loader (`sampled-values.patch`)
    that reads node values from a file. The patch touches only input: it fills
    the same matrix column every other branch fills, and no line of the
    arrangement code is changed. `analytic vs sampled` below is the control that
    checks this, running one field down both routes and requiring the same
    answer.

The symbolic perturbation is switched off here. It exists to break exact
coincidences, and introducing a displacement they do not apply would guarantee a
mismatch at precisely the configurations worth comparing. All cases below are
generic, so it is not needed.

Run with `python compare_du.py`. Requires the Docker image built from
`external/du2022/Dockerfile`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import torch
from torch import Tensor

import envelope3d
from extract3d import build_tetrahedral_grid
from fields import MultiLabelField, NeuralMultiLabelField, PowerDiagramField

torch.set_default_dtype(torch.float64)

BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
SITES = torch.tensor([
    [0.35, 0.20, 0.10], [-0.30, 0.25, -0.15], [0.05, -0.40, 0.22],
    [-0.15, -0.10, 0.45], [0.10, 0.05, -0.48],
])
IMAGE = "du2022:latest"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "external", "du2022", "example")


class ReferenceSpecField(MultiLabelField):
    """The primitives their loader understands, evaluated exactly as it does.

    Signs and normalisations are theirs, taken from `primitives.h`: a sphere is
    ``radius - ||x - centre||``, so the inside is the larger value, and a plane
    is ``n.(x - p)`` with ``n`` normalised at construction. The normalisation
    matters and cannot be folded away, because scaling one logit changes which
    label wins against the others.
    """

    dim = 3

    def __init__(self, spec: list[dict]):
        super().__init__()
        self.spec = spec
        self.num_classes = len(spec)

    def logits(self, x: Tensor) -> Tensor:
        columns = []
        for entry in self.spec:
            kind = entry["type"]
            if kind == "sphere":
                centre = torch.as_tensor(entry["center"], dtype=x.dtype, device=x.device)
                columns.append(entry["radius"] - (x - centre).norm(dim=-1))
            elif kind == "plane":
                point = torch.as_tensor(entry["point"], dtype=x.dtype, device=x.device)
                normal = torch.as_tensor(entry["normal"], dtype=x.dtype, device=x.device)
                columns.append((x - point) @ (normal / normal.norm()))
            elif kind == "zero":
                columns.append(x.new_zeros(x.shape[0]))
            else:
                raise ValueError(f"no matched evaluation for {kind!r}")
        return torch.stack(columns, dim=-1)


# --------------------------------------------------------------------------
# Driving their binary
# --------------------------------------------------------------------------
def run_reference(spec: list[dict] | None, nodes: Tensor, tets: Tensor,
                  workdir: str, values: Tensor | None = None) -> tuple[dict, dict, Tensor]:
    """Run their material_interface on our tetrahedra and read back the result.

    With `values` given, each label is passed as a column of node values through
    the patched loader instead of as an analytic primitive.
    """
    if values is not None:
        spec = []
        for k in range(values.shape[1]):
            with open(os.path.join(workdir, f"values_{k}.json"), "w") as fh:
                json.dump(values[:, k].tolist(), fh)
            spec.append({"type": "sampled", "file": f"values_{k}.json"})
    with open(os.path.join(workdir, "func.json"), "w") as fh:
        json.dump(spec, fh)
    with open(os.path.join(workdir, "tets.json"), "w") as fh:
        json.dump([nodes.tolist(), tets.tolist()], fh)
    with open(os.path.join(workdir, "config.json"), "w") as fh:
        json.dump({
            "tetMeshFile": "tets.json",
            "funcFile": "func.json",
            "outputDir": "./",
            "useLookup": True,
            "useSecondaryLookup": True,
            "useTopoRayShooting": True,
        }, fh)

    started = time.perf_counter()
    done = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{workdir}:/work", IMAGE, "/work/config.json"],
        capture_output=True, text=True,
    )
    wall = time.perf_counter() - started
    if done.returncode != 0:
        raise RuntimeError(f"reference implementation failed:\n{done.stdout}\n{done.stderr}")

    with open(os.path.join(workdir, "stats.json")) as fh:
        stats = json.load(fh)
    with open(os.path.join(workdir, "mesh.json")) as fh:
        mesh = json.load(fh)
    with open(os.path.join(workdir, "timings.json")) as fh:
        timings = json.load(fh)

    theirs = {
        "vertices": stats["num_MI_verts"],
        "polygons": stats["num_MI_faces"],
        "patches": stats["num_patches"],
        # Their `num_cells` counts connected regions, of which one label can own
        # several. Comparing that to a label count would be comparing two
        # different things, so the distinct labels among their cells are what
        # line up with ours.
        "labels": len(set(mesh["cells_label"][0])),
        "regions": stats["num_cells"],
        "label_pairs": {tuple(sorted(p)) for p in mesh["patches_label"][0]},
        "wall_seconds": wall,
        "solve_seconds": _reference_solve_time(timings),
    }
    return theirs, mesh, torch.tensor(mesh["points"], dtype=torch.get_default_dtype())


def _reference_solve_time(timings: dict) -> float:
    """Their own timing of the arrangement, excluding I/O and table loading.

    Reported separately from wall time because wall time here is dominated by
    container startup and by parsing a tet mesh they would normally generate
    in-process, neither of which is their algorithm.
    """
    keys = ("MI(2 func)", "MI(3 func)", "MI(>=4 func)", "extract mesh",
            "compute xyz", "edge-face connectivity", "patches", "chains",
            "order patches around chains", "shells and components",
            "material cells", "highest func", "filter")
    return sum(float(timings[k]) for k in keys if k in timings)


# --------------------------------------------------------------------------
# The same quantities, measured on our output
# --------------------------------------------------------------------------
class _Union:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _patch_components(surf) -> int:
    """Connected components of same-pair triangles, which is their patch notion.

    Our triangles come from fanning each cell's convex polygon, so a fan diagonal
    joins triangles within one polygon and a polygon boundary edge joins the two
    cells sharing it. Keying on the label pair as well as the edge stops the walk
    at triple curves, where three patches share an edge but no two share a pair,
    which is exactly where their patches end too.
    """
    tris = surf.triangles.tolist()
    pairs = surf.triangle_labels.tolist()
    seen: dict[tuple, int] = {}
    uf = _Union(len(tris))
    for t, (tri, pair) in enumerate(zip(tris, pairs)):
        p = tuple(sorted(pair))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (p, min(tri[a], tri[b]), max(tri[a], tri[b]))
            if key in seen:
                uf.union(seen[key], t)
            else:
                seen[key] = t
    return len({uf.find(t) for t in range(len(tris))})


def our_stats(field: MultiLabelField, resolution: int, box=BOX,
              perturb: float = 0.0) -> tuple[dict, Tensor]:
    started = time.perf_counter()
    surf = envelope3d.extract(field, resolution=resolution, box=box, perturb=perturb)
    wall = time.perf_counter() - started
    d = surf.diagnostics
    return {
        "vertices": d["num_crossings"] + d["num_triple_points"] + d["num_quadruple_points"],
        # `num_patches` counts one convex polygon per cell and label pair, which
        # is what they call a face; their "patch" is a connected union of those.
        "polygons": d["num_patches"],
        "patches": _patch_components(surf),
        "labels": len(d["labels_present"]),
        "label_pairs": {tuple(sorted(p)) for p in surf.triangle_labels.tolist()},
        "wall_seconds": wall,
    }, surf.vertices.detach()


def hausdorff(a: Tensor, b: Tensor, chunk: int = 512) -> float:
    """Symmetric nearest-neighbour distance between two vertex sets.

    Chunked because both sets run to tens of thousands of points and the full
    matrix does not fit comfortably. The compute mode is not the default: cdist
    otherwise expands the distance as ||a||^2 + ||b||^2 - 2a.b, which cancels
    catastrophically once the two points nearly coincide and cannot resolve
    anything below sqrt(eps) ~ 1e-8. Two extractors that agree to machine
    precision are exactly the case it cannot measure.
    """
    def one_way(p: Tensor, q: Tensor) -> float:
        worst = 0.0
        for i in range(0, p.shape[0], chunk):
            d = torch.cdist(p[i:i + chunk], q,
                            compute_mode="donot_use_mm_for_euclid_dist")
            worst = max(worst, float(d.min(dim=1).values.max()))
        return worst

    if a.numel() == 0 or b.numel() == 0:
        return math.inf
    return max(one_way(a, b), one_way(b, a))


# --------------------------------------------------------------------------
# Cases, stated in their language
# --------------------------------------------------------------------------
def spheres_from_their_example() -> list[dict]:
    """Their own published 18-sphere configuration, used unmodified."""
    with open(os.path.join(EXAMPLE_DIR, "18-sphere.json")) as fh:
        return json.load(fh)


def random_spheres(count: int, seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    centres = (torch.rand((count, 3), generator=g) - 0.5) * 1.2
    radii = 0.25 + 0.45 * torch.rand(count, generator=g)
    return [{"type": "sphere", "center": c.tolist(), "radius": float(r)}
            for c, r in zip(centres, radii)]


def random_planes(count: int, seed: int) -> list[dict]:
    """Planes give an arrangement with flat interfaces and straight triple curves.

    Worth separating from the spheres because here the linear interpolant equals
    the field exactly, so any disagreement is purely combinatorial and cannot be
    blamed on how either side samples a curved surface.
    """
    g = torch.Generator().manual_seed(seed)
    points = (torch.rand((count, 3), generator=g) - 0.5) * 1.0
    normals = torch.randn((count, 3), generator=g)
    return [{"type": "plane", "point": p.tolist(), "normal": n.tolist()}
            for p, n in zip(points, normals)]


@dataclass
class Case:
    """One matched run. `spec` is None when the field reaches them as node values."""
    name: str
    field: MultiLabelField
    resolution: int
    spec: list[dict] | None = None
    box: tuple = BOX


class FixedValues(MultiLabelField):
    """A field that reports stored node values, so both sides read the same array.

    Needed for the degeneracy control below: a field can only be made generic
    once, outside both extractors, if the modified values are what our extractor
    reads as well.
    """

    dim = 3

    def __init__(self, nodes: Tensor, values: Tensor):
        super().__init__()
        self.register_buffer("_nodes", nodes)
        self.register_buffer("_values", values)
        self.num_classes = values.shape[1]

    def logits(self, x: Tensor) -> Tensor:
        if x.shape == self._nodes.shape and torch.equal(x, self._nodes):
            return self._values
        raise ValueError("this field is defined only at the nodes it was built from")


def count_exact_ties(values: Tensor) -> int:
    """Nodes where the argmax is not unique, which is where the two can differ."""
    top2 = values.topk(2, dim=1).values
    return int((top2[:, 0] == top2[:, 1]).sum())


def make_generic(values: Tensor, amount: float = 1e-7, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(values.shape, generator=g, dtype=values.dtype) - 0.5
    return values + amount * noise


def build_cases() -> list[Case]:
    eighteen = spheres_from_their_example()
    cases = [
        Case("their 18-sphere example", ReferenceSpecField(eighteen), 21, eighteen),
        Case("8 spheres", ReferenceSpecField(random_spheres(8, 0)), 24,
             random_spheres(8, 0)),
        Case("5 planes", ReferenceSpecField(random_planes(5, 1)), 24,
             random_planes(5, 1)),
        Case("12 planes", ReferenceSpecField(random_planes(12, 2)), 20,
             random_planes(12, 2)),
        # Control on the loader patch: the same field they evaluate analytically
        # above, handed to them instead as node values. Anything but an exact
        # match means the patch is not a faithful input path.
        Case("analytic vs sampled (control)", ReferenceSpecField(random_planes(5, 1)), 24),
        # Fields with no closed form, which is what the paper is actually about.
        Case("neural field, K=6", NeuralMultiLabelField(num_classes=6, hidden=48,
                                                        dim=3, seed=2), 24),
        Case("power diagram, K=5", PowerDiagramField(SITES), 24),
    ]
    liver = _liver_case()
    if liver is not None:
        cases.append(liver)
    return cases


def _liver_case() -> Case | None:
    """The clinical segmentation field, if the volume is available locally."""
    try:
        import realdata
        from realdata_experiment import ORIGIN_VOX, RESOLUTION, WIDTH
    except Exception:
        return None
    if not realdata.PROBS.exists():
        return None
    field, box, _block, _spacing = realdata.load_block(ORIGIN_VOX, WIDTH)
    return Case("clinical liver segmentation", field, RESOLUTION, box=box)


def _report(case: Case, nodes: Tensor, tets: Tensor, values: Tensor | None,
            label: str | None = None, scored: bool = True,
            field: MultiLabelField | None = None) -> int:
    """Run both extractors on one input, print the table, return 1 if it fails."""
    spec = case.spec if values is None else None
    workdir = tempfile.mkdtemp(prefix="du_compare_")
    try:
        theirs, _mesh, their_points = run_reference(
            spec, nodes, tets, workdir, values=values)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    ours, our_points = our_stats(field or case.field, case.resolution, case.box)

    indent = "  " if label is None else "    "
    if label is not None:
        print(f"  {label}:")
    print(f"{indent}{'quantity':<20}{'theirs':>10}{'ours':>10}   agree")
    agree = True
    for key in ("vertices", "polygons", "patches", "labels"):
        same = theirs[key] == ours[key]
        agree &= same
        print(f"{indent}{key:<20}{theirs[key]:>10}{ours[key]:>10}"
              f"   {'yes' if same else 'NO'}")

    same_pairs = theirs["label_pairs"] == ours["label_pairs"]
    agree &= same_pairs
    print(f"{indent}{'interface pairs':<20}{len(theirs['label_pairs']):>10}"
          f"{len(ours['label_pairs']):>10}   {'yes' if same_pairs else 'NO'}")

    gap = hausdorff(their_points, our_points)
    agree &= gap < 1e-12
    print(f"{indent}vertex sets coincide to {gap:.3e}")
    print(f"{indent}their arrangement {theirs['solve_seconds'] * 1e3:.1f} ms"
          f"  |  ours {ours['wall_seconds'] * 1e3:.0f} ms (Python)")
    if not scored:
        print(f"{indent}=> differs, as a degenerate input permits")
        return 0
    print(f"{indent}=> {'AGREE' if agree else 'DISAGREE'}")
    return 0 if agree else 1


def main() -> int:
    print("=" * 78)
    print("Our extractor vs the reference implementation of Du et al., same tetrahedra")
    print("=" * 78)

    cases = build_cases()
    failures = 0
    for case in cases:
        nodes, tets = build_tetrahedral_grid(case.resolution, case.box)
        with torch.no_grad():
            values = None if case.spec is not None else case.field.logits(nodes).detach()
        num_classes = len(case.spec) if case.spec is not None else values.shape[1]

        route = "analytic" if case.spec is not None else "node values"
        print(f"\n{case.name}  (K={num_classes}, {nodes.shape[0]} nodes, "
              f"{tets.shape[0]} tets, via {route})")

        ties = 0 if values is None else count_exact_ties(values)
        if ties:
            # At an exact tie the arrangement is not unique, so the two are
            # under no obligation to match and the raw run is reported without
            # being scored. What is fair to require is that the disagreement is
            # confined to those coincidences, which the generic run below tests
            # by removing them from the input both extractors read.
            print(f"  {ties} nodes carry an exact argmax tie; reporting both the"
                  f" field as given\n  and a generic version of it")
            _report(case, nodes, tets, values, label="as given", scored=False)
            generic = make_generic(values)
            failures += _report(case, nodes, tets, generic, label="made generic",
                                field=FixedValues(nodes, generic))
        else:
            failures += _report(case, nodes, tets, values)

    print("\n" + "=" * 78)
    print(f"{len(cases) - failures}/{len(cases)} cases agree")
    print("=" * 78)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
