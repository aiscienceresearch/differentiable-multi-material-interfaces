"""Check the manuscript's numbers and citations against their sources.

Every figure quoted in the paper was typed in by hand, so every one of them is
a chance to have left a stale number behind after a re-run. This re-reads the
JSON the experiments wrote and the .aux files LaTeX wrote, and reports any
claim whose two halves have drifted apart.

Run it from anywhere:

    python src/check_claims.py

It exits non-zero if anything fails, so it can go in a pre-submission hook.
The .aux files must exist, which means the paper must have been built at
least once; the JSON is committed, so the numeric half needs nothing.
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DATA = ROOT / "data"

main = (PAPER / "main.tex").read_text(encoding="utf-8", errors="replace")
supp = (PAPER / "supplementary.tex").read_text(encoding="utf-8", errors="replace")

failures = []


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

abl = json.loads((DATA / "ablation_double_bubble.json").read_text())
fcx = json.loads((DATA / "flexicubes_comparison.json").read_text())
real = json.loads((DATA / "realdata_results.json").read_text())

_checked = 0


def claim(label, literal, expected, actual, tol=0.0):
    """Assert that `literal` appears in the manuscript and matches `actual`.

    `expected` is what the manuscript says, restated here so a reader can see
    both halves at once; `literal` is the exact string to find, which is not
    always derivable from `expected` because of LaTeX spacing and units.
    """
    global _checked
    _checked += 1
    if literal not in main and literal not in supp:
        failures.append("NOT IN TEXT  %-38s looked for %r" % (label, literal))
    elif abs(expected - actual) > tol:
        failures.append("STALE        %-38s text %g, data %g" % (label, expected, actual))


runs = abl["runs"]
claim("ablation exact A/V", "$9.1481$", 9.1481, runs["exact arrangement"]["final_ratio"], 5e-5)
claim("ablation exact r/d", "$1.0064$", 1.0064, runs["exact arrangement"]["r_over_d"][0], 5e-5)
claim("ablation closed-form r/d", "$0.8657$", 0.8657, runs["corner-label, closed form"]["r_over_d"][0], 5e-5)
claim("ablation nopert r/d", "$0.9423$", 0.9423, runs["corner-label, no perturbation"]["r_over_d"][0], 5e-5)
claim("ablation Newton r/d", "$1.2759$", 1.2759, runs["corner-label, Newton"]["r_over_d"][0], 5e-5)
claim("ablation nopert A/V", "$9.1981$", 9.1981, runs["corner-label, no perturbation"]["final_ratio"], 5e-5)
claim("roughness exact", "$0.022$", 0.022, runs["exact arrangement"]["roughness_rel"], 5e-4)
claim("roughness closed form", "$0.061$", 0.061, runs["corner-label, closed form"]["roughness_rel"], 5e-4)
claim("roughness Newton", "$1.437$", 1.437, runs["corner-label, Newton"]["roughness_rel"], 5e-4)
claim("optimal A/V", "$9.1394$", 9.1394, abl["optimal_ratio"], 5e-5)

for name, lo, hi in (("exact arrangement", 0.4592, 1.0091),
                     ("corner-label, closed form", 0.7967, 1.0644),
                     ("corner-label, no perturbation", 0.7033, 0.9679),
                     ("corner-label, Newton", 0.9044, 2.8705)):
    rs = [x["r_over_d"] for x in abl["basins"][name]]
    claim("basin lo %s" % name, "$%.4f$" % lo, lo, min(rs), 5e-5)
    claim("basin hi %s" % name, "$%.4f$" % hi, hi, max(rs), 5e-5)

ell = fcx["arm_b"]["ellipsoid"]
claim("armB ours rms", "$0.221$", 0.221, ell["ours"]["rms_dist_cells"], 5e-4)
claim("armB ours max", "$3.6$", 3.6, ell["ours"]["max_dist_cells"], 0.05)
claim("armB flexicubes rms", "$0.045$", 0.045, ell["flexicubes"]["rms_dist_cells"], 5e-4)
claim("armB flexicubes max", "$0.46$", 0.46, ell["flexicubes"]["max_dist_cells"], 5e-3)
claim("armB ours unsmoothed", "$0.457$", 0.457,
      min(s["rms_dist_cells"] for s in ell["_sweep"]
          if s["method"] == "ours" and s["reg"] == 0.0), 5e-4)


def arm(method):
    rows = [r for r in fcx["arm_c"] if r["method"] == method]
    return sorted(rows, key=lambda r: r["resolution"])


# Arm C is scored on 40,000 uniform samples, so every fraction is a multiple
# of 0.0025% and several land exactly on a .xx5 rounding boundary. The
# tolerance below admits either rounding convention; the table caption gives
# the sample count so a reader can recover the exact value either way.
claim("armC flexicubes overlap lo", "$0.90\\%$", 0.90, 100 * arm("flexicubes")[0]["overlap_frac"], 5e-3)
claim("armC flexicubes overlap hi", "$1.41\\%$", 1.41, 100 * arm("flexicubes")[-1]["overlap_frac"], 5e-3)
claim("armC one-vs-rest err lo", "$0.67\\%$", 0.67, 100 * arm("ours, one-vs-rest")[0]["mislabel_frac"], 6e-3)
claim("armC one-vs-rest err hi", "$0.08\\%$", 0.075, 100 * arm("ours, one-vs-rest")[-1]["mislabel_frac"], 6e-3)
claim("armC ours err lo", "$0.51\\%$", 0.51, 100 * arm("ours")[0]["mislabel_frac"], 6e-3)
claim("armC ours err hi", "$0.06\\%$", 0.0625, 100 * arm("ours")[-1]["mislabel_frac"], 6e-3)
claim("armC one-vs-rest no overlap", "the overlap is zero at", 0.0,
      max(r["overlap_frac"] for r in arm("ours, one-vs-rest")), 1e-12)

claim("clinical tetrahedra", "$663\\,552$", 663552, real["grid"]["tetrahedra"])
claim("clinical triangles", "$69\\,211$", 69211, real["ours"]["triangles"])
claim("clinical patches", "$11$ distinct", 11, real["ours"]["patches"])
claim("clinical triple segments", "$920$", 920, real["ours"]["triple_segments"])
claim("clinical quadruple points", "two quadruple points", 2, real["ours"]["vertex_kinds"]["quadruple"])
claim("clinical extract time", "$0.95$\\,s", 0.95, real["grid"]["extract_seconds"], 5e-3)
claim("clinical interface median", "$4.2\\times10^{-16}$", 4.2e-16,
      real["ours"]["interface_distance"]["median_mm"], 5e-17)
claim("surfacenets median mm", "$0.115$\\,mm", 0.115,
      real["surfacenets"]["interface_distance"]["median_mm"], 5e-4)


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------

keys = set()
for b in glob.glob(str(PAPER / "*.bib")):
    keys |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,",
                           Path(b).read_text(encoding="utf-8", errors="replace")))

cited = set()
for a in ("main.aux", "supplementary.aux"):
    path = PAPER / a
    if not path.exists():
        failures.append("NO AUX      %s -- build the paper before checking citations" % a)
        continue
    for m in re.findall(r"\\citation\{([^}]*)\}",
                        path.read_text(encoding="utf-8", errors="replace")):
        cited |= {k.strip() for k in m.split(",") if k.strip() and k.strip() != "*"}

if cited - keys:
    failures.append("CITED, NOT IN BIB: %s" % ", ".join(sorted(cited - keys)))
if keys - cited:
    failures.append("IN BIB, NEVER CITED: %s" % ", ".join(sorted(keys - cited)))


# --------------------------------------------------------------------------

print("%d numeric claims, %d bibliography entries, %d cited"
      % (_checked, len(keys), len(cited)))
if failures:
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("  everything agrees with its source")
