"""Build the two Main Manuscript files CGF's Research Exchange asks for.

The form wants a LaTeX archive --- "a single archive including all LaTeX
files, BibTeX files, figures, tables, all LaTeX classes and packages" ---
and, separately, "a single, compiled PDF output generated from your LaTeX
main document file(s)". Both come out of here, and the PDF is compiled from
the unpacked archive rather than from paper/, so the proof is guaranteed to
be what the submitted source actually produces.

Two things are deliberate. Figures are flattened next to main.tex and
\\graphicspath is dropped, because the archive is unpacked flat and a path
pointing at ../figures/ would not resolve. And the supplement is excluded:
the upload form states the main manuscript "should not include any
supplementary materials", which goes in its own slot as a PDF.

    python paper/make_submission.py

Writes submission-cgf.zip and submission-cgf.pdf to the repository root.
Needs pdflatex and bibtex on PATH.
"""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "submission-cgf"
ARCHIVE = ROOT / "submission-cgf.zip"
PROOF = ROOT / "submission-cgf.pdf"

# The Eurographics class, the font descriptors it loads, the logo it draws on
# the title page, and the bibliography style the .tex names.
SUPPORT = [
    "egpubl.cls",
    "dfadobe.sty",
    "dfT1pcr.fd",
    "dfT1phv.fd",
    "dfT1ptm.fd",
    "egweblnk.sty",
    "eg_new.jpg",
    "orcid.pdf",
    "eg-alpha-doi.bst",
    "refs.bib",
    # Bundled so the archive compiles even where bibtex is not run: Wiley's
    # converter is not guaranteed to make a second pass.
    "main.bbl",
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

src = (HERE / "main.tex").read_text(encoding="utf8")
src = src.replace("\\graphicspath{{../figures/}}\n", "")

figures = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src)))
for name in figures:
    shutil.copy(ROOT / "figures" / name, OUT / name)

(OUT / "main.tex").write_text(src, encoding="utf8")
for name in SUPPORT:
    shutil.copy(HERE / name, OUT / name)

# Zip before compiling, so no .aux/.log lands in the submitted archive.
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(OUT))

for step in (["pdflatex", "-interaction=nonstopmode", "main.tex"],
             ["bibtex", "main"],
             ["pdflatex", "-interaction=nonstopmode", "main.tex"],
             ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
    subprocess.run(step, cwd=OUT, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)

built = OUT / "main.pdf"
if not built.exists():
    raise SystemExit("the archive did not compile; see %s" % (OUT / "main.log"))
shutil.copy(built, PROOF)

log = (OUT / "main.log").read_text(encoding="utf8", errors="replace")
pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
unresolved = "There were undefined references" in log

print("%s  %d figures, %d support files, %.1f MB"
      % (ARCHIVE.name, len(figures), len(SUPPORT), ARCHIVE.stat().st_size / 1e6))
print("%s  %s pages, %.1f MB, undefined references: %s"
      % (PROOF.name, pages.group(1) if pages else "?",
         PROOF.stat().st_size / 1e6, unresolved))
print()
print("Main Manuscript slot : %s  then  %s" % (ARCHIVE.name, PROOF.name))
print("supplementary slot   : paper/supplementary.pdf")
