"""Measure how much of the paper each section occupies.

Trimming toward a page target needs a budget, not an impression. This reads
the section titles out of main.aux, locates each one in the rendered PDF, and
reports the column-height it spans. Units are column-pages: 1.0 means a
section fills one full page of the two-column body, so the numbers sum to the
body length.

    python src/page_budget.py            # per-subsection detail
    python src/page_budget.py --top      # top-level sections only
    python src/page_budget.py --ink      # split the body into image/text/gap

Use --ink when prose cuts stop paying: it says how much of the body is
actually text, and therefore how much of the length is reachable by editing
rather than by dropping floats.

The paper must have been built at least once, since both the .aux and the
.pdf are read.
"""
import re
import sys
from pathlib import Path

import pymupdf

PAPER = Path(__file__).resolve().parents[1] / "paper"
doc = pymupdf.open(PAPER / "main.pdf")
aux = (PAPER / "main.aux").read_text(encoding="utf-8", errors="replace")

toc = [(num, re.sub(r"\\\w+\s*|[{}]", "", title).strip())
       for num, title in re.findall(
           r"\\contentsline \{\w+\}\{\\numberline \{([\d.]+)\}([^}]*)\}", aux)]

page_h = doc[0].rect.height
mid_x = doc[0].rect.width / 2
top, bot = 60.0, page_h - 55.0
col_h = bot - top


def flat(pno, x, y):
    """Position along the single reading thread, in column-pages."""
    col = 0 if x < mid_x else 1
    return pno + (col + (min(max(y, top), bot) - top) / col_h) / 2.0


def find(needle):
    """First occurrence of `needle` set in a heading-sized font."""
    for pno, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                txt = "".join(s["text"] for s in line["spans"]).strip()
                if txt.lower().startswith(needle.lower()) and line["spans"][0]["size"] > 8.6:
                    s = line["spans"][0]
                    return flat(pno, s["bbox"][0], s["bbox"][1])
    return None


marks = []
for num, title in toc:
    pos = find("%s %s" % (num, title)) or find("%s. %s" % (num, title)) or find(title)
    if pos is not None:
        marks.append((pos, num, title))
marks.sort()
end = find("References")

if "--ink" in sys.argv:
    # One page of body is two columns of live area.
    cap = 2 * col_h
    body_pages = int(end)
    tot_img = tot_txt = 0.0
    print("page   image    text     gap   (column-points, %.0f per page)" % cap)
    for pno in range(body_pages):
        page = doc[pno]
        img = sum((i["bbox"][3] - i["bbox"][1]) *
                  (2.0 if (i["bbox"][2] - i["bbox"][0]) > doc[0].rect.width * 0.6 else 1.0)
                  for i in page.get_image_info())
        txt = sum(l["bbox"][3] - l["bbox"][1]
                  for b in page.get_text("dict")["blocks"] for l in b.get("lines", []))
        tot_img, tot_txt = tot_img + img, tot_txt + txt
        print("%4d  %6.0f  %6.0f  %6.0f" % (pno + 1, img, txt, cap - img - txt))
    gap = body_pages * cap - tot_img - tot_txt
    print("-" * 52)
    print("image %6.2f pages   text %6.2f pages (glyph boxes, ~1.25x with leading)"
          % (tot_img / cap, tot_txt / cap))
    print("gap   %6.2f pages   float placement, headings and inter-line space" % (gap / cap))
    raise SystemExit

only_top = "--top" in sys.argv
print("%-46s %7s %7s" % ("section", "pages", "starts"))
print("-" * 64)
for i, (pos, num, title) in enumerate(marks):
    nxt = marks[i + 1][0] if i + 1 < len(marks) else end
    depth = num.count(".")
    if only_top:
        # Span a top-level section to the next top-level section, not the next
        # subsection, so the numbers still sum to the body length.
        if depth:
            continue
        later = [m[0] for m in marks[i + 1:] if not m[1].count(".")]
        nxt = later[0] if later else end
    print("%-46s %7.2f %7.2f"
          % (("    " * depth + num + " " + title)[:46], nxt - pos, pos))
print("-" * 64)
print("body %.2f column-pages; PDF is %d pages with references"
      % (end, doc.page_count))
