"""Derived figure crops for the paper.

    realdata_liver_junction.png panels (c) and (d) of realdata_liver.png

The crop is cut along an all-white horizontal band of the source image, so
regenerating the source (figures_realdata.py) and re-running this script
keeps it consistent. Panels (a) and (b) are not cropped: the main paper uses
realdata_liver_io.png, drawn by figures_realdata.figure_realdata_io.
"""
import os

import numpy as np
from PIL import Image

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def white_bands(img, thresh=250):
    """Rows that are entirely (near-)white, as a list of (start, end) runs."""
    a = np.asarray(img.convert("L"))
    white = (a >= thresh).all(axis=1)
    bands, start = [], None
    for y, w in enumerate(white):
        if w and start is None:
            start = y
        if not w and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(white)))
    return bands


def trim(img, pad=6, thresh=250):
    """Remove the white margin around the content, keeping `pad` pixels."""
    a = np.asarray(img.convert("L"))
    ys, xs = np.where(a < thresh)
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0]))
    return img.crop(box)


def split_at_band(img, lo_frac, hi_frac, min_width=20):
    """Cut the image at the first all-white band of at least `min_width` rows
    whose centre lies in [lo_frac, hi_frac] of the height.
    Returns (top, bottom)."""
    h = img.height
    bands = [b for b in white_bands(img)
             if lo_frac * h <= 0.5 * (b[0] + b[1]) <= hi_frac * h
             and b[1] - b[0] >= min_width]
    if not bands:
        raise RuntimeError("no white band in the requested range")
    s, e = bands[0]
    cut = (s + e) // 2
    return img.crop((0, 0, img.width, cut)), img.crop((0, cut, img.width, h))


def realdata_split():
    img = Image.open(os.path.join(FIGDIR, "realdata_liver.png"))
    # the first wide band below the (b) legend separates the two rows
    _, bottom = split_at_band(img, 0.48, 0.56)
    bottom = trim(bottom)
    bottom.save(os.path.join(FIGDIR, "realdata_liver_junction.png"))
    print("wrote realdata_liver_junction.png", bottom.size)


if __name__ == "__main__":
    realdata_split()
