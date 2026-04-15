"""
dsa_v4_encoder.py — DSA v4 Physics-Integrated Encoder (Prototype)
==================================================================
Encodes a DSA v1–v3 coefficient stream into a v4 ring layout using
rotational color blending as the decoding mechanism.

See RESEARCH.md §13.2–13.6 for the physics model.

Architecture:
    DSA v1 MDCT coefficients
        │
        ▼
    v4 VisualEncoder
        │  maps coefficient values → print ratios via calibration LUT
        ▼
    Ring layout (list of rings, each ring = list of (color_a, color_b, ratio))
        │
        ▼
    v4 RingRenderer (uses calib_disc_gen primitives to produce printable bands)

Usage (standalone test):
    python3 dsa_v4_encoder.py --tables calibration_tables.json --demo

    Generates demo_v4_disc.png showing a 12-band test encoding using
    synthetic coefficients and the supplied calibration tables.

    With real audio:
    python3 dsa_v4_encoder.py --tables calibration_tables.json \\
                               --dsa audio.dsa1 --out disc_v4.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow required: pip install Pillow")

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required: pip install numpy")

# ── Constants ─────────────────────────────────────────────────────────────────

# Fallback calibration: ideal linear blending (for testing without a real
# calibration_tables.json).  ratio_pct 10–90 maps to luminance 0–255 linearly.
IDEAL_LUT = {
    'ratio_pct_to_lum': [(r, r * 255 / 100) for r in range(10, 100, 10)],
}

# Color pairs ranked by discriminability (from ideal physics; real order comes
# from calibration_tables.json pair_discriminability).
DEFAULT_PAIR_RANKING = [
    ('B', 'K'), ('R', 'K'), ('M', 'K'), ('R', 'B'),
    ('B', 'M'), ('R', 'M'), ('G', 'K'), ('G', 'B'),
    ('C', 'K'), ('B', 'C'), ('R', 'G'), ('G', 'M'),
]

COLORS = {
    'R': (255,   0,   0),
    'G': (  0, 200,   0),
    'B': (  0,   0, 255),
    'C': (  0, 200, 255),
    'M': (255,   0, 200),
    'Y': (255, 220,   0),
    'K': (  0,   0,   0),
    'W': (255, 255, 255),
}

# ── Calibration tables ────────────────────────────────────────────────────────

class CalibrationTables:
    """
    Wraps calibration_tables.json for encoder use.
    Falls back to ideal linear behavior when tables not available.
    """

    def __init__(self, path=None):
        self.tables = None
        self.pair_ranking = DEFAULT_PAIR_RANKING[:]

        if path and Path(path).exists():
            with open(path) as f:
                self.tables = json.load(f)
            print(f"Loaded calibration tables: {path}")
            self._parse_pair_ranking()
        else:
            print("No calibration tables — using ideal linear model")

    def _parse_pair_ranking(self):
        disc = self.tables.get('pair_discriminability', [])
        ranked = [(d['pair'].split('+')[0], d['pair'].split('+')[1])
                  for d in disc if d.get('readable', True)]
        if ranked:
            self.pair_ranking = ranked

    def coefficient_to_ratio(self, value, pair, speed_rpm=33.0):
        """
        Map a normalized coefficient value [0.0, 1.0] to a print ratio [0.0, 1.0].

        With calibration tables: invert the ratio→luminance curve via piecewise
        linear interpolation on the encoder_lut.

        Without tables: linear mapping (ratio = value).
        """
        if self.tables is None:
            return float(value)

        pair_key = f"{pair[0]}+{pair[1]}"
        lut = (self.tables.get('encoder_lut', {})
                          .get(pair_key, {})
                          .get(str(float(speed_rpm)), None))

        # Try nearby speed keys if exact not found
        if lut is None:
            for sp_key in self.tables.get('encoder_lut', {}).get(pair_key, {}):
                lut = self.tables['encoder_lut'][pair_key][sp_key]
                break

        if lut is None or len(lut) < 2:
            return float(value)

        # lut is sorted by luminance; we want to map value (0–1) through
        # a target luminance, then find the ratio
        lum_min = lut[0]['luminance']
        lum_max = lut[-1]['luminance']
        target_lum = lum_min + value * (lum_max - lum_min)

        # Interpolate ratio_pct from luminance
        for i in range(len(lut) - 1):
            lo, hi = lut[i], lut[i + 1]
            if lo['luminance'] <= target_lum <= hi['luminance']:
                t = ((target_lum - lo['luminance']) /
                     (hi['luminance'] - lo['luminance'] + 1e-9))
                ratio_pct = lo['ratio_pct'] + t * (hi['ratio_pct'] - lo['ratio_pct'])
                return ratio_pct / 100.0

        return float(value)

    def best_pair_for_band(self, band_index):
        """Return the color pair to use for a given frequency band index."""
        if band_index < len(self.pair_ranking):
            return self.pair_ranking[band_index]
        # Wrap around
        return self.pair_ranking[band_index % len(self.pair_ranking)]

    def min_module_mm(self, speed_rpm=33.0):
        """Return the minimum usable module size in mm at the given speed."""
        if self.tables is None:
            return 0.5
        thr = self.tables.get('density_threshold', {})
        sp_key = str(float(speed_rpm))
        if sp_key in thr:
            return thr[sp_key]['module_mm']
        # Fall back to smallest available
        if thr:
            return min(v['module_mm'] for v in thr.values())
        return 0.5


# ── v4 band/ring model ────────────────────────────────────────────────────────

class V4Ring:
    """One printable ring encoding one frequency band at one time slice."""

    def __init__(self, band_idx, time_slice, pair, ratio, r_out_mm, r_in_mm):
        self.band_idx   = band_idx
        self.time_slice = time_slice
        self.pair       = pair      # (color_a_name, color_b_name)
        self.ratio      = ratio     # fraction of color_a [0.0, 1.0]
        self.r_out_mm   = r_out_mm
        self.r_in_mm    = r_in_mm

    def __repr__(self):
        return (f"V4Ring(band={self.band_idx} t={self.time_slice} "
                f"{self.pair[0]}+{self.pair[1]} ratio={self.ratio:.2f} "
                f"r={self.r_in_mm:.1f}–{self.r_out_mm:.1f}mm)")


# ── Encoder ───────────────────────────────────────────────────────────────────

class V4Encoder:
    """
    Encodes a 2-D coefficient array (bands × time_slices) into a list of V4Ring
    objects ready for rendering.

    Layout strategy:
        - Outermost rings → highest frequency bands (L2 — best optics needed)
        - Innermost rings → lowest frequency bands (L0 — always readable)
        - Within each band: rings advance inward as time progresses
        - Each ring encodes one (band, time_slice) coefficient
    """

    def __init__(self, calib: CalibrationTables,
                 outer_r_mm=149.4, inner_r_mm=25.0,
                 gap_mm=0.2, speed_rpm=33.0):
        self.calib       = calib
        self.outer_r_mm  = outer_r_mm
        self.inner_r_mm  = inner_r_mm
        self.gap_mm      = gap_mm
        self.speed_rpm   = speed_rpm

    def encode(self, coefficients: np.ndarray) -> list:
        """
        coefficients: shape (n_bands, n_time_slices), values in [0.0, 1.0]
            representing normalized spectral magnitude.

        Returns list of V4Ring objects.
        """
        n_bands, n_time = coefficients.shape
        total_rings = n_bands * n_time
        span_mm = self.outer_r_mm - self.inner_r_mm
        ring_width = (span_mm - total_rings * self.gap_mm) / total_rings

        if ring_width < 0.1:
            raise ValueError(
                f"Not enough radial space for {total_rings} rings "
                f"({span_mm:.1f}mm span). Reduce n_bands×n_time or increase disc size."
            )

        rings = []
        r = self.outer_r_mm

        # Outer bands = high frequency (band index n_bands-1 first)
        for band_idx in range(n_bands - 1, -1, -1):
            pair = self.calib.best_pair_for_band(band_idx)
            for t in range(n_time):
                coeff = float(coefficients[band_idx, t])
                ratio = self.calib.coefficient_to_ratio(coeff, pair, self.speed_rpm)
                r_out = r
                r_in  = r - ring_width
                rings.append(V4Ring(band_idx, t, pair, ratio, r_out, r_in))
                r = r_in - self.gap_mm

        return rings


# ── Renderer ──────────────────────────────────────────────────────────────────

def mm2px(mm, dpi):
    return mm * dpi / 25.4


def _checker_tile(color_a, color_b, ratio, cell_px):
    """Build a checkerboard tile representing the given ratio."""
    period = 10
    n_a = max(1, min(period - 1, round(ratio * period)))
    tile_w = period * cell_px
    tile_h = cell_px * 2
    tile = Image.new('RGB', (tile_w, tile_h))
    d = ImageDraw.Draw(tile)
    row0 = [color_a if i < n_a else color_b for i in range(period)]
    row1 = [color_a if (i + (period - n_a)) % period < n_a else color_b
            for i in range(period)]
    for i, c in enumerate(row0):
        d.rectangle([i*cell_px, 0, (i+1)*cell_px-1, cell_px-1], fill=c)
    for i, c in enumerate(row1):
        d.rectangle([i*cell_px, cell_px, (i+1)*cell_px-1, tile_h-1], fill=c)
    return tile


def render_rings(rings: list, dpi=300, dia_mm=304.8, module_mm=0.5) -> Image.Image:
    """Render V4Ring list to a PIL Image."""
    img_px = int(mm2px(dia_mm, dpi))
    cx = cy = img_px // 2
    cell_px = max(2, int(mm2px(module_mm, dpi)))

    img = Image.new('RGB', (img_px, img_px), (230, 230, 230))

    for ring in rings:
        r_out_px = mm2px(ring.r_out_mm, dpi)
        r_in_px  = mm2px(ring.r_in_mm,  dpi)
        ca = COLORS[ring.pair[0]]
        cb = COLORS[ring.pair[1]]

        # Build tiled pattern
        tile = _checker_tile(ca, cb, ring.ratio, cell_px)
        layer = Image.new('RGB', (img_px, img_px))
        tw, th = tile.size
        for y in range(0, img_px, th):
            for x in range(0, img_px, tw):
                layer.paste(tile, (x, y))

        # Annulus mask
        mask = Image.new('L', (img_px, img_px), 0)
        dm = ImageDraw.Draw(mask)
        dm.ellipse([cx-r_out_px, cy-r_out_px, cx+r_out_px, cy+r_out_px], fill=255)
        dm.ellipse([cx-r_in_px,  cy-r_in_px,  cx+r_in_px,  cy+r_in_px],  fill=0)
        img.paste(layer, mask=mask)

    # Outer edge
    d = ImageDraw.Draw(img)
    r_edge = mm2px(dia_mm / 2 - 1, dpi)
    d.ellipse([cx-r_edge, cy-r_edge, cx+r_edge, cy+r_edge],
              outline=(0, 0, 0), width=2)

    return img


# ── Demo ──────────────────────────────────────────────────────────────────────

def make_demo_coefficients(n_bands=12, n_time=8, seed=0):
    """
    Generate synthetic coefficient array for demo purposes.
    Simulates a signal with decreasing energy at higher bands (natural audio).
    """
    rng = np.random.default_rng(seed)
    # Base: pink-noise-like falloff across bands
    band_weights = np.array([1.0 / (1 + b * 0.5) for b in range(n_bands)])
    coeff = rng.uniform(0, 1, (n_bands, n_time))
    coeff *= band_weights[:, np.newaxis]
    # Normalize to [0, 1]
    coeff = coeff / coeff.max()
    return coeff


def run_demo(args):
    calib = CalibrationTables(args.tables)

    n_bands = 12
    n_time  = 8
    coefficients = make_demo_coefficients(n_bands, n_time)

    print(f"Demo: {n_bands} bands × {n_time} time slices = {n_bands*n_time} rings")
    print(f"Coefficient range: {coefficients.min():.3f} – {coefficients.max():.3f}")

    encoder = V4Encoder(calib, speed_rpm=33.0)
    rings = encoder.encode(coefficients)

    print(f"Encoded {len(rings)} rings")
    print(f"  Outermost: {rings[0]}")
    print(f"  Innermost: {rings[-1]}")

    # Print band→pair assignment
    print("\nBand → color pair assignment:")
    for b in range(n_bands):
        pair = calib.best_pair_for_band(b)
        print(f"  Band {b:2d}: {pair[0]}+{pair[1]}")

    module_mm = calib.min_module_mm(33.0)
    out_path  = args.out or 'demo_v4_disc.png'
    print(f"\nRendering at {args.dpi} DPI, module size {module_mm}mm...")
    img = render_rings(rings, dpi=args.dpi, dia_mm=args.dia, module_mm=module_mm)
    img.save(out_path, dpi=(args.dpi, args.dpi))
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DSA v4 physics-integrated encoder prototype")
    p.add_argument('--tables', default='calibration_tables.json',
                   help='calibration_tables.json path')
    p.add_argument('--demo',   action='store_true', help='Run with synthetic coefficients')
    p.add_argument('--dsa',    help='DSA v1 .dsa1 file to encode (future)')
    p.add_argument('--out',    help='Output PNG path')
    p.add_argument('--dpi',    type=float, default=300)
    p.add_argument('--dia',    type=float, default=304.8, help='Disc diameter mm')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.demo or not args.dsa:
        run_demo(args)
    else:
        print("DSA file encoding not yet implemented — run with --demo")
