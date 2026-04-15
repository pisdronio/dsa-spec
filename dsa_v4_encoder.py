"""
dsa_v4_encoder.py — DSA v4 Physics-Integrated Encoder
======================================================
Encodes a .dsa1 file (or synthetic demo) into a printable v4 disc PNG
using rotational color blending as the decoding mechanism.

See RESEARCH.md §13.2–13.6 for the physics model.

Pipeline:
    .dsa1 file
        │  DSA1Reader.extract_band_energies()
        ▼
    48-band × n_frames energy array
        │  BandMapper.map()
        ▼
    n_v4_bands × n_time coefficient array  (fits disc geometry)
        │  CalibrationTables.coefficient_to_ratio()
        ▼
    RingDescriptor list  (sync + header + data rings)
        │  render_full_disc()
        ▼
    Printable disc PNG

Usage:
    # Real audio
    python3 dsa_v4_encoder.py --dsa audio.dsa1 --out disc_v4.png

    # Synthetic demo
    python3 dsa_v4_encoder.py --demo --out demo_v4_disc.png

    # Options
    --tables  calibration_tables.json   (default: calibration_tables.json)
    --speed   33                        (target rpm, default: 33)
    --time    16                        (time slices per disc, default: auto)
    --dpi     300
    --dia     304.8                     (disc diameter mm, default: 12")
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

from dsa_v4_format import (
    V4Header, ZoneGeometry, build_ring_list, RingDescriptor,
    HEADER_RING_COLOR_A, HEADER_RING_COLOR_B,
)
from dsa_v4_reader import (
    DSA1Reader, NUM_BANDS, L0_BANDS, L1_BANDS, L2_BANDS, BINS, NYQUIST,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_RING_WIDTH_MM = 0.3    # hard floor — narrower than this won't print cleanly

# Fallback pair ranking (used when calibration tables not available).
# Ordered by expected discriminability: dark pairs first, light last.
DEFAULT_PAIR_RANKING = [
    ('B', 'K'), ('R', 'K'), ('M', 'K'), ('R', 'B'),
    ('B', 'M'), ('R', 'M'), ('G', 'K'), ('G', 'B'),
    ('C', 'K'), ('B', 'C'), ('R', 'G'), ('G', 'M'),
    ('C', 'M'), ('R', 'C'), ('G', 'C'), ('R', 'Y'),
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

# ── Band mapper ───────────────────────────────────────────────────────────────

class BandMapper:
    """
    Maps 48 DSA frequency bands to n_v4_bands v4 rings using log-spaced
    aggregation (perceptual strategy) or uniform aggregation.

    Log spacing gives low-frequency bands more v4 rings, which matches
    perceptual importance — bass carries more musical content per Hz than
    treble.

    Aggregation: max across DSA sub-bands within each v4 band.  Max
    preserves signal peaks (e.g. a strong tone in one sub-band should
    dominate its v4 ring) rather than diluting it with silent neighbors.
    """

    def __init__(self, n_v4_bands: int, strategy: str = 'perceptual'):
        self.n_v4_bands = n_v4_bands
        self.strategy   = strategy
        self._mapping   = self._build()

    def _build(self) -> list:
        """Return list of length n_v4_bands; each entry is a list of DSA band indices."""
        # Center frequency of each DSA band in Hz
        sample_rate = 44100
        mdct_m      = 512
        band_centers = [
            sample_rate * (lo + hi) / 2 / (2 * mdct_m)
            for lo, hi in BINS
        ]

        if self.strategy == 'perceptual':
            # Log-spaced edges from 20 Hz to Nyquist
            f_lo  = 20.0
            f_hi  = float(NYQUIST)
            edges = [f_lo * (f_hi / f_lo) ** (i / self.n_v4_bands)
                     for i in range(self.n_v4_bands + 1)]
        else:  # 'uniform'
            edges = [i * NYQUIST / self.n_v4_bands
                     for i in range(self.n_v4_bands + 1)]

        mapping = []
        for i in range(self.n_v4_bands):
            lo, hi = edges[i], edges[i + 1]
            members = [b for b, fc in enumerate(band_centers) if lo <= fc < hi]
            if not members:
                # Edge case: no DSA band falls here — snap to nearest
                nearest = min(range(NUM_BANDS),
                              key=lambda b: abs(band_centers[b] - (lo + hi) / 2))
                members = [nearest]
            mapping.append(members)

        return mapping

    def map(self, energies: np.ndarray) -> np.ndarray:
        """
        energies: shape (NUM_BANDS, n_frames), values in [0, 1]
        Returns: shape (n_v4_bands, n_frames), values in [0, 1]
        """
        n_frames = energies.shape[1]
        out = np.zeros((self.n_v4_bands, n_frames), dtype=np.float32)
        for vi, dsa_bands in enumerate(self._mapping):
            out[vi] = energies[dsa_bands, :].max(axis=0)
        return out

    def describe(self) -> list:
        """Return human-readable band descriptions for the legend."""
        sample_rate = 44100
        mdct_m      = 512
        band_hz = [(sample_rate * lo / (2 * mdct_m),
                    sample_rate * hi / (2 * mdct_m))
                   for lo, hi in BINS]

        descriptions = []
        for vi, dsa_bands in enumerate(self._mapping):
            lo_hz = band_hz[dsa_bands[0]][0]
            hi_hz = band_hz[dsa_bands[-1]][1]
            descriptions.append(
                f"v4 band {vi}: {lo_hz:.0f}–{hi_hz:.0f} Hz "
                f"(DSA bands {dsa_bands[0]}–{dsa_bands[-1]})"
            )
        return descriptions

    @staticmethod
    def max_bands(geom: 'ZoneGeometry', n_time: int,
                  min_width_mm: float = MIN_RING_WIDTH_MM) -> int:
        """Maximum v4 bands that fit on the disc given n_time slices."""
        # data_span = ring_width * n_bands * n_time + gap * n_bands * n_time
        # Solving for n_bands:
        n_rings_max = int(geom.data_span_mm / (min_width_mm + geom.gap_mm))
        return max(1, n_rings_max // n_time)


# ── Calibration tables ────────────────────────────────────────────────────────

class CalibrationTables:
    """
    Wraps calibration_tables.json.
    Falls back to ideal linear behaviour when the file is absent or empty.
    Replace calibration_tables.json with real measurements (from
    calib_build_tables.py) to get physically accurate encoding.
    """

    def __init__(self, path=None):
        self.tables       = None
        self.pair_ranking = DEFAULT_PAIR_RANKING[:]

        if path and Path(path).exists():
            with open(path) as f:
                self.tables = json.load(f)
            print(f"Loaded calibration tables: {path}")
            self._parse_pair_ranking()
        else:
            print("No calibration tables — using ideal linear model "
                  "(replace calibration_tables.json with real measurements)")

    def _parse_pair_ranking(self):
        disc = self.tables.get('pair_discriminability', [])
        ranked = [(d['pair'].split('+')[0], d['pair'].split('+')[1])
                  for d in disc if d.get('readable', True)]
        if ranked:
            self.pair_ranking = ranked

    def coefficient_to_ratio(self, value: float, pair: tuple,
                              speed_rpm: float = 33.0) -> float:
        """
        Map normalised coefficient [0, 1] → print ratio [0, 1].

        With real tables: inverts the measured ratio→luminance curve for
        this color pair and speed via piecewise linear interpolation.

        Without real tables: identity mapping (ratio = value).
        """
        if self.tables is None:
            return float(value)

        pair_key = f"{pair[0]}+{pair[1]}"
        lut = (self.tables.get('encoder_lut', {})
                          .get(pair_key, {})
                          .get(str(float(speed_rpm))))

        if lut is None:
            # Try any available speed for this pair
            avail = self.tables.get('encoder_lut', {}).get(pair_key, {})
            lut = next(iter(avail.values()), None) if avail else None

        if lut is None or len(lut) < 2:
            return float(value)

        lum_min    = lut[0]['luminance']
        lum_max    = lut[-1]['luminance']
        target_lum = lum_min + value * (lum_max - lum_min)

        for i in range(len(lut) - 1):
            lo, hi = lut[i], lut[i + 1]
            if lo['luminance'] <= target_lum <= hi['luminance']:
                t = ((target_lum - lo['luminance']) /
                     (hi['luminance'] - lo['luminance'] + 1e-9))
                return (lo['ratio_pct'] + t * (hi['ratio_pct'] - lo['ratio_pct'])) / 100.0

        return float(value)

    def best_pair_for_band(self, band_index: int) -> tuple:
        return self.pair_ranking[band_index % len(self.pair_ranking)]

    def min_module_mm(self, speed_rpm: float = 33.0) -> float:
        if self.tables is None:
            return 0.5
        thr    = self.tables.get('density_threshold', {})
        sp_key = str(float(speed_rpm))
        if sp_key in thr:
            return thr[sp_key]['module_mm']
        if thr:
            return min(v['module_mm'] for v in thr.values())
        return 0.5


# ── Ring model ────────────────────────────────────────────────────────────────

class V4Ring:
    """One printable ring: one v4 band at one time slice."""
    __slots__ = ('band_idx', 'time_slice', 'pair', 'ratio', 'r_out_mm', 'r_in_mm')

    def __init__(self, band_idx, time_slice, pair, ratio, r_out_mm, r_in_mm):
        self.band_idx   = band_idx
        self.time_slice = time_slice
        self.pair       = pair
        self.ratio      = ratio
        self.r_out_mm   = r_out_mm
        self.r_in_mm    = r_in_mm

    def __repr__(self):
        return (f"V4Ring(band={self.band_idx} t={self.time_slice} "
                f"{self.pair[0]}+{self.pair[1]} ratio={self.ratio:.2f} "
                f"r={self.r_in_mm:.1f}–{self.r_out_mm:.1f}mm)")


# ── Renderer ──────────────────────────────────────────────────────────────────

def _mm2px(mm, dpi):
    return mm * dpi / 25.4


def _checker_tile(color_a, color_b, ratio, cell_px):
    period = 10
    n_a    = max(1, min(period - 1, round(ratio * period)))
    tile   = Image.new('RGB', (period * cell_px, cell_px * 2))
    d      = ImageDraw.Draw(tile)
    row0   = [color_a if i < n_a else color_b for i in range(period)]
    row1   = [color_a if (i + period - n_a) % period < n_a else color_b
              for i in range(period)]
    for i, c in enumerate(row0):
        d.rectangle([i*cell_px, 0, (i+1)*cell_px-1, cell_px-1], fill=c)
    for i, c in enumerate(row1):
        d.rectangle([i*cell_px, cell_px, (i+1)*cell_px-1, cell_px*2-1], fill=c)
    return tile


def _draw_ring(img, cx, cy, r_out_px, r_in_px, color_a, color_b, ratio, cell_px):
    size = img.size[0]
    tile = _checker_tile(color_a, color_b, ratio, cell_px)
    layer = Image.new('RGB', (size, size))
    tw, th = tile.size
    for y in range(0, size, th):
        for x in range(0, size, tw):
            layer.paste(tile, (x, y))
    mask = Image.new('L', (size, size), 0)
    dm = ImageDraw.Draw(mask)
    dm.ellipse([cx-r_out_px, cy-r_out_px, cx+r_out_px, cy+r_out_px], fill=255)
    dm.ellipse([cx-r_in_px,  cy-r_in_px,  cx+r_in_px,  cy+r_in_px],  fill=0)
    img.paste(layer, mask=mask)


def render_full_disc(ring_descs: list, dpi: float = 300,
                     dia_mm: float = 304.8, module_mm: float = 0.5) -> Image.Image:
    """
    Render a list of RingDescriptor objects to a PIL Image.
    Handles sync (solid), header (B+W ratio), and data (pair+ratio) rings.
    """
    img_px  = int(_mm2px(dia_mm, dpi))
    cx = cy = img_px // 2
    cell_px = max(2, int(_mm2px(module_mm, dpi)))
    img = Image.new('RGB', (img_px, img_px), (230, 230, 230))

    for rd in ring_descs:
        r_out = _mm2px(rd.r_out_mm, dpi)
        r_in  = _mm2px(rd.r_in_mm,  dpi)

        if rd.kind == 'sync':
            c = COLORS[rd.color]
            _draw_ring(img, cx, cy, r_out, r_in, c, c, 1.0, cell_px)

        elif rd.kind == 'header':
            ca = COLORS[HEADER_RING_COLOR_A]
            cb = COLORS[HEADER_RING_COLOR_B]
            _draw_ring(img, cx, cy, r_out, r_in, ca, cb,
                       rd.byte_val / 255.0, cell_px)

        elif rd.kind == 'data':
            ca = COLORS[rd.pair[0]]
            cb = COLORS[rd.pair[1]]
            _draw_ring(img, cx, cy, r_out, r_in, ca, cb, rd.ratio, cell_px)

    d = ImageDraw.Draw(img)
    r_edge = _mm2px(dia_mm / 2 - 1, dpi)
    d.ellipse([cx-r_edge, cy-r_edge, cx+r_edge, cy+r_edge],
              outline=(0, 0, 0), width=2)
    return img


# ── Legacy render_rings (for callers that pass V4Ring lists) ──────────────────

def render_rings(rings: list, dpi=300, dia_mm=304.8, module_mm=0.5) -> Image.Image:
    """Render a V4Ring list (legacy interface — converts to RingDescriptors)."""
    descs = []
    for r in rings:
        descs.append(RingDescriptor(
            kind='data', r_out_mm=r.r_out_mm, r_in_mm=r.r_in_mm,
            pair=r.pair, ratio=r.ratio,
        ))
    return render_full_disc(descs, dpi=dpi, dia_mm=dia_mm, module_mm=module_mm)


# ── DSA file encoder ──────────────────────────────────────────────────────────

def encode_dsa1(dsa_path: str, calib: CalibrationTables,
                speed_rpm: float = 33.0, n_time: int = None,
                dia_mm: float = 304.8, dpi: float = 300,
                out_path: str = None) -> Image.Image:
    """
    Full pipeline: .dsa1 → printable disc PNG.

    n_time: number of time slices (frames) to encode per disc side.
            Defaults to the maximum that fits given disc geometry.
    """
    reader = DSA1Reader(dsa_path)
    print(f"Input:  {reader}")

    if not reader.verify_crc():
        print("WARNING: CRC mismatch — file may be corrupted")

    # Disc geometry
    outer_r = dia_mm / 2 - 3.0
    inner_r = 25.0 if dia_mm >= 280 else 12.0
    geom    = ZoneGeometry(outer_r_mm=outer_r, inner_r_mm=inner_r)

    # Time slices: default = all frames, capped to what fits
    if n_time is None:
        n_time = reader.n_frames
    n_time = min(n_time, reader.n_frames)

    # Maximum bands that fit on this disc
    n_v4_bands = BandMapper.max_bands(geom, n_time)
    print(f"Layout: {n_v4_bands} bands × {n_time} time slices = "
          f"{n_v4_bands * n_time} data rings")
    ring_w = geom.data_ring_width_mm(n_v4_bands * n_time)
    print(f"        ring width = {ring_w:.3f} mm")

    # Extract and map energies
    energies = reader.extract_band_energies()          # (48, n_frames)
    energies  = energies[:, :n_time]                   # trim to n_time
    mapper    = BandMapper(n_v4_bands, strategy='perceptual')
    coeffs    = mapper.map(energies)                   # (n_v4_bands, n_time)

    print("Band mapping (perceptual, log-spaced):")
    for desc in mapper.describe():
        print(f"  {desc}")

    # Build header + ring list
    header = V4Header(
        n_bands        = n_v4_bands,
        n_time         = n_time,
        speed_rpm      = int(speed_rpm),
        sample_rate    = reader.sample_rate,
        n_audio_frames = reader.n_frames,
    )
    print(f"Header: {header}  bytes={header.to_bytes().hex()}")

    rings = build_ring_list(header, geom, coeffs, calib)
    n_sync   = sum(1 for r in rings if r.kind == 'sync')
    n_header = sum(1 for r in rings if r.kind == 'header')
    n_data   = sum(1 for r in rings if r.kind == 'data')
    print(f"Rings:  {len(rings)} total  ({n_sync} sync + {n_header} header + {n_data} data)")

    # Render
    module_mm = calib.min_module_mm(speed_rpm)
    print(f"Render: {dpi} DPI, module {module_mm} mm ...")
    img = render_full_disc(rings, dpi=dpi, dia_mm=dia_mm, module_mm=module_mm)

    if out_path:
        img.save(out_path, dpi=(dpi, dpi))
        print(f"Saved:  {out_path}")

    return img


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo(args):
    calib  = CalibrationTables(args.tables)
    outer_r = args.dia / 2 - 3.0
    inner_r = 25.0 if args.dia >= 280 else 12.0
    geom    = ZoneGeometry(outer_r_mm=outer_r, inner_r_mm=inner_r)

    n_time     = args.time or 8
    n_v4_bands = BandMapper.max_bands(geom, n_time)
    print(f"Demo: {n_v4_bands} bands × {n_time} time slices = "
          f"{n_v4_bands * n_time} data rings")

    rng     = np.random.default_rng(0)
    weights = np.array([1.0 / (1 + b * 0.5) for b in range(n_v4_bands)])
    coeffs  = rng.uniform(0, 1, (n_v4_bands, n_time)).astype(np.float32)
    coeffs *= weights[:, np.newaxis]
    coeffs /= coeffs.max()

    header = V4Header(n_bands=n_v4_bands, n_time=n_time,
                      speed_rpm=int(args.speed))
    rings  = build_ring_list(header, geom, coeffs, calib)

    module_mm = calib.min_module_mm(args.speed)
    out_path  = args.out or 'demo_v4_disc.png'
    img = render_full_disc(rings, dpi=args.dpi, dia_mm=args.dia, module_mm=module_mm)
    img.save(out_path, dpi=(args.dpi, args.dpi))
    print(f"Saved: {out_path}  ({n_v4_bands * n_time + 11} rings total)")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DSA v4 physics-integrated encoder")
    p.add_argument('--dsa',    help='.dsa1 input file')
    p.add_argument('--demo',   action='store_true', help='Run with synthetic coefficients')
    p.add_argument('--tables', default='calibration_tables.json')
    p.add_argument('--speed',  type=float, default=33.0, help='Target disc speed (rpm)')
    p.add_argument('--time',   type=int,   default=None, help='Time slices per side')
    p.add_argument('--out',    help='Output PNG path')
    p.add_argument('--dpi',    type=float, default=300)
    p.add_argument('--dia',    type=float, default=304.8, help='Disc diameter mm')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    calib = CalibrationTables(args.tables)

    if args.dsa:
        out = args.out or (Path(args.dsa).stem + '_v4.png')
        encode_dsa1(args.dsa, calib,
                    speed_rpm=args.speed, n_time=args.time,
                    dia_mm=args.dia, dpi=args.dpi, out_path=out)
    else:
        run_demo(args)
