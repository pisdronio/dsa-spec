"""
calib_disc_gen.py — Digilog Calibration Disc Generator
=======================================================
Generates a printable calibration disc for DSA v4 optical channel measurements.
See RESEARCH.md §13.4 (zone architecture) and §16 Phase 1 (channel capacity).

Usage:
    python calib_disc_gen.py [--dpi 300] [--dia 304.8] [--out calib_disc.png]

Output files:
    calib_disc.png               — printable disc at target DPI
    calib_disc_legend.txt        — ring index: ring → what it encodes
    calib_measurements_template.csv — blank measurement table
"""

import argparse
import csv
import itertools
import math
import os
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow required: pip install Pillow")

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Digilog calibration disc")
    p.add_argument("--dpi",  type=float, default=300.0,   help="Output DPI (default 300)")
    p.add_argument("--dia",  type=float, default=304.8,   help="Disc diameter in mm (default 304.8 = 12\")")
    p.add_argument("--out",  type=str,   default="calib_disc.png", help="Output PNG path")
    p.add_argument("--module", type=float, default=0.5,   help="Base module size in mm for Zone 1/2/4 (default 0.5)")
    return p.parse_args()

# ── Color palette ─────────────────────────────────────────────────────────────
#
# Must match PALETTE_RGB in dsa_color.py (single source of truth).
# Canonical values: sRGB (IEC 61966-2-1), D65 white point.
# 'M' maps to DSA 'purple' — NOT printer magenta.  The printer's
# RGB→CMYK conversion is handled by the driver; what matters is that
# the same canonical RGB values are used here AND in the encoder.
# For direct ink control, use the _rgb_to_cmyk_pct() helper below.

COLOR_NAMES = ['R', 'G', 'B', 'C', 'M', 'Y', 'K', 'W']
COLORS = {
    'R': (220,  50,  50),   # PALETTE_RGB['red']
    'G': ( 50, 180,  50),   # PALETTE_RGB['green']
    'B': ( 50,  50, 220),   # PALETTE_RGB['blue']
    'C': (  0, 210, 210),   # PALETTE_RGB['cyan']
    'M': (160,  50, 200),   # PALETTE_RGB['purple']
    'Y': (240, 220,   0),   # PALETTE_RGB['yellow']
    'K': (  0,   0,   0),   # PALETTE_RGB['black']
    'W': (255, 255, 255),   # PALETTE_RGB['white']
}

def _rgb_to_cmyk_pct(r: int, g: int, b: int) -> tuple:
    """Standard K-extraction RGB→CMYK.  Returns (C, M, Y, K) percentages 0–100.
    Multiply by 2.55 for PIL CMYK mode (0–255 per channel)."""
    c, m, y = 1 - r / 255.0, 1 - g / 255.0, 1 - b / 255.0
    k = min(c, m, y)
    if k < 1.0:
        s = 1.0 - k
        c, m, y = (c - k) / s, (m - k) / s, (y - k) / s
    else:
        c = m = y = 0.0
    return (round(c * 100), round(m * 100), round(y * 100), round(k * 100))

COLORS_CMYK = {name: _rgb_to_cmyk_pct(*rgb) for name, rgb in COLORS.items()}

# Zone 4 pairs (ordered)
Z4_PAIRS = [
    ('R', 'G'), ('R', 'B'), ('G', 'B'),
    ('R', 'G'),  # R+G+B represented as 33/33/33 blend — approximated as R+G here then overridden
    ('C', 'M'), ('C', 'Y'), ('M', 'Y'),
]
Z4_LABELS = ['R+G', 'R+B', 'G+B', 'R+G+B', 'C+M', 'C+Y', 'M+Y']
Z4_TRIPLES = {
    'R+G':   (('R', 'G'),       (0.5, 0.5)),
    'R+B':   (('R', 'B'),       (0.5, 0.5)),
    'G+B':   (('G', 'B'),       (0.5, 0.5)),
    'R+G+B': (('R', 'G', 'B'),  (1/3, 1/3, 1/3)),
    'C+M':   (('C', 'M'),       (0.5, 0.5)),
    'C+Y':   (('C', 'Y'),       (0.5, 0.5)),
    'M+Y':   (('M', 'Y'),       (0.5, 0.5)),
}

# Zone 2 pairs (6 pairs that get ratio curves)
Z2_PAIRS = [('R', 'G'), ('R', 'B'), ('G', 'B'), ('C', 'M'), ('C', 'Y'), ('M', 'Y')]

# Zone 3 module sizes (mm), 8 densities from coarsest to finest
Z3_MODULE_SIZES_MM = [2.0, 1.6, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]
Z3_PAIR = ('C', 'M')

# Zone 2 ratios: 9 steps 10%–90%
Z2_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ── Geometry helpers ──────────────────────────────────────────────────────────

def mm2px(mm, dpi):
    return mm * dpi / 25.4

def px2mm(px, dpi):
    return px * 25.4 / dpi

# ── Drawing primitives ────────────────────────────────────────────────────────

def _make_annulus_mask(size, cx, cy, r_inner, r_outer):
    """Return an L-mode (grayscale) mask that is 255 inside the annulus."""
    mask = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=255)
    if r_inner > 0:
        d.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=0)
    return mask


def draw_solid_ring(img, cx, cy, r_inner_px, r_outer_px, color):
    """Fill an annulus with a solid color."""
    size = img.size[0]
    layer = Image.new('RGB', (size, size), color)
    mask = _make_annulus_mask(size, cx, cy, r_inner_px, r_outer_px)
    img.paste(layer, mask=mask)


def _build_checker_tile(color_a, color_b, ratio_a, cell_px):
    """
    Build a repeating checkerboard tile.
    ratio_a controls the fraction of cells that show color_a.
    For 50/50 use ratio_a=0.5 (standard 1:1 checkerboard).
    For other ratios, a 10-cell-wide row is used with round(ratio_a*10) cells as A.
    """
    # Compute period: use 10-cell rows to represent ratios in 10% steps
    period = 10
    n_a = max(1, min(period - 1, round(ratio_a * period)))
    n_b = period - n_a
    tile_w = period * cell_px
    tile_h = cell_px * 2  # two rows: row0 = A...B, row1 = B...A (offset for visual break-up)

    tile = Image.new('RGB', (tile_w, tile_h))
    d = ImageDraw.Draw(tile)

    # Row 0: n_a A-cells then n_b B-cells
    for i in range(period):
        c = color_a if i < n_a else color_b
        d.rectangle([i * cell_px, 0, (i + 1) * cell_px - 1, cell_px - 1], fill=c)

    # Row 1: offset by n_b so the boundary falls at a different position
    for i in range(period):
        idx = (i + n_b) % period
        c = color_a if idx < n_a else color_b
        d.rectangle([i * cell_px, cell_px, (i + 1) * cell_px - 1, tile_h - 1], fill=c)

    return tile


def draw_pattern_ring(img, cx, cy, r_inner_px, r_outer_px, color_a, color_b, ratio_a, cell_px):
    """Fill an annulus with a tiled dot pattern."""
    size = img.size[0]
    tile = _build_checker_tile(color_a, color_b, ratio_a, cell_px)
    # Tile onto a full-size layer
    layer = Image.new('RGB', (size, size))
    tw, th = tile.size
    for y in range(0, size, th):
        for x in range(0, size, tw):
            layer.paste(tile, (x, y))
    mask = _make_annulus_mask(size, cx, cy, r_inner_px, r_outer_px)
    img.paste(layer, mask=mask)


def _make_sector_mask(size, cx, cy, r_inner, r_outer, angle_start_deg, angle_end_deg):
    """
    Mask that is 255 inside the annulus sector between two angles.
    Angles are measured from the positive-x axis, counterclockwise.
    PIL arc angles go clockwise from 3 o'clock.
    """
    mask = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(mask)
    # Draw pie slice
    d.pieslice(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        start=angle_start_deg, end=angle_end_deg, fill=255
    )
    # Punch out inner
    if r_inner > 0:
        d.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=0)
    return mask


def draw_sector_ring(img, cx, cy, r_inner_px, r_outer_px, sectors, cell_px):
    """
    Draw a ring divided into sectors with different ratio patterns.
    sectors: list of (color_a, color_b, ratio_a, label)
    """
    size = img.size[0]
    n = len(sectors)
    sector_deg = 360.0 / n
    for i, (color_a, color_b, ratio_a, _label) in enumerate(sectors):
        a_start = i * sector_deg - 90  # start at top (12 o'clock)
        a_end   = a_start + sector_deg

        # Build tiled pattern layer
        tile = _build_checker_tile(color_a, color_b, ratio_a, cell_px)
        layer = Image.new('RGB', (size, size))
        tw, th = tile.size
        for y in range(0, size, th):
            for x in range(0, size, tw):
                layer.paste(tile, (x, y))

        # Build sector annulus mask
        mask = _make_sector_mask(size, cx, cy, r_inner_px, r_outer_px, a_start, a_end)
        img.paste(layer, mask=mask)


def draw_three_color_ring(img, cx, cy, r_inner_px, r_outer_px, colors, fracs, cell_px):
    """Draw a ring with a three-color pattern (e.g. R+G+B at 1/3 each)."""
    size = img.size[0]
    ca, cb, cc = colors
    fa, _fb, _fc = fracs  # assume equal thirds
    # Represent as: period=3, one cell each
    period = 3
    tile_w = period * cell_px
    tile_h = cell_px * 2
    tile = Image.new('RGB', (tile_w, tile_h))
    d = ImageDraw.Draw(tile)
    row0 = [ca, cb, cc]
    row1 = [cb, cc, ca]
    for i, c in enumerate(row0):
        d.rectangle([i * cell_px, 0, (i + 1) * cell_px - 1, cell_px - 1], fill=COLORS[c])
    for i, c in enumerate(row1):
        d.rectangle([i * cell_px, cell_px, (i + 1) * cell_px - 1, tile_h - 1], fill=COLORS[c])

    layer = Image.new('RGB', (size, size))
    tw, th = tile.size
    for y in range(0, size, th):
        for x in range(0, size, tw):
            layer.paste(tile, (x, y))
    mask = _make_annulus_mask(size, cx, cy, r_inner_px, r_outer_px)
    img.paste(layer, mask=mask)


# ── Zone ring allocation ──────────────────────────────────────────────────────

def compute_zones(outer_r_mm, inner_r_mm, gap_mm=0.3):
    """
    Allocate radial bands for each zone from outer to inner.
    Returns a dict: zone_name -> list of (r_outer_mm, r_inner_mm, label, extra)
    """
    # Count rings per zone
    z5_count = 8    # solid colors
    z4_count = 7    # white-achievement pairs
    z3_count = 8    # density matrix rings
    z2_count = 6    # ratio-curve rings
    z1_count = 28   # all unique pairs C(8,2)

    total_rings = z5_count + z4_count + z3_count + z2_count + z1_count  # 57
    total_gaps  = total_rings  # gap after each ring (including innermost)
    # Add 5 zone separator gaps (slightly wider)
    zone_sep = 5
    zone_sep_width_mm = 1.0

    span_mm = outer_r_mm - inner_r_mm
    usable_mm = span_mm - total_gaps * gap_mm - zone_sep * zone_sep_width_mm
    ring_width_mm = usable_mm / total_rings

    rings = []   # list of dicts
    r = outer_r_mm

    def add_zone_separator():
        nonlocal r
        r -= zone_sep_width_mm

    def add_rings(zone, items, color_fn):
        nonlocal r
        for item in items:
            r_out = r
            r_in  = r - ring_width_mm
            rings.append({
                'zone':    zone,
                'r_out':   r_out,
                'r_in':    r_in,
                **color_fn(item),
            })
            r = r_in - gap_mm

    # Zone 5 — solid colors
    def z5_info(name):
        return {'label': f'Z5 solid {name}', 'type': 'solid', 'color': name}
    add_rings('Z5', COLOR_NAMES, z5_info)
    add_zone_separator()

    # Zone 4 — white achievement
    def z4_info(label):
        return {'label': f'Z4 {label}', 'type': 'z4', 'pair_label': label}
    add_rings('Z4', Z4_LABELS, z4_info)
    add_zone_separator()

    # Zone 3 — density matrix
    def z3_info(mod_size):
        return {'label': f'Z3 {Z3_PAIR[0]}+{Z3_PAIR[1]} mod={mod_size:.1f}mm',
                'type': 'z3', 'module_mm': mod_size}
    add_rings('Z3', Z3_MODULE_SIZES_MM, z3_info)
    add_zone_separator()

    # Zone 2 — ratio encoding
    def z2_info(pair):
        return {'label': f'Z2 {pair[0]}+{pair[1]} ratio_sectors',
                'type': 'z2', 'pair': pair}
    add_rings('Z2', Z2_PAIRS, z2_info)
    add_zone_separator()

    # Zone 1 — all unique pairs
    all_pairs = list(itertools.combinations(COLOR_NAMES, 2))  # 28 pairs
    def z1_info(pair):
        return {'label': f'Z1 {pair[0]}+{pair[1]} 50/50',
                'type': 'z1', 'pair': pair}
    add_rings('Z1', all_pairs, z1_info)

    return rings, ring_width_mm


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_disc(args):
    dpi        = args.dpi
    dia_mm     = args.dia
    out_path   = args.out
    base_mod_mm = args.module

    radius_mm  = dia_mm / 2.0
    inner_r_mm = 25.0 if dia_mm >= 280 else 12.0   # LP vs smaller disc label area
    outer_r_mm = radius_mm - 3.0                    # 3mm edge margin

    gap_mm = 0.3

    img_px   = int(mm2px(dia_mm, dpi))
    cx = cy  = img_px // 2

    print(f"Canvas: {img_px}×{img_px} px  ({dia_mm}mm @ {dpi} dpi)")
    print(f"Data band: {inner_r_mm:.1f}mm → {outer_r_mm:.1f}mm  "
          f"(span {outer_r_mm - inner_r_mm:.1f}mm)")

    rings, ring_width_mm = compute_zones(outer_r_mm, inner_r_mm, gap_mm)
    print(f"Ring width: {ring_width_mm:.3f}mm = {mm2px(ring_width_mm, dpi):.1f}px")
    print(f"Total rings: {len(rings)}")

    # Warn if rings are too thin to print reliably
    ring_px = mm2px(ring_width_mm, dpi)
    if ring_px < 8:
        print(f"WARNING: ring width {ring_px:.1f}px is below 8px — consider larger disc or lower DPI")

    # Background: light grey
    img = Image.new('RGB', (img_px, img_px), (220, 220, 220))

    # Draw rings from outer to inner
    cell_base_px = max(2, int(mm2px(base_mod_mm, dpi)))

    for ring in rings:
        r_out_px = mm2px(ring['r_out'], dpi)
        r_in_px  = mm2px(ring['r_in'],  dpi)
        rtype    = ring['type']

        if rtype == 'solid':
            draw_solid_ring(img, cx, cy, r_in_px, r_out_px, COLORS[ring['color']])

        elif rtype == 'z4':
            lbl = ring['pair_label']
            info = Z4_TRIPLES[lbl]
            pair_colors, fracs = info
            if len(pair_colors) == 3:
                draw_three_color_ring(img, cx, cy, r_in_px, r_out_px,
                                      pair_colors, fracs, cell_base_px)
            else:
                ca, cb = pair_colors
                draw_pattern_ring(img, cx, cy, r_in_px, r_out_px,
                                   COLORS[ca], COLORS[cb], 0.5, cell_base_px)

        elif rtype == 'z3':
            mod_px = max(2, int(mm2px(ring['module_mm'], dpi)))
            ca, cb = Z3_PAIR
            draw_pattern_ring(img, cx, cy, r_in_px, r_out_px,
                               COLORS[ca], COLORS[cb], 0.5, mod_px)

        elif rtype == 'z2':
            ca, cb = ring['pair']
            sectors = [(COLORS[ca], COLORS[cb], r, f"{int(r*100)}%") for r in Z2_RATIOS]
            draw_sector_ring(img, cx, cy, r_in_px, r_out_px, sectors, cell_base_px)

        elif rtype == 'z1':
            ca, cb = ring['pair']
            draw_pattern_ring(img, cx, cy, r_in_px, r_out_px,
                               COLORS[ca], COLORS[cb], 0.5, cell_base_px)

    # Centre hole
    hole_r_px = mm2px(inner_r_mm * 0.35, dpi)
    draw_solid_ring(img, cx, cy, 0, hole_r_px, (180, 180, 180))

    # Disc outer edge circle
    d = ImageDraw.Draw(img)
    d.ellipse([cx - mm2px(radius_mm, dpi), cy - mm2px(radius_mm, dpi),
               cx + mm2px(radius_mm, dpi), cy + mm2px(radius_mm, dpi)],
              outline=(0, 0, 0), width=3)

    img.save(out_path, dpi=(dpi, dpi))
    print(f"Saved: {out_path}")
    return rings, ring_width_mm


# ── Legend file ───────────────────────────────────────────────────────────────

def write_legend(rings, ring_width_mm, path):
    with open(path, 'w') as f:
        f.write("Digilog Calibration Disc — Ring Index\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ring width: {ring_width_mm:.3f} mm\n\n")
        current_zone = None
        for i, ring in enumerate(rings):
            if ring['zone'] != current_zone:
                current_zone = ring['zone']
                zone_desc = {
                    'Z5': 'Reference anchors (solid colors) — normalize white balance',
                    'Z4': 'White achievement survey — 50/50 pair blending',
                    'Z3': 'Dot density / speed threshold matrix — C+M, 8 module sizes',
                    'Z2': 'Ratio encoding curve — 9 sectors × 6 pairs',
                    'Z1': 'Color pair discriminability — all 28 pairs at 50/50',
                }
                f.write(f"\n── {current_zone}: {zone_desc[current_zone]} ──\n")
            f.write(f"  Ring {i+1:3d}: {ring['r_out']:.2f}mm → {ring['r_in']:.2f}mm  |  {ring['label']}\n")
            if ring['type'] == 'z2':
                for j, r in enumerate(Z2_RATIOS):
                    angle_start = j * (360 / 9) - 90
                    angle_end   = angle_start + 360 / 9
                    f.write(f"             Sector {j+1}: {angle_start:.1f}°–{angle_end:.1f}°  ratio={int(r*100)}% {ring['pair'][0]}\n")
    print(f"Saved legend: {path}")


# ── Measurement template CSV ──────────────────────────────────────────────────

def write_measurement_template(rings, path):
    rows = []

    for i, ring in enumerate(rings):
        rtype = ring['type']
        base = {
            'zone':        ring['zone'],
            'ring_number': i + 1,
            'ring_label':  ring['label'],
            'r_out_mm':    f"{ring['r_out']:.2f}",
            'r_in_mm':     f"{ring['r_in']:.2f}",
        }

        if rtype == 'z2':
            # One row per sector × speed
            for j, ratio in enumerate(Z2_RATIOS):
                for speed in [0, 33, 45]:
                    rows.append({**base,
                        'sector':    j + 1,
                        'ratio_pct': int(ratio * 100),
                        'speed_rpm': speed,
                        'R': '', 'G': '', 'B': '',
                        'notes': ''})
        elif rtype == 'z3':
            for speed in [0, 33, 45]:
                rows.append({**base,
                    'sector':    '',
                    'ratio_pct': 50,
                    'module_mm': ring['module_mm'],
                    'speed_rpm': speed,
                    'R': '', 'G': '', 'B': '',
                    'notes': ''})
        else:
            for speed in ([0] if rtype in ('z5', 'solid') else [0, 33, 45]):
                rows.append({**base,
                    'sector':    '',
                    'ratio_pct': '',
                    'module_mm': '',
                    'speed_rpm': speed,
                    'R': '', 'G': '', 'B': '',
                    'notes': ''})

    fieldnames = ['zone', 'ring_number', 'ring_label', 'r_out_mm', 'r_in_mm',
                  'sector', 'ratio_pct', 'module_mm', 'speed_rpm', 'R', 'G', 'B', 'notes']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"Saved measurement template: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    out_dir = os.path.dirname(args.out) or '.'
    legend_path = os.path.join(out_dir, 'calib_disc_legend.txt')
    template_path = os.path.join(out_dir, 'calib_measurements_template.csv')

    rings, ring_width_mm = render_disc(args)
    write_legend(rings, ring_width_mm, legend_path)
    write_measurement_template(rings, template_path)

    print("\nDone. Next steps:")
    print("  1. Print calib_disc.png at exactly the specified DPI on reference substrate")
    print("  2. Mount on turntable, position Rig camera at 15cm height")
    print("  3. Photograph at 0rpm, 33rpm, 45rpm")
    print("  4. Extract ring RGB values and fill calib_measurements_template.csv")
    print("  5. Compute W and SNR from Zone 3 and Zone 5 data (see RESEARCH.md §16)")
