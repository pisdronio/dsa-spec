"""
calib_extract.py — Digilog Calibration Disc Measurement Extractor
==================================================================
Takes a photograph of a printed calibration disc and extracts mean RGB
per ring (and per sector for Zone 2), then normalizes against Zone 5
reference anchors and appends rows to a measurements CSV.

Usage:
    python3 calib_extract.py PHOTO [options]

    PHOTO               path to disc photograph (JPEG, PNG, TIFF)

    --legend  PATH      calib_disc_legend.txt from the generator
                        (default: calib_disc_legend.txt in same dir as PHOTO)
    --csv     PATH      output CSV path
                        (default: calib_measurements_<speed>rpm.csv)
    --speed   RPM       disc speed when photographed: 0, 33, or 45
                        (default: 0)
    --cx      PX        disc centre x in pixels (auto-detect if omitted)
    --cy      PX        disc centre y in pixels (auto-detect if omitted)
    --scale   PX/MM     pixels per mm (auto-detect from disc edge if omitted)
    --margin  FRAC      fraction of ring width to exclude at inner/outer edge
                        (default: 0.15, i.e. exclude 15% each side)
    --debug             save debug overlay image showing sampled regions

Auto-detection:
    If --cx/--cy/--scale are omitted the script finds the disc edge via
    Hough circle detection on the Canny-edge image.  Works well for photos
    taken straight-on against a contrasting background.  If detection fails,
    provide the values manually.

Normalization:
    After sampling all rings, Zone 5 solid-color rings are used to build a
    per-channel correction factor:
        correction[c] = expected[c] / measured[c]   (for each RGB channel c)
    All other ring measurements are multiplied by this correction before
    writing to CSV.  expected values are the generator's COLORS dict values.

Output CSV columns:
    photo, zone, ring_number, ring_label, r_out_mm, r_in_mm,
    sector, ratio_pct, module_mm, speed_rpm,
    R_raw, G_raw, B_raw, R_norm, G_norm, B_norm, notes
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required: pip install numpy")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow required: pip install Pillow")

# ── Expected Zone 5 reference colors (must match calib_disc_gen.py COLORS) ──

Z5_EXPECTED = {
    'R': (255,   0,   0),
    'G': (  0, 200,   0),
    'B': (  0,   0, 255),
    'C': (  0, 200, 255),
    'M': (255,   0, 200),
    'Y': (255, 220,   0),
    'K': (  0,   0,   0),
    'W': (255, 255, 255),
}
COLOR_ORDER = ['R', 'G', 'B', 'C', 'M', 'Y', 'K', 'W']

Z2_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ── Legend parser ─────────────────────────────────────────────────────────────

def parse_legend(legend_path):
    """
    Parse calib_disc_legend.txt into a list of ring dicts.
    Each dict: {ring_number, zone, label, r_out_mm, r_in_mm, type, sectors?}
    """
    rings = []
    current_ring = None
    ring_re  = re.compile(
        r'Ring\s+(\d+):\s+([\d.]+)mm\s+[→-]+>?\s*([\d.]+)mm\s+\|\s+(.+)')
    sector_re = re.compile(
        r'Sector\s+(\d+):\s+([-\d.]+)°.*ratio=(\d+)%')

    with open(legend_path) as f:
        for line in f:
            m = ring_re.search(line)
            if m:
                current_ring = {
                    'ring_number': int(m.group(1)),
                    'r_out_mm':    float(m.group(2)),
                    'r_in_mm':     float(m.group(3)),
                    'label':       m.group(4).strip(),
                    'zone':        m.group(4).strip().split()[0],
                    'sectors':     [],
                }
                rings.append(current_ring)
                continue
            m = sector_re.search(line)
            if m and current_ring is not None:
                current_ring['sectors'].append({
                    'number':    int(m.group(1)),
                    'angle_deg': float(m.group(2)),  # start angle
                    'ratio_pct': int(m.group(3)),
                })

    for ring in rings:
        lbl = ring['label']
        if lbl.startswith('Z5'):
            ring['type'] = 'z5'
            ring['color_name'] = lbl.split()[-1]
        elif lbl.startswith('Z4'):
            ring['type'] = 'z4'
        elif lbl.startswith('Z3'):
            ring['type'] = 'z3'
            m = re.search(r'mod=([\d.]+)mm', lbl)
            ring['module_mm'] = float(m.group(1)) if m else None
        elif lbl.startswith('Z2'):
            ring['type'] = 'z2'
        else:
            ring['type'] = 'z1'

    return rings


# ── Disc geometry detection ───────────────────────────────────────────────────

def detect_disc_geometry(img_arr):
    """
    Detect disc centre (cx, cy) and scale (px/mm) from the disc outer edge.
    Uses a simple approach: find the largest circle-like feature by thresholding
    and fitting to the outer boundary points.
    Returns (cx, cy, scale_px_per_mm) or raises RuntimeError.
    """
    try:
        from skimage.transform import hough_circle, hough_circle_peaks
        from skimage.feature import canny
        from skimage.color import rgb2gray
    except ImportError:
        raise RuntimeError(
            "scikit-image required for auto-detection: pip install scikit-image\n"
            "Or provide --cx, --cy, --scale manually."
        )

    gray = rgb2gray(img_arr)
    edges = canny(gray, sigma=2.0)

    h, w = gray.shape
    # Expect disc to occupy 60–95% of the image
    r_min = int(min(h, w) * 0.30)
    r_max = int(min(h, w) * 0.50)
    radii = np.arange(r_min, r_max, 20)

    hspaces = hough_circle(edges, radii)
    accums, cx_arr, cy_arr, rad_arr = hough_circle_peaks(
        hspaces, radii, num_peaks=1)

    if len(cx_arr) == 0:
        raise RuntimeError("Hough circle detection found no disc — provide --cx/--cy/--scale")

    cx, cy, r_px = float(cx_arr[0]), float(cy_arr[0]), float(rad_arr[0])
    # r_px corresponds to the outer usable radius; disc physical radius is dia/2
    # We'll use 12" (304.8mm) disc, outer data radius = 149.4mm (from generator)
    outer_data_r_mm = 149.4
    scale = r_px / outer_data_r_mm
    print(f"Auto-detected: cx={cx:.1f} cy={cy:.1f} r={r_px:.1f}px  scale={scale:.3f}px/mm")
    return cx, cy, scale


# ── Ring/sector pixel sampling ────────────────────────────────────────────────

def sample_annulus(img_arr, cx, cy, r_inner_px, r_outer_px, margin_frac=0.15):
    """
    Sample all pixels in the annulus between r_inner_px and r_outer_px
    (with inset margin on each side) and return mean (R, G, B).
    """
    h, w = img_arr.shape[:2]
    ring_width = r_outer_px - r_inner_px
    r_inner_px = r_inner_px + ring_width * margin_frac
    r_outer_px = r_outer_px - ring_width * margin_frac

    if r_outer_px <= r_inner_px or r_outer_px <= 0:
        return None

    # Bounding box to avoid iterating the whole image
    x0 = max(0, int(cx - r_outer_px) - 1)
    x1 = min(w, int(cx + r_outer_px) + 2)
    y0 = max(0, int(cy - r_outer_px) - 1)
    y1 = min(h, int(cy + r_outer_px) + 2)

    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 >= r_inner_px ** 2) & (dist2 <= r_outer_px ** 2)

    pixels = img_arr[y0:y1, x0:x1][mask]
    if len(pixels) == 0:
        return None
    mean = pixels.mean(axis=0)
    return tuple(float(v) for v in mean[:3])


def sample_sector(img_arr, cx, cy, r_inner_px, r_outer_px,
                  angle_start_deg, angle_end_deg, margin_frac=0.15):
    """
    Sample pixels in an annular sector between two angles.
    Angles follow PIL convention (clockwise from 3 o'clock = east).
    """
    h, w = img_arr.shape[:2]
    ring_width = r_outer_px - r_inner_px
    r_inner_px = r_inner_px + ring_width * margin_frac
    r_outer_px = r_outer_px - ring_width * margin_frac

    if r_outer_px <= r_inner_px or r_outer_px <= 0:
        return None

    x0 = max(0, int(cx - r_outer_px) - 1)
    x1 = min(w, int(cx + r_outer_px) + 2)
    y0 = max(0, int(cy - r_outer_px) - 1)
    y1 = min(h, int(cy + r_outer_px) + 2)

    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

    # Angle of each pixel in PIL convention (clockwise from east, degrees)
    # math.atan2 uses standard convention (CCW from east)
    # PIL angle: clockwise from east → negate the y
    angle_px = np.degrees(np.arctan2(-(yy - cy), xx - cx))  # -180..180, CCW from east

    # Normalise angles to compare with PIL-convention start/end
    # PIL: angle_start_deg measured clockwise → convert to math convention
    # math_angle = -PIL_angle   (for comparison)
    # So: pixel is in sector if -angle_end_deg <= angle_px <= -angle_start_deg
    # But we need to handle wraparound.

    # Simpler: work in PIL angle space (clockwise from east)
    angle_cw = (-angle_px) % 360  # 0..360, clockwise from east

    a0 = angle_start_deg % 360
    a1 = angle_end_deg   % 360

    if a0 <= a1:
        sector_mask = (angle_cw >= a0) & (angle_cw < a1)
    else:
        sector_mask = (angle_cw >= a0) | (angle_cw < a1)

    ring_mask = (dist2 >= r_inner_px ** 2) & (dist2 <= r_outer_px ** 2)
    mask = ring_mask & sector_mask

    pixels = img_arr[y0:y1, x0:x1][mask]
    if len(pixels) == 0:
        return None
    mean = pixels.mean(axis=0)
    return tuple(float(v) for v in mean[:3])


# ── Zone 5 normalization ──────────────────────────────────────────────────────

def build_correction(z5_measurements):
    """
    Build per-channel correction factors from Zone 5 measurements.
    z5_measurements: dict color_name -> (R_meas, G_meas, B_meas)
    Returns correction: dict color_name -> (cr, cg, cb)
    """
    corrections = {}
    for name, meas in z5_measurements.items():
        exp = Z5_EXPECTED[name]
        # Avoid division by zero for black (K) channel
        cr = exp[0] / meas[0] if meas[0] > 1 else 1.0
        cg = exp[1] / meas[1] if meas[1] > 1 else 1.0
        cb = exp[2] / meas[2] if meas[2] > 1 else 1.0
        corrections[name] = (cr, cg, cb)
    return corrections


def apply_correction(rgb, corrections):
    """
    Apply white-balance correction to a measured RGB tuple.
    Uses the W (white) correction as the global illuminant correction.
    """
    if 'W' not in corrections:
        return rgb
    cr, cg, cb = corrections['W']
    r = min(255.0, rgb[0] * cr)
    g = min(255.0, rgb[1] * cg)
    b = min(255.0, rgb[2] * cb)
    return (r, g, b)


# ── Debug overlay ─────────────────────────────────────────────────────────────

def save_debug_overlay(img_arr, cx, cy, scale, rings, out_path):
    from PIL import Image as PILImage, ImageDraw as PILDraw
    overlay = PILImage.fromarray(img_arr.astype('uint8'))
    d = PILDraw.Draw(overlay, 'RGBA')

    for ring in rings:
        r_out = ring['r_out_mm'] * scale
        r_in  = ring['r_in_mm']  * scale
        # Draw ring boundaries
        for r in (r_out, r_in):
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      outline=(255, 255, 0, 180), width=1)

    overlay.save(out_path)
    print(f"Debug overlay saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract ring RGB from calibration disc photo")
    p.add_argument('photo',            help='Disc photograph path')
    p.add_argument('--legend',         help='calib_disc_legend.txt path')
    p.add_argument('--csv',            help='Output CSV path')
    p.add_argument('--speed', type=float, default=0, help='Disc speed (rpm)')
    p.add_argument('--cx',    type=float, help='Disc centre x (px)')
    p.add_argument('--cy',    type=float, help='Disc centre y (px)')
    p.add_argument('--scale', type=float, help='Pixels per mm')
    p.add_argument('--margin', type=float, default=0.15, help='Edge margin fraction')
    p.add_argument('--debug', action='store_true', help='Save debug overlay')
    return p.parse_args()


def main():
    args = parse_args()

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    # Resolve legend path
    legend_path = args.legend or (photo_path.parent / 'calib_disc_legend.txt')
    if not Path(legend_path).exists():
        sys.exit(f"Legend not found: {legend_path}\nRun calib_disc_gen.py first or pass --legend")

    # Resolve output CSV path
    speed_tag = f"{int(args.speed)}rpm"
    csv_path = args.csv or str(photo_path.parent / f"calib_measurements_{speed_tag}.csv")

    print(f"Photo:   {photo_path}")
    print(f"Legend:  {legend_path}")
    print(f"Output:  {csv_path}")
    print(f"Speed:   {args.speed} rpm")

    # Load image
    img = Image.open(photo_path).convert('RGB')
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]
    print(f"Image:   {w}×{h} px")

    # Geometry
    if args.cx and args.cy and args.scale:
        cx, cy, scale = args.cx, args.cy, args.scale
        print(f"Geometry (manual): cx={cx} cy={cy} scale={scale:.3f} px/mm")
    else:
        print("Auto-detecting disc geometry...")
        try:
            cx, cy, scale = detect_disc_geometry(img_arr)
        except RuntimeError as e:
            sys.exit(str(e))

    # Parse legend
    rings = parse_legend(legend_path)
    print(f"Rings:   {len(rings)}")

    # Debug overlay
    if args.debug:
        debug_path = str(photo_path.parent / f"calib_debug_{speed_tag}.png")
        save_debug_overlay(img_arr, cx, cy, scale, rings, debug_path)

    # Sample all rings
    print("Sampling rings...", end='', flush=True)

    z5_measurements = {}
    rows = []

    for ring in rings:
        r_out_px = ring['r_out_mm'] * scale
        r_in_px  = ring['r_in_mm']  * scale

        base_row = {
            'photo':       photo_path.name,
            'zone':        ring['zone'],
            'ring_number': ring['ring_number'],
            'ring_label':  ring['label'],
            'r_out_mm':    f"{ring['r_out_mm']:.2f}",
            'r_in_mm':     f"{ring['r_in_mm']:.2f}",
            'speed_rpm':   args.speed,
        }

        rtype = ring['type']

        if rtype == 'z2' and ring['sectors']:
            # Sample each sector separately
            for sec in ring['sectors']:
                a_start = sec['angle_deg']
                a_end   = a_start + 360.0 / len(ring['sectors'])
                rgb = sample_sector(img_arr, cx, cy, r_in_px, r_out_px,
                                    a_start, a_end, args.margin)
                rows.append({**base_row,
                    'sector':    sec['number'],
                    'ratio_pct': sec['ratio_pct'],
                    'module_mm': '',
                    'R_raw': f"{rgb[0]:.1f}" if rgb else '',
                    'G_raw': f"{rgb[1]:.1f}" if rgb else '',
                    'B_raw': f"{rgb[2]:.1f}" if rgb else '',
                    'R_norm': '', 'G_norm': '', 'B_norm': '',
                    'notes': 'no pixels sampled' if rgb is None else '',
                })
        else:
            rgb = sample_annulus(img_arr, cx, cy, r_in_px, r_out_px, args.margin)

            row = {**base_row,
                'sector':    '',
                'ratio_pct': '',
                'module_mm': ring.get('module_mm', ''),
                'R_raw': f"{rgb[0]:.1f}" if rgb else '',
                'G_raw': f"{rgb[1]:.1f}" if rgb else '',
                'B_raw': f"{rgb[2]:.1f}" if rgb else '',
                'R_norm': '', 'G_norm': '', 'B_norm': '',
                'notes': 'no pixels sampled' if rgb is None else '',
            }
            rows.append(row)

            if rtype == 'z5' and rgb:
                z5_measurements[ring['color_name']] = rgb

        print('.', end='', flush=True)

    print()

    # Build white-balance correction from Zone 5
    if z5_measurements:
        corrections = build_correction(z5_measurements)
        print(f"Zone 5 normalization built from {len(z5_measurements)} anchors")
        if 'W' in corrections:
            cr, cg, cb = corrections['W']
            print(f"  White correction: R×{cr:.3f}  G×{cg:.3f}  B×{cb:.3f}")
    else:
        corrections = {}
        print("WARNING: No Zone 5 measurements — normalization skipped")

    # Apply normalization
    for row in rows:
        if row['R_raw'] and corrections:
            rgb_raw = (float(row['R_raw']), float(row['G_raw']), float(row['B_raw']))
            rgb_norm = apply_correction(rgb_raw, corrections)
            row['R_norm'] = f"{rgb_norm[0]:.1f}"
            row['G_norm'] = f"{rgb_norm[1]:.1f}"
            row['B_norm'] = f"{rgb_norm[2]:.1f}"

    # Write CSV
    fieldnames = ['photo', 'zone', 'ring_number', 'ring_label', 'r_out_mm', 'r_in_mm',
                  'sector', 'ratio_pct', 'module_mm', 'speed_rpm',
                  'R_raw', 'G_raw', 'B_raw', 'R_norm', 'G_norm', 'B_norm', 'notes']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"Written {len(rows)} rows → {csv_path}")

    # Print Zone 5 summary for quick sanity check
    if z5_measurements:
        print("\nZone 5 anchor readings (raw):")
        for name in COLOR_ORDER:
            if name in z5_measurements:
                meas = z5_measurements[name]
                exp  = Z5_EXPECTED[name]
                print(f"  {name}: measured ({meas[0]:.0f},{meas[1]:.0f},{meas[2]:.0f})  "
                      f"expected {exp}")


if __name__ == '__main__':
    main()
