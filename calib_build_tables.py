"""
calib_build_tables.py — Digilog Calibration Table Builder
==========================================================
Reads one or more filled measurement CSVs (output of calib_extract.py) and
produces calibration_tables.json containing:

  1. ratio_curves[pair][speed] — Zone 2 data
        list of {ratio_pct, R, G, B} sorted by ratio_pct
        maps print ratio → expected read RGB at that speed

  2. density_threshold[speed] — Zone 3 data
        smallest module_mm whose RGB distance from fully-integrated
        reference is below INTEGRATION_THRESHOLD
        i.e. the minimum printable dot density for clean blending

  3. pair_discriminability — Zone 1 data
        list of {pair, delta_E, readable} sorted by delta_E descending
        delta_E: perceptual distance between the blended 50/50 read value
                 and the background (Zone 5 W anchor)

  4. white_achievement[label] — Zone 4 data
        {R, G, B, delta_E_from_white} for each pair label

  5. z5_anchors[speed] — Zone 5 reference readings (normalized)
        {color: [R, G, B]} — should be ≈ expected values after normalization;
        residual deviation reveals print/camera non-ideality

  6. encoder_lut[pair][speed] — derived inverse of ratio_curves
        maps read_value (scalar) → print_ratio (0.0–1.0) via piecewise
        linear interpolation on the luminance channel

Usage:
    python3 calib_build_tables.py CSV [CSV ...] --out calibration_tables.json

    Multiple CSVs (one per speed) are merged automatically.
    If only one CSV exists (static measurements), dynamic fields are omitted.

Output:
    calibration_tables.json   — machine-readable tables for the v4 encoder
    calibration_tables_report.txt — human-readable summary with quality flags
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

INTEGRATION_THRESHOLD = 15.0   # max RGB Euclidean distance to count as "integrated"
MIN_DISCRIMINABILITY  = 20.0   # min delta_E to consider a pair usable

Z2_RATIOS_PCT = [10, 20, 30, 40, 50, 60, 70, 80, 90]
Z3_MODULE_SIZES_MM = [2.0, 1.6, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]
Z3_PAIR = ('C', 'M')

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

# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            # Skip blank measurement rows (R/G/B empty)
            if not row.get('R_norm') and not row.get('R_raw'):
                continue
            # Prefer normalized values; fall back to raw
            r_key = 'R_norm' if row.get('R_norm') else 'R_raw'
            g_key = 'G_norm' if row.get('G_norm') else 'G_raw'
            b_key = 'B_norm' if row.get('B_norm') else 'B_raw'
            try:
                row['_R'] = float(row[r_key])
                row['_G'] = float(row[g_key])
                row['_B'] = float(row[b_key])
                row['_speed'] = float(row['speed_rpm'])
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def load_all(paths):
    rows = []
    for p in paths:
        loaded = load_csv(p)
        print(f"  {p}: {len(loaded)} rows")
        rows.extend(loaded)
    return rows

# ── Color math ────────────────────────────────────────────────────────────────

def rgb_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def luminance(r, g, b):
    """Approximate perceptual luminance (BT.601)."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def delta_e_simple(rgb1, rgb2):
    """
    Fast perceptual distance in RGB space (not true CIE delta-E but sufficient
    for relative ranking).  Uses weighted Euclidean distance.
    """
    dr = (rgb1[0] - rgb2[0]) * 0.299
    dg = (rgb1[1] - rgb2[1]) * 0.587
    db = (rgb1[2] - rgb2[2]) * 0.114
    return math.sqrt(dr ** 2 + dg ** 2 + db ** 2)

# ── Zone extractors ───────────────────────────────────────────────────────────

def extract_z5(rows):
    """
    Returns z5[speed][color_name] = (R, G, B)
    """
    z5 = defaultdict(dict)
    for row in rows:
        if not row['zone'].startswith('Z5'):
            continue
        lbl = row['ring_label']          # e.g. "Z5 solid R"
        parts = lbl.split()
        if len(parts) < 3:
            continue
        color = parts[-1]
        speed = row['_speed']
        z5[speed][color] = (row['_R'], row['_G'], row['_B'])
    return dict(z5)


def extract_z4(rows):
    """
    Returns z4[speed][pair_label] = (R, G, B)
    """
    z4 = defaultdict(dict)
    for row in rows:
        if not row['zone'].startswith('Z4'):
            continue
        # label: "Z4 R+G" or "Z4 R+G+B"
        label = row['ring_label'].replace('Z4 ', '').strip()
        speed = row['_speed']
        z4[speed][label] = (row['_R'], row['_G'], row['_B'])
    return dict(z4)


def extract_z3(rows):
    """
    Returns z3[speed][module_mm] = (R, G, B)
    """
    z3 = defaultdict(dict)
    for row in rows:
        if not row['zone'].startswith('Z3'):
            continue
        mod = row.get('module_mm')
        if not mod:
            continue
        try:
            mod_f = float(mod)
        except ValueError:
            continue
        speed = row['_speed']
        z3[speed][mod_f] = (row['_R'], row['_G'], row['_B'])
    return dict(z3)


def extract_z2(rows):
    """
    Returns z2[speed][pair_label][ratio_pct] = (R, G, B)
    pair_label extracted from ring_label, e.g. "Z2 R+G ratio_sectors" → "R+G"
    """
    z2 = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if not row['zone'].startswith('Z2'):
            continue
        lbl = row['ring_label']
        # Extract pair name
        parts = lbl.replace('Z2 ', '').split()
        pair = parts[0] if parts else '?'
        ratio_pct = row.get('ratio_pct')
        if not ratio_pct:
            continue
        try:
            ratio_i = int(ratio_pct)
        except ValueError:
            continue
        speed = row['_speed']
        z2[speed][pair][ratio_i] = (row['_R'], row['_G'], row['_B'])
    # Convert nested defaultdicts
    return {sp: {pair: dict(ratios) for pair, ratios in pairs.items()}
            for sp, pairs in z2.items()}


def extract_z1(rows):
    """
    Returns z1[speed][pair_label] = (R, G, B)
    """
    z1 = defaultdict(dict)
    for row in rows:
        if not row['zone'].startswith('Z1'):
            continue
        lbl = row['ring_label'].replace('Z1 ', '').replace(' 50/50', '').strip()
        speed = row['_speed']
        z1[speed][lbl] = (row['_R'], row['_G'], row['_B'])
    return dict(z1)

# ── Table builders ────────────────────────────────────────────────────────────

def build_ratio_curves(z2_data):
    """
    ratio_curves[pair][speed] = sorted list of {ratio_pct, R, G, B}
    """
    curves = defaultdict(dict)
    for speed, pairs in z2_data.items():
        for pair, ratios in pairs.items():
            curves[pair][speed] = sorted(
                [{'ratio_pct': r, 'R': v[0], 'G': v[1], 'B': v[2]}
                 for r, v in ratios.items()],
                key=lambda x: x['ratio_pct']
            )
    return dict(curves)


def build_density_threshold(z3_data, z5_data):
    """
    density_threshold[speed] = {module_mm, R, G, B, dist_from_reference}

    The reference for "fully integrated" is the Zone 5 50/50 blend — but since
    we don't measure that directly, use the mean of the two Z5 solid anchor
    colors for Z3_PAIR (C and M).
    """
    thresholds = {}
    for speed, modules in z3_data.items():
        # Derive reference integrated color for C+M at this speed
        z5_at_speed = z5_data.get(speed, z5_data.get(0.0, {}))
        c_ref = z5_at_speed.get('C', (0, 200, 255))
        m_ref = z5_at_speed.get('M', (255, 0, 200))
        integrated_ref = tuple((c_ref[i] + m_ref[i]) / 2 for i in range(3))

        # Find finest (smallest) module that is within threshold of integrated_ref
        # Sort ascending = finest first (0.3mm is hardest to resolve)
        best = None
        for mod in sorted(modules.keys()):   # finest → coarsest
            rgb = modules[mod]
            dist = rgb_dist(rgb, integrated_ref)
            if dist <= INTEGRATION_THRESHOLD:
                best = {'module_mm': mod, 'R': rgb[0], 'G': rgb[1], 'B': rgb[2],
                        'dist_from_reference': round(dist, 2)}
                break  # stop at finest passing module

        if best:
            thresholds[speed] = best
        else:
            # Nothing integrates — record the coarsest reading with a flag
            coarsest = sorted(modules.keys())[0]
            rgb = modules[coarsest]
            thresholds[speed] = {
                'module_mm': coarsest,
                'R': rgb[0], 'G': rgb[1], 'B': rgb[2],
                'dist_from_reference': round(rgb_dist(rgb, integrated_ref), 2),
                'warning': 'no module size achieved integration threshold'
            }

    return thresholds


def build_pair_discriminability(z1_data, z5_data):
    """
    pair_discriminability = list of {pair, delta_E, readable}
    sorted descending by delta_E.
    Uses 0rpm (static) data.  If unavailable, uses first available speed.
    """
    speed = 0.0 if 0.0 in z1_data else (sorted(z1_data.keys())[0] if z1_data else None)
    if speed is None:
        return []

    z5_at_speed = z5_data.get(speed, z5_data.get(0.0, {}))
    white = z5_at_speed.get('W', (255, 255, 255))

    results = []
    for pair, rgb in z1_data[speed].items():
        de = delta_e_simple(rgb, white)
        results.append({
            'pair': pair,
            'R': round(rgb[0], 1), 'G': round(rgb[1], 1), 'B': round(rgb[2], 1),
            'delta_E': round(de, 2),
            'readable': de >= MIN_DISCRIMINABILITY,
        })

    results.sort(key=lambda x: x['delta_E'], reverse=True)
    return results


def build_white_achievement(z4_data, z5_data):
    """
    white_achievement[label] = {R, G, B, delta_E_from_white}
    Uses 0rpm data.
    """
    speed = 0.0 if 0.0 in z4_data else (sorted(z4_data.keys())[0] if z4_data else None)
    if speed is None:
        return {}

    z5_at_speed = z5_data.get(speed, z5_data.get(0.0, {}))
    white = z5_at_speed.get('W', (255, 255, 255))

    out = {}
    for label, rgb in z4_data[speed].items():
        de = delta_e_simple(rgb, white)
        out[label] = {
            'R': round(rgb[0], 1), 'G': round(rgb[1], 1), 'B': round(rgb[2], 1),
            'delta_E_from_white': round(de, 2),
        }
    return out


def build_encoder_lut(ratio_curves):
    """
    encoder_lut[pair][speed] = list of {luminance, ratio_pct} sorted by luminance.
    The encoder interpolates: given a read luminance, find the corresponding print ratio.
    """
    lut = defaultdict(dict)
    for pair, speeds in ratio_curves.items():
        for speed, curve in speeds.items():
            lut_points = []
            for pt in curve:
                lum = luminance(pt['R'], pt['G'], pt['B'])
                lut_points.append({'luminance': round(lum, 2), 'ratio_pct': pt['ratio_pct']})
            lut_points.sort(key=lambda x: x['luminance'])
            lut[pair][speed] = lut_points
    return dict(lut)


def encoder_lookup(lut_points, target_luminance):
    """
    Piecewise linear interpolation: luminance → ratio_pct.
    Returns ratio_pct as a float 0–100.
    """
    if not lut_points:
        return 50.0
    if target_luminance <= lut_points[0]['luminance']:
        return float(lut_points[0]['ratio_pct'])
    if target_luminance >= lut_points[-1]['luminance']:
        return float(lut_points[-1]['ratio_pct'])
    for i in range(len(lut_points) - 1):
        lo, hi = lut_points[i], lut_points[i + 1]
        if lo['luminance'] <= target_luminance <= hi['luminance']:
            t = ((target_luminance - lo['luminance']) /
                 (hi['luminance'] - lo['luminance'] + 1e-9))
            return lo['ratio_pct'] + t * (hi['ratio_pct'] - lo['ratio_pct'])
    return 50.0

# ── Report ────────────────────────────────────────────────────────────────────

def write_report(tables, path):
    with open(path, 'w') as f:
        def p(*args): print(*args, file=f)

        p("Digilog Calibration Tables — Quality Report")
        p("=" * 60)
        p()

        # Z5 anchors
        p("── Zone 5: Reference anchor residuals ──")
        for speed, anchors in sorted(tables.get('z5_anchors', {}).items()):
            p(f"  Speed {speed} rpm:")
            for color, rgb in sorted(anchors.items()):
                exp = Z5_EXPECTED.get(color, (0, 0, 0))
                dist = rgb_dist(rgb, exp)
                flag = '  ✓' if dist < 5 else '  ⚠ RESIDUAL HIGH'
                p(f"    {color}: ({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})  "
                  f"expected {exp}  dist={dist:.1f}{flag}")
        p()

        # Density thresholds
        p("── Zone 3: Dot density / integration thresholds ──")
        for speed, thr in sorted(tables.get('density_threshold', {}).items()):
            warn = f"  ⚠ {thr.get('warning','')}" if 'warning' in thr else '  ✓'
            p(f"  {speed} rpm: finest integrating module = {thr['module_mm']} mm"
              f"  dist={thr['dist_from_reference']}{warn}")
        p()

        # Pair discriminability
        p("── Zone 1: Color pair discriminability (top 10 and bottom 5) ──")
        disc = tables.get('pair_discriminability', [])
        for entry in disc[:10]:
            flag = '✓' if entry['readable'] else '✗'
            p(f"  {flag} {entry['pair']:7s}  delta_E={entry['delta_E']:.1f}  "
              f"RGB=({entry['R']:.0f},{entry['G']:.0f},{entry['B']:.0f})")
        if len(disc) > 15:
            p("  ...")
        for entry in disc[-5:]:
            flag = '✓' if entry['readable'] else '✗'
            p(f"  {flag} {entry['pair']:7s}  delta_E={entry['delta_E']:.1f}  "
              f"RGB=({entry['R']:.0f},{entry['G']:.0f},{entry['B']:.0f})")
        p()

        # White achievement
        p("── Zone 4: White achievement ──")
        for label, wa in sorted(tables.get('white_achievement', {}).items(),
                                 key=lambda x: x[1]['delta_E_from_white']):
            flag = '✓' if wa['delta_E_from_white'] < 30 else '⚠'
            p(f"  {flag} {label:9s}  delta_E_from_white={wa['delta_E_from_white']:.1f}  "
              f"RGB=({wa['R']:.0f},{wa['G']:.0f},{wa['B']:.0f})")
        p()

        # Ratio curves: print linearity check
        p("── Zone 2: Ratio curve linearity (R² vs linear fit) ──")
        for pair, speeds in sorted(tables.get('ratio_curves', {}).items()):
            for speed, curve in sorted(speeds.items()):
                if len(curve) < 3:
                    continue
                xs = [pt['ratio_pct'] for pt in curve]
                # Use luminance as scalar output for linearity check
                ys = [luminance(pt['R'], pt['G'], pt['B']) for pt in curve]
                r2 = _r_squared(xs, ys)
                flag = '✓' if r2 > 0.95 else '⚠ NON-LINEAR'
                p(f"  {pair:7s} @{speed:2.0f}rpm  R²={r2:.3f}  {flag}")
        p()

        # Encoder LUT: spot-check inverse at 50%
        p("── Encoder LUT spot-check (luminance @ 50% → should recover ≈50%) ──")
        for pair, speeds in sorted(tables.get('ratio_curves', {}).items()):
            for speed, curve in sorted(speeds.items()):
                pt50 = next((pt for pt in curve if pt['ratio_pct'] == 50), None)
                if pt50 is None:
                    continue
                lum50 = luminance(pt50['R'], pt50['G'], pt50['B'])
                lut_pts = tables['encoder_lut'].get(pair, {}).get(speed, [])
                recovered = encoder_lookup(lut_pts, lum50)
                err = abs(recovered - 50.0)
                flag = '✓' if err < 2 else f'⚠ err={err:.1f}'
                p(f"  {pair:7s} @{speed:2.0f}rpm  lum={lum50:.1f}  "
                  f"recovered={recovered:.1f}%  {flag}")


def _r_squared(xs, ys):
    """Coefficient of determination for linear fit."""
    n = len(xs)
    if n < 2:
        return 1.0
    xm = sum(xs) / n
    ym = sum(ys) / n
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    sxx = sum((x - xm) ** 2 for x in xs) + 1e-9
    slope = sxy / sxx
    ssres = sum((y - (ym + slope * (x - xm))) ** 2 for x, y in zip(xs, ys))
    sstot = sum((y - ym) ** 2 for y in ys) + 1e-9
    return max(0.0, 1.0 - ssres / sstot)

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build DSA v4 calibration tables from measurement CSVs")
    p.add_argument('csvs', nargs='+', help='Measurement CSV files (one or more speeds)')
    p.add_argument('--out', default='calibration_tables.json', help='Output JSON path')
    p.add_argument('--threshold', type=float, default=INTEGRATION_THRESHOLD,
                   help=f'Integration distance threshold (default {INTEGRATION_THRESHOLD})')
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading CSVs:")
    rows = load_all(args.csvs)
    if not rows:
        sys.exit("No valid measurement rows found — fill in the CSV first")

    print(f"Total rows: {len(rows)}")
    speeds = sorted(set(r['_speed'] for r in rows))
    print(f"Speeds:     {speeds} rpm")

    print("Extracting zones...")
    z5 = extract_z5(rows)
    z4 = extract_z4(rows)
    z3 = extract_z3(rows)
    z2 = extract_z2(rows)
    z1 = extract_z1(rows)

    print("Building tables...")
    ratio_curves  = build_ratio_curves(z2)
    density_thr   = build_density_threshold(z3, z5)
    pair_disc     = build_pair_discriminability(z1, z5)
    white_ach     = build_white_achievement(z4, z5)
    encoder_lut   = build_encoder_lut(ratio_curves)

    # Serialize z5 anchors (convert tuple values to lists for JSON)
    z5_serial = {str(sp): {c: [round(v, 1) for v in rgb]
                            for c, rgb in anchors.items()}
                 for sp, anchors in z5.items()}

    tables = {
        'format_version':       1,
        'speeds_rpm':           speeds,
        'z5_anchors':           z5_serial,
        'ratio_curves':         ratio_curves,
        'density_threshold':    {str(k): v for k, v in density_thr.items()},
        'pair_discriminability': pair_disc,
        'white_achievement':    white_ach,
        'encoder_lut':          encoder_lut,
    }

    out_path = args.out
    report_path = str(Path(out_path).with_suffix('')) + '_report.txt'

    with open(out_path, 'w') as f:
        json.dump(tables, f, indent=2)
    print(f"Saved: {out_path}")

    write_report(tables, report_path)
    print(f"Saved: {report_path}")

    # Quick summary
    n_readable = sum(1 for e in pair_disc if e['readable'])
    print(f"\nSummary:")
    print(f"  Usable color pairs: {n_readable}/{len(pair_disc)}")
    if density_thr:
        for sp, thr in sorted(density_thr.items()):
            print(f"  Min module @ {sp}rpm: {thr['module_mm']} mm")
    print(f"  Ratio curves built: {sum(len(v) for v in ratio_curves.values())} (pair×speed)")


if __name__ == '__main__':
    main()
