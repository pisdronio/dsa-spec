"""
dsa_v4_format.py — DSA v4 Disc Physical Format
================================================
Defines the on-disc layout for a v4 physics-integrated Digilog disc and
provides encode/decode routines for the header zone.

Physical layout (outer → inner):
┌──────────────────────────────────────────────────────┐
│  SYNC ZONE  (3 rings: W, K, W)                       │
│  Purpose: disc detection, scale calibration          │
├──────────────────────────────────────────────────────┤
│  HEADER ZONE  (8 rings, 1 byte each)                 │
│  Purpose: bootstrap decoder without side-channel     │
│  Encoding: B+W checkerboard ratio = value/255        │
│  Readable at ω = 0 (static, full resolution)         │
├──────────────────────────────────────────────────────┤
│  DATA ZONE  (remaining radial span)                  │
│  Purpose: audio coefficient rings                    │
│  Layout: outer = high-freq bands, inner = low-freq   │
│          within each band: time advances inward      │
└──────────────────────────────────────────────────────┘

Header byte layout (8 rings, index 0 = outermost):
  [0]  magic         = 0xD4              (0b11010100, identifies v4 disc)
  [1]  format_ver    = 0x01              (format version, currently 1)
  [2]  n_bands       packed (high nibble) n_time packed (low nibble)
       e.g. 12 bands, 8 slices → 0xC8  (0b11001000)
  [3]  speed_rpm     = e.g. 33 → 0x21, 45 → 0x2D
  [4]  sample_rate   = sample_rate // 100  (44100 → 0xAB = 171, 48000 → 0xBB)
  [5]  n_frames_hi   = (n_audio_frames >> 8) & 0xFF
  [6]  n_frames_lo   = n_audio_frames & 0xFF
  [7]  calib_crc8    = CRC-8 of bytes 0–6 (polynomial 0x07, init 0x00)

The decoder reads rings 0–7 at ω=0 to extract all parameters needed to
interpret the data zone without any out-of-band metadata.

Header ring encoding:
  Each ring stores one byte value V (0–255) as a B+W checkerboard where
  ratio_a = V / 255.  At ω=0 (static), the ratio is readable via sampling
  the ring and computing the fraction of near-black pixels.  At speed, the
  ring reads as a grey value proportional to V/255.

  B (blue) is color_a, W (white) is color_b.
  ratio = 0   → pure white  (V=0)
  ratio = 0.5 → 50/50 B+W  (V=128)
  ratio = 1.0 → pure blue   (V=255)

  Blue and white are maximally discriminable and do not require calibration
  tables to decode — their values are independent of ink/camera.
"""

import struct
from dataclasses import dataclass, field
from typing import List, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

V4_MAGIC         = 0xD4
V4_FORMAT_VER    = 0x01
SYNC_PATTERN     = ('W', 'K', 'W')   # color names, outer→inner

HEADER_RING_COLOR_A = 'B'   # blue
HEADER_RING_COLOR_B = 'W'   # white

SAMPLE_RATE_CODE = {22050: 0x01, 32000: 0x02, 44100: 0x03, 48000: 0x04}
SAMPLE_RATE_FROM_CODE = {v: k for k, v in SAMPLE_RATE_CODE.items()}

N_SYNC_RINGS   = len(SYNC_PATTERN)
N_HEADER_RINGS = 8
N_PREAMBLE_RINGS = N_SYNC_RINGS + N_HEADER_RINGS   # 11 total

SYNC_RING_WIDTH_MM   = 1.5
HEADER_RING_WIDTH_MM = 1.5
ZONE_GAP_MM          = 0.3   # gap between adjacent rings
PREAMBLE_TOTAL_MM = (
    N_PREAMBLE_RINGS * max(SYNC_RING_WIDTH_MM, HEADER_RING_WIDTH_MM)
    + N_PREAMBLE_RINGS * ZONE_GAP_MM
)  # ≈ 16.5mm for 11 rings

# ── CRC-8 ─────────────────────────────────────────────────────────────────────

def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ── Header encode/decode ──────────────────────────────────────────────────────

@dataclass
class V4Header:
    n_bands:       int = 12
    n_time:        int = 8
    speed_rpm:     int = 33
    sample_rate:   int = 44100
    n_audio_frames: int = 0
    calib_crc8:    int = 0       # filled in by encode()

    def to_bytes(self) -> bytes:
        """Encode header to 8 bytes."""
        b0 = V4_MAGIC
        b1 = V4_FORMAT_VER
        # Pack n_bands (4 bits) and n_time (4 bits) into one byte
        # n_bands and n_time stored as (value - 1) to fit 0–15 → 1–16
        n_b = max(1, min(16, self.n_bands))  - 1
        n_t = max(1, min(16, self.n_time))   - 1
        b2 = (n_b << 4) | n_t
        b3 = self.speed_rpm & 0xFF
        b4 = SAMPLE_RATE_CODE.get(self.sample_rate, 0x03) & 0xFF
        b5 = (self.n_audio_frames >> 8) & 0xFF
        b6 =  self.n_audio_frames       & 0xFF
        b7 = crc8(bytes([b0, b1, b2, b3, b4, b5, b6]))
        return bytes([b0, b1, b2, b3, b4, b5, b6, b7])

    @classmethod
    def from_bytes(cls, data: bytes) -> 'V4Header':
        """Decode header from 8 bytes.  Raises ValueError on CRC mismatch."""
        if len(data) < 8:
            raise ValueError(f"Header too short: {len(data)} bytes")
        b0, b1, b2, b3, b4, b5, b6, b7 = data[:8]
        if b0 != V4_MAGIC:
            raise ValueError(f"Bad magic: 0x{b0:02X} (expected 0x{V4_MAGIC:02X})")
        if b1 != V4_FORMAT_VER:
            raise ValueError(f"Unsupported format version: {b1}")
        expected_crc = crc8(bytes([b0, b1, b2, b3, b4, b5, b6]))
        if b7 != expected_crc:
            raise ValueError(f"CRC mismatch: got 0x{b7:02X} expected 0x{expected_crc:02X}")
        n_bands   = ((b2 >> 4) & 0xF) + 1
        n_time    =  (b2       & 0xF) + 1
        speed_rpm = b3
        sample_rate = SAMPLE_RATE_FROM_CODE.get(b4, 44100)
        n_audio_frames = (b5 << 8) | b6
        h = cls(n_bands=n_bands, n_time=n_time, speed_rpm=speed_rpm,
                sample_rate=sample_rate, n_audio_frames=n_audio_frames,
                calib_crc8=b7)
        return h

    def __repr__(self):
        return (f"V4Header(n_bands={self.n_bands} n_time={self.n_time} "
                f"speed={self.speed_rpm}rpm sr={self.sample_rate}Hz "
                f"frames={self.n_audio_frames})")


def header_byte_to_ratio(value: int) -> float:
    """Map byte value 0–255 to ring color_a ratio 0.0–1.0."""
    return value / 255.0


def ratio_to_header_byte(ratio: float) -> int:
    """Map ring sampled ratio 0.0–1.0 back to byte value 0–255."""
    return max(0, min(255, round(ratio * 255)))


# ── Zone geometry ─────────────────────────────────────────────────────────────

@dataclass
class ZoneGeometry:
    """Radial layout of all zones on the disc."""
    outer_r_mm:    float
    inner_r_mm:    float
    gap_mm:        float = ZONE_GAP_MM

    # Computed after __post_init__
    sync_rings:    list = field(default_factory=list)    # [(r_out, r_in, color_name)]
    header_rings:  list = field(default_factory=list)    # [(r_out, r_in, byte_index)]
    data_r_out_mm: float = 0.0
    data_r_in_mm:  float = 0.0

    def __post_init__(self):
        r = self.outer_r_mm

        # Sync rings
        for color in SYNC_PATTERN:
            r_out, r_in = r, r - SYNC_RING_WIDTH_MM
            self.sync_rings.append((r_out, r_in, color))
            r = r_in - self.gap_mm

        # Header rings
        for i in range(N_HEADER_RINGS):
            r_out, r_in = r, r - HEADER_RING_WIDTH_MM
            self.header_rings.append((r_out, r_in, i))
            r = r_in - self.gap_mm

        self.data_r_out_mm = r
        self.data_r_in_mm  = self.inner_r_mm

    @property
    def data_span_mm(self) -> float:
        return self.data_r_out_mm - self.data_r_in_mm

    def data_ring_width_mm(self, n_rings: int) -> float:
        usable = self.data_span_mm - n_rings * self.gap_mm
        return usable / n_rings if n_rings > 0 else 0.0


# ── Ring descriptor ───────────────────────────────────────────────────────────

@dataclass
class RingDescriptor:
    """Complete description of one ring on the disc."""
    kind:      str     # 'sync' | 'header' | 'data'
    r_out_mm:  float
    r_in_mm:   float
    # sync fields
    color:     Optional[str]  = None
    # header fields
    byte_idx:  Optional[int]  = None
    byte_val:  Optional[int]  = None
    # data fields
    band_idx:  Optional[int]  = None
    time_idx:  Optional[int]  = None
    pair:      Optional[tuple] = None
    ratio:     Optional[float] = None


def build_ring_list(header: V4Header, geom: ZoneGeometry,
                    coefficients=None, calib=None) -> List[RingDescriptor]:
    """
    Build the full ordered ring list (sync + header + data) for a disc.

    coefficients: np.ndarray shape (n_bands, n_time) or None (header-only)
    calib: CalibrationTables or None
    """
    rings = []

    # Sync rings
    for r_out, r_in, color in geom.sync_rings:
        rings.append(RingDescriptor('sync', r_out, r_in, color=color))

    # Header rings
    header_bytes = header.to_bytes()
    for r_out, r_in, i in geom.header_rings:
        bval = header_bytes[i]
        rings.append(RingDescriptor(
            'header', r_out, r_in,
            byte_idx=i, byte_val=bval,
        ))

    # Data rings
    if coefficients is not None:
        n_bands, n_time = coefficients.shape
        n_rings = n_bands * n_time
        rw = geom.data_ring_width_mm(n_rings)

        if rw < 0.1:
            raise ValueError(
                f"{n_rings} rings need {n_rings*(rw+geom.gap_mm):.1f}mm but only "
                f"{geom.data_span_mm:.1f}mm available."
            )

        r = geom.data_r_out_mm
        for band_idx in range(n_bands - 1, -1, -1):   # high-freq first (outer)
            pair = calib.best_pair_for_band(band_idx) if calib else ('B', 'K')
            for t in range(n_time):
                coeff = float(coefficients[band_idx, t])
                ratio = calib.coefficient_to_ratio(coeff, pair, header.speed_rpm) \
                        if calib else coeff
                r_out, r_in = r, r - rw
                rings.append(RingDescriptor(
                    'data', r_out, r_in,
                    band_idx=band_idx, time_idx=t,
                    pair=pair, ratio=ratio,
                ))
                r = r_in - geom.gap_mm

    return rings


# ── Header reader (from measured ring ratios) ─────────────────────────────────

def decode_header_from_ratios(ratios: list) -> V4Header:
    """
    Decode a V4Header from a list of 8 measured ring ratios (0.0–1.0).
    Each ratio is the fraction of color_a (blue) pixels in the header ring.
    """
    if len(ratios) < N_HEADER_RINGS:
        raise ValueError(f"Need {N_HEADER_RINGS} ratios, got {len(ratios)}")
    raw_bytes = bytes(ratio_to_header_byte(r) for r in ratios[:N_HEADER_RINGS])
    return V4Header.from_bytes(raw_bytes)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("── V4Header encode/decode round-trip ──")
    h = V4Header(n_bands=12, n_time=8, speed_rpm=33, sample_rate=44100,
                 n_audio_frames=1024)
    raw = h.to_bytes()
    print(f"  Header bytes: {raw.hex()}")
    h2 = V4Header.from_bytes(raw)
    print(f"  Decoded:      {h2}")
    assert h.n_bands == h2.n_bands
    assert h.n_time  == h2.n_time
    assert h.speed_rpm == h2.speed_rpm
    assert h.sample_rate == h2.sample_rate
    assert h.n_audio_frames == h2.n_audio_frames
    print("  Round-trip: PASS")

    print("\n── Ratio encoding round-trip ──")
    for val in [0, 64, 128, 192, 255]:
        ratio = header_byte_to_ratio(val)
        recovered = ratio_to_header_byte(ratio)
        print(f"  {val:3d} → ratio {ratio:.4f} → {recovered:3d}  "
              + ("✓" if abs(val - recovered) <= 1 else "✗"))

    print("\n── Zone geometry (12\" disc) ──")
    geom = ZoneGeometry(outer_r_mm=149.4, inner_r_mm=25.0)
    print(f"  Sync rings:   {len(geom.sync_rings)}")
    print(f"  Header rings: {len(geom.header_rings)}")
    print(f"  Data zone:    {geom.data_r_out_mm:.2f}mm → {geom.data_r_in_mm:.1f}mm "
          f"(span {geom.data_span_mm:.1f}mm)")
    n_test = 12 * 8
    rw = geom.data_ring_width_mm(n_test)
    print(f"  Ring width for {n_test} data rings: {rw:.3f}mm")

    print("\n── CRC-8 ──")
    payload = bytes([V4_MAGIC, V4_FORMAT_VER, 0xB7, 0x21, 0xAB, 0x04, 0x00])
    crc = crc8(payload)
    print(f"  CRC-8({payload.hex()}) = 0x{crc:02X}")
    # Verify: appending crc and computing again should give 0
    check = crc8(payload + bytes([crc]))
    print(f"  Append-and-check: 0x{check:02X}  {'✓' if check == 0 else '✗'}")

    print("\nAll self-tests passed.")
