"""
dsa_v4_reader.py — Standalone DSA v1 (.dsa1) File Reader
=========================================================
Reads a .dsa1 file and extracts per-band energy coefficients
suitable for feeding into the DSA v4 encoder.

Self-contained — does NOT import from the reference implementation.
Implements only what is needed:
  - Header parsing (32-byte struct)
  - Frame index parsing (6 bytes/frame)
  - Band-level Huffman decode (LAMBDA=0.4, 34 symbols)
  - Per-band RMS energy extraction

Output: np.ndarray shape (n_bands, n_frames), values in [0, 1]

Band order matches DSA spec §3:
  bands 0..7   = L0 (bass, 0–1 kHz)
  bands 8..23  = L1 (mid, 1–6 kHz)
  bands 24..47 = L2 (treble, 6–22.05 kHz)
"""

import heapq
import math
import struct
import zlib
import sys
from pathlib import Path

import numpy as np

# ── DSA v1 constants (from SPEC.md §3 and §9) ─────────────────────────────────

FILE_MAGIC     = b'DSA1'
HEADER_FMT     = '<4sBBIIHIIII'   # 32 bytes
HEADER_SIZE    = 32
FRAME_IDX_ENTRY = 6               # type:u8 + gop_pos:u8 + frame_idx:u32

FTYPE_K = 0x00
FTYPE_B = 0x01
FTYPE_S = 0x02

# Band layout — 48 bands total (§3.1)
L0_BANDS = 8
L1_BANDS = 16
L2_BANDS = 24
NUM_BANDS = L0_BANDS + L1_BANDS + L2_BANDS   # 48

# L0: 8 equal-width bands from 0 to 1000 Hz
SAMPLE_RATE = 44100
L0_EDGES = [round(i * 1000 / L0_BANDS) for i in range(L0_BANDS + 1)]

# L1: 16 equal-width bands from 1000 to 6000 Hz
L1_EDGES = [round(1000 + i * 5000 / L1_BANDS) for i in range(L1_BANDS + 1)]

# L2: 24 log-spaced bands from 6000 to 22050 Hz (§3.1)
# b[k] = round(6000 × (22050/6000)^(k/24)) for k=0..24
L2_EDGES = [round(6000 * (22050 / 6000) ** (k / 24)) for k in range(L2_BANDS + 1)]

MDCT_M = 512   # MDCT window half-size (1024-pt MDCT)
NYQUIST = SAMPLE_RATE // 2

def _hz_to_bin(hz):
    return int(hz * MDCT_M / NYQUIST)

# Bin ranges per band (lo_inclusive, hi_exclusive)
BINS = []
all_edges = (L0_EDGES[:-1] + [L0_EDGES[-1]] +
             L1_EDGES[1:-1] + [L1_EDGES[-1]] +
             L2_EDGES[1:-1] + [L2_EDGES[-1]])
# Build (lo_hz, hi_hz) per band
band_hz_ranges = []
for edges in [L0_EDGES, L1_EDGES, L2_EDGES]:
    for i in range(len(edges) - 1):
        band_hz_ranges.append((edges[i], edges[i + 1]))

for lo_hz, hi_hz in band_hz_ranges:
    lo_bin = _hz_to_bin(lo_hz)
    hi_bin = max(lo_bin + 1, _hz_to_bin(hi_hz))
    BINS.append((lo_bin, hi_bin))

BAND_SIZES = [hi - lo for lo, hi in BINS]

_LAYER_BAND_SIZES = [
    BAND_SIZES[:L0_BANDS],
    BAND_SIZES[L0_BANDS: L0_BANDS + L1_BANDS],
    BAND_SIZES[L0_BANDS + L1_BANDS:],
]

# ── Huffman codec (§9) ─────────────────────────────────────────────────────────

MAX_DIRECT = 31
SYM_ESC    = 32
SYM_EOB    = 33
N_SYMBOLS  = 34
LAMBDA     = 0.4

def _build_huffman():
    probs = {}
    for k in range(MAX_DIRECT + 1):
        probs[k] = math.exp(-LAMBDA * k)
    tail = math.exp(-LAMBDA * (MAX_DIRECT + 1)) / (1.0 - math.exp(-LAMBDA) + 1e-15)
    probs[SYM_ESC] = max(tail, 1e-9)
    probs[SYM_EOB] = math.exp(-LAMBDA * 4)
    total = sum(probs.values())
    for k in probs:
        probs[k] /= total

    heap, cnt = [], 0
    for sym, p in probs.items():
        heapq.heappush(heap, (p, cnt, sym))
        cnt += 1
    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (p1 + p2, cnt, [n1, n2]))
        cnt += 1
    _, _, root = heap[0]

    def _build(node, code, bits):
        if isinstance(node, int):
            return node
        left, right = node
        lt = _build(left,  code << 1,       bits + 1)
        rt = _build(right, (code << 1) | 1, bits + 1)
        return (lt, rt)

    return _build(root, 0, 0)

_DECODE_ROOT = _build_huffman()


class _BitReader:
    __slots__ = ('_data', '_bp', '_pp')

    def __init__(self, data: bytes):
        self._data = data
        self._bp = 0
        self._pp = 0

    def read_bit(self) -> int:
        if self._bp >= len(self._data):
            return 0
        b = (self._data[self._bp] >> (7 - self._pp)) & 1
        self._pp += 1
        if self._pp == 8:
            self._pp = 0
            self._bp += 1
        return b

    def read(self, n_bits: int) -> int:
        v = 0
        for _ in range(n_bits):
            v = (v << 1) | self.read_bit()
        return v


def _decode_sym(r: _BitReader) -> int:
    node = _DECODE_ROOT
    while isinstance(node, tuple):
        node = node[r.read_bit()]
    return node


def decode_band(data: bytes, n_coeffs: int) -> np.ndarray:
    """
    Decode one band's Huffman data into an int16 coefficient array.
    Returns zero array for empty data.
    """
    coeffs = np.zeros(n_coeffs, dtype=np.int16)
    if not data:
        return coeffs
    r = _BitReader(data)
    i = 0
    while i < n_coeffs:
        sym = _decode_sym(r)
        if sym == SYM_EOB:
            break
        if sym == SYM_ESC:
            mag = r.read(12)
        else:
            mag = sym
        if mag != 0:
            sign = r.read_bit()
            coeffs[i] = -mag if sign else mag
        i += 1
    return coeffs


# ── .dsa1 parser ──────────────────────────────────────────────────────────────

class DSA1Reader:
    """
    Read a .dsa1 file and extract MDCT coefficients per band per frame.
    """

    def __init__(self, path: str):
        self.path = path
        self.data = Path(path).read_bytes()
        self._parse_header()
        self._parse_frame_index()

    def _parse_header(self):
        if self.data[:4] != FILE_MAGIC:
            raise ValueError(f"Not a DSA1 file: {self.path}")
        (magic, version, mode, sr, n_frames, kbps,
         l0_off, l1_off, l2_off, crc_off) = struct.unpack_from(HEADER_FMT, self.data)
        self.version      = version
        self.mode         = mode
        self.sample_rate  = sr
        self.n_frames     = n_frames
        self.bitrate_kbps = kbps
        self.l_offsets    = [l0_off, l1_off, l2_off]
        self.crc_offset   = crc_off

    def _parse_frame_index(self):
        base = HEADER_SIZE
        self.frame_types    = []
        self.frame_gop_pos  = []
        self.frame_indices  = []
        for i in range(self.n_frames):
            off = base + i * FRAME_IDX_ENTRY
            ftype, gop_pos, frame_idx = struct.unpack_from('<BBI', self.data, off)
            self.frame_types.append(ftype)
            self.frame_gop_pos.append(gop_pos)
            self.frame_indices.append(frame_idx)

    def verify_crc(self) -> bool:
        stored   = struct.unpack_from('<I', self.data, self.crc_offset)[0]
        computed = zlib.crc32(self.data[:self.crc_offset]) & 0xFFFFFFFF
        return stored == computed

    def _read_layer(self, layer_idx: int):
        """
        Yield (frame_i, band_sizes, step_array, coeffs_array) for each frame.
        step_array: float32 array of length n_bands_in_layer
        coeffs_array: list of int16 arrays (one per band)
        """
        band_sizes = _LAYER_BAND_SIZES[layer_idx]
        n_bands_l  = len(band_sizes)
        offset = self.l_offsets[layer_idx]

        for fi, ftype in enumerate(self.frame_types):
            size = struct.unpack_from('<H', self.data, offset)[0]
            offset += 2

            if ftype == FTYPE_S or size == 0:
                offset += size
                yield fi, band_sizes, None, None
                continue

            raw = self.data[offset: offset + size]
            offset += size

            # Parse bands: [step:f32][huff_n:u16][huff_data × huff_n bytes]
            steps  = []
            coeffs = []
            pos = 0
            for bi, n_coeffs in enumerate(band_sizes):
                step    = struct.unpack_from('<f', raw, pos)[0];  pos += 4
                huff_n  = struct.unpack_from('<H', raw, pos)[0];  pos += 2
                huff_data = raw[pos: pos + huff_n];               pos += huff_n
                c = decode_band(huff_data, n_coeffs)
                steps.append(step)
                coeffs.append(c)

            yield fi, band_sizes, np.array(steps, dtype=np.float32), coeffs

    def extract_band_energies(self) -> np.ndarray:
        """
        Extract per-band RMS energy for every frame.

        Returns np.ndarray shape (NUM_BANDS, n_frames), dtype float32.
        Values represent physical amplitude (step × RMS of quantized coeffs).
        Normalized to [0, 1] across the full array.
        """
        energies = np.zeros((NUM_BANDS, self.n_frames), dtype=np.float32)

        for layer_idx in range(3):
            band_offset = (0 if layer_idx == 0 else
                           L0_BANDS if layer_idx == 1 else
                           L0_BANDS + L1_BANDS)
            for fi, band_sizes, steps, coeffs in self._read_layer(layer_idx):
                if steps is None:
                    continue
                for bi, (step, c) in enumerate(zip(steps, coeffs)):
                    rms = float(np.sqrt(np.mean(c.astype(np.float32) ** 2)))
                    energies[band_offset + bi, fi] = step * rms

        # Normalize to [0, 1]
        peak = energies.max()
        if peak > 0:
            energies /= peak

        return energies

    @property
    def duration_s(self) -> float:
        frame_ms = MDCT_M * 1000.0 / self.sample_rate
        return self.n_frames * frame_ms / 1000.0

    def __repr__(self):
        return (f"DSA1Reader('{self.path}' "
                f"sr={self.sample_rate} frames={self.n_frames} "
                f"{self.duration_s:.1f}s {self.bitrate_kbps}kbps)")


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description="DSA v1 reader / energy extractor")
    p.add_argument('dsa_file', help='.dsa1 file path')
    p.add_argument('--plot',   action='store_true', help='Plot band energies (requires matplotlib)')
    args = p.parse_args()

    reader = DSA1Reader(args.dsa_file)
    print(reader)

    ok = reader.verify_crc()
    print(f"CRC: {'OK' if ok else 'FAIL'}")

    energies = reader.extract_band_energies()
    print(f"Band energies: shape={energies.shape}  "
          f"min={energies.min():.4f}  max={energies.max():.4f}")
    print(f"Non-zero frames: {np.count_nonzero(energies.max(axis=0))}/{reader.n_frames}")

    # Band-average energy
    band_avg = energies.mean(axis=1)
    print("\nPer-layer average energy:")
    print(f"  L0 (bands  0-7):  {band_avg[:L0_BANDS].mean():.4f}")
    print(f"  L1 (bands  8-23): {band_avg[L0_BANDS:L0_BANDS+L1_BANDS].mean():.4f}")
    print(f"  L2 (bands 24-47): {band_avg[L0_BANDS+L1_BANDS:].mean():.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.imshow(energies, aspect='auto', origin='lower',
                       extent=[0, reader.duration_s, 0, NUM_BANDS])
            plt.colorbar(label='normalized energy')
            plt.xlabel('time (s)')
            plt.ylabel('band index')
            plt.title(f'DSA v1 band energies — {Path(args.dsa_file).name}')
            plt.tight_layout()
            plt.savefig('band_energies.png', dpi=150)
            print("Saved band_energies.png")
        except ImportError:
            print("matplotlib not available — skipping plot")
