# DSA Format Specification

**DSA — Digilog Scalable Audio**
**Version:** 1.0
**Status:** Draft
**Repository:** https://github.com/pisdronio/dsa
**Format spec:** https://github.com/pisdronio/digilog-spec
**License:** Creative Commons Attribution-ShareAlike 4.0 (CC-BY-SA 4.0)

---

## About this document

This is the normative specification for the DSA audio format. It defines the bitstream layout, frame structure, quantization model, and layered encoding rules that any conforming DSA encoder or decoder must implement.

The reference implementation is at https://github.com/pisdronio/dsa (Python, GPL v3).
The scientific documentation is in `RESEARCH.md` in the same repository.

---

## 1. Overview

DSA is an audio codec for the Digilog physical audio format. It encodes audio as a layered bitstream that maps directly to concentric rings on a printed disc:

```
Layer 0  →  inner disc rings   (8 bands,  20Hz – 800Hz)    always readable
Layer 1  →  middle rings       (16 bands, 800Hz – 6kHz)    readable on phone camera
Layer 2  →  outer rings        (24 bands, 6kHz – 22kHz)    requires Digilog Rig
```

Core properties:
- Native bidirectional playback (reverse = true reversed audio, not corruption)
- Layered scalable decoding (each layer genuinely enhances the previous)
- Analog degradation model (partial reads → attenuated audio, not silence)
- Variable-speed playback with pitch coupling (vinyl feel)

---

## 2. Transform

### 2.1 MDCT parameters

```
N  = 2048   window length (samples)
M  = 1024   output coefficients (N/2)
HOP = 1024  hop size (50% overlap)
```

Sample rate: 44100 Hz
Frame duration: HOP / sample_rate = 23.2 ms

### 2.2 Window

Sine window:

```
w[n] = sin(π/N × (n + 0.5))    for n = 0, 1, ..., N−1
```

Satisfies the Princen-Bradley (TDAC) condition:

```
w[n]² + w[n + M]² = 1    for all n = 0, ..., M−1
```

### 2.3 Forward MDCT

```
X[k] = sqrt(2/M) × Σ_{n=0}^{N−1}  w[n] × x[n] × cos(π/M × (n + 0.5 + M/2) × (k + 0.5))

for k = 0, 1, ..., M−1
```

### 2.4 Inverse MDCT

```
x'[n] = sqrt(2/M) × Σ_{k=0}^{M−1}  X[k] × cos(π/M × (n + 0.5 + M/2) × (k + 0.5))

for n = 0, 1, ..., N−1
```

### 2.5 Overlap-add reconstruction

Output is reconstructed by overlap-add of consecutive IMDCT frames with HOP=M shift:

```
out[i×HOP : i×HOP + N] += imdct(coeffs[i])
```

TDAC guarantees perfect reconstruction in the valid interior region (SNR ≥ 238 dB).

---

## 3. Frequency Bands

48 perceptual bands split across three layers:

```
Layer 0:   8 bands   20Hz –   800Hz   linear spacing    (indices 0–7)
Layer 1:  16 bands  800Hz –  6000Hz   log spacing       (indices 8–23)
Layer 2:  24 bands  6000Hz – 22050Hz  log spacing       (indices 24–47)
```

### 3.1 Band boundaries (Hz)

```
Layer 0 — linear, 8 bands:
  Band  0:    20 –   120 Hz
  Band  1:   120 –   220 Hz
  Band  2:   220 –   320 Hz
  Band  3:   320 –   420 Hz
  Band  4:   420 –   520 Hz
  Band  5:   520 –   620 Hz
  Band  6:   620 –   720 Hz
  Band  7:   720 –   800 Hz

Layer 1 — log, 16 bands:
  Band  8:   800 –   900 Hz
  Band  9:   900 –  1013 Hz
  Band 10:  1013 –  1139 Hz
  Band 11:  1139 –  1282 Hz
  Band 12:  1282 –  1442 Hz
  Band 13:  1442 –  1622 Hz
  Band 14:  1622 –  1825 Hz
  Band 15:  1825 –  2054 Hz
  Band 16:  2054 –  2311 Hz
  Band 17:  2311 –  2600 Hz
  Band 18:  2600 –  2926 Hz
  Band 19:  2926 –  3294 Hz
  Band 20:  3294 –  3708 Hz
  Band 21:  3708 –  4174 Hz
  Band 22:  4174 –  4699 Hz
  Band 23:  4699 –  6000 Hz

Layer 2 — log, 24 bands:
  Band 24:  6000 –  6334 Hz
  Band 25:  6334 –  6687 Hz
  Band 26:  6687 –  7060 Hz
  Band 27:  7060 –  7454 Hz
  Band 28:  7454 –  7869 Hz
  Band 29:  7869 –  8307 Hz
  Band 30:  8307 –  8770 Hz
  Band 31:  8770 –  9259 Hz
  Band 32:  9259 –  9776 Hz
  Band 33:  9776 – 10320 Hz
  Band 34: 10320 – 10895 Hz
  Band 35: 10895 – 11502 Hz
  Band 36: 11502 – 12143 Hz
  Band 37: 12143 – 12820 Hz
  Band 38: 12820 – 13534 Hz
  Band 39: 13534 – 14289 Hz
  Band 40: 14289 – 15085 Hz
  Band 41: 15085 – 15926 Hz
  Band 42: 15926 – 16813 Hz
  Band 43: 16813 – 17750 Hz
  Band 44: 17750 – 18739 Hz
  Band 45: 18739 – 19784 Hz
  Band 46: 19784 – 20886 Hz
  Band 47: 20886 – 22050 Hz
```

L2 band boundaries are derived from the formula:

```
b[k] = round(6000 × (22050 / 6000)^(k / 24))   for k = 0, 1, ..., 24
```

Band 24+k spans b[k] to b[k+1]. The inter-band ratio is (22050/6000)^(1/24) ≈ 1.0557.

Exact MDCT bin ranges are derived from the band Hz boundaries using truncation (not rounding):

```
bin_lo[b] = max(0,        int(lo_hz × N / sample_rate))
bin_hi[b] = min(MDCT_M−1, max(int(hi_hz × N / sample_rate), bin_lo[b] + 1))
```

The `max(..., bin_lo[b] + 1)` ensures every band contains at least one MDCT coefficient.

### 3.2 Perceptual weights

Simplified ISO 226 equal-loudness weighting applied to quantization step allocation:

```
< 100 Hz:       0.25   sub-bass
100 – 300 Hz:   0.55   bass
300 Hz – 1 kHz: 0.80   low-mid
1 – 4 kHz:      1.00   peak sensitivity (maximum weight)
4 – 8 kHz:      0.75   presence
8 – 12 kHz:     0.45   air
> 12 kHz:       0.20   ultra-high
```

---

## 4. Frame Structure

### 4.1 Frame types

| Type | Symbol | Description |
|------|--------|-------------|
| Keyframe | K | Self-contained spectral snapshot. Decodable without reference to other frames. |
| Bidirectional | B | Residual relative to interpolation between surrounding K-frames. |
| Silence | S | Energy below silence threshold (−55 dBFS). Minimal data. |

### 4.2 GOP structure

Group of Pictures: 8 frames (~185ms at 23.2ms/frame).

```
K  B  B  B  B  B  B  B  K  B  B  B  ...
0  1  2  3  4  5  6  7  8  9  10 11
↑                       ↑
Keyframe               Keyframe
(resync point)         (resync point)
```

K-frames occur at frame indices that are multiples of 8.
B-frames at positions 1–7 within each GOP reference the K-frames at positions 0 and 8.

### 4.3 Silence threshold

A frame is classified as silence (S-frame) when:

```
max(energies_db) < −55 dBFS
```

S-frames store a silence flag only. Their MDCT coefficients are all zero.

### 4.4 K-frame encoding

1. Apply perceptual quantization to all 48 bands
2. Entropy-code with static Huffman table per band group
3. Pack into self-contained K-frame record

### 4.5 B-frame encoding

1. Compute spectral interpolation between surrounding K-frames:
   ```
   α = gop_pos / GOP_SIZE    (0 < α < 1)
   interp[k] = (1 − α) × K0_coeffs[k] + α × K1_coeffs[k]
   ```
2. Compute residual: `residual[k] = actual_coeffs[k] − interp[k]`
3. Quantize residual (typically much smaller than full coefficients)
4. Entropy-code and pack with references to surrounding K-frame indices

B-frames decode identically in forward and reverse order because both surrounding K-frames are always available regardless of playback direction.

---

## 5. Perceptual Quantization

### 5.1 Absolute Threshold of Hearing (ATH)

Per-band ATH in dBFS (normalized to 96 dBSPL for 16-bit audio):

```
ATH_dBSPL(f_kHz) = 3.64×f^−0.8 − 6.5×exp(−0.6×(f−3.3)²) + 0.001×f⁴

ATH_dBFS[b] = ATH_dBSPL(fc[b]) − 96
```

where `fc[b]` is the center frequency of band b in kHz.

### 5.2 Masking thresholds

Bark-scale simultaneous masking using asymmetric spreading function:

```
S[b, m] = −MI − 25 × dz    if dz ≥ 0  (upward masking)
S[b, m] = −MI − 40 × |dz|  if dz < 0  (downward masking)

where MI = 14 dB  (masking index)
      dz = Bark(b) − Bark(m)
      Bark(f) = 13×arctan(0.00076f) + 3.5×arctan((f/7500)²)
```

Global masking threshold per band:

```
T[b] = max(ATH[b],  max_m( L[m] + S[b, m] ))
```

### 5.3 Step sizes

```
step[b] = 10^((T[b] − 3dB) / 20)
```

The 3 dB headroom keeps quantization noise safely below (not merely at) the masking threshold.

### 5.4 Bidirectional rate-distortion budget enforcement

Layer degradation priority: L2 first (outer rings), L1 second, L0 last (inner rings).

**Phase 1 — over budget:** Binary search for minimum upward scale factor per layer that brings estimated bit cost within budget. Applied L2 → L1 → L0 until cost ≤ budget.

**Phase 2 — under budget:** When surplus budget > 5% of total, binary search for a uniform downward scale factor across all bands that consumes the budget. Lower bound: `peak_coeff[b] / (MAX_QUANT × step[b])` per band, to prevent quantized value saturation at MAX_QUANT = 2047.

### 5.5 Quantization

Uniform scalar quantization per band:

```
q[k] = clip(round(coeff[k] / step[b]), −MAX_QUANT, +MAX_QUANT)
```

where MAX_QUANT = 2047 and b is the band containing bin k.

---

## 6. Analog Degradation Model

### 6.1 Confidence vector

The visual decoder supplies a per-band confidence vector α of shape (48,) with α[b] ∈ [0.0, 1.0]:

```
Mode 1 (discrete dots):
  α[b] = 1.0   band b read cleanly
  α[b] = 0.0   band b region unreadable

Mode 2 (gradient dots):
  α[b] = continuous value from gradient clarity
         0.0 = completely unreadable
         1.0 = perfect read
```

### 6.2 Confidence-weighted dequantization

```
C̃[k] = q[k] × step[b] × α[b]
```

At α = 1.0: standard dequantization.
At α = 0.5: −6 dB attenuation (exactly 20×log₁₀(0.5)).
At α = 0.0: band contributes zero to IMDCT reconstruction.

### 6.3 Layer selection

Layer selection is implemented as a confidence mask:

```
L0 excluded:  α[0:8]    = 0.0
L1 excluded:  α[8:24]   = 0.0
L2 excluded:  α[24:48]  = 0.0
```

### 6.4 K-frame loss recovery

When a K-frame is unreadable, the decoder holds the last valid spectral shape and applies exponential decay:

```
decay_per_frame = exp(−frame_ms / (τ × 1000))
                = exp(−23.2 / 60000)
                ≈ 0.679

substitute_K[n_frames_lost] = last_valid_K × 0.679^n_frames_lost
```

τ = 60 ms. Acoustic result: a note fading out, not a digital dropout.

---

## 7. Reverse Playback

Process frames in reverse index order, then apply IMDCT and overlap-add in that order:

```
# Forward
for i in 0, 1, 2, ..., N−1:
    out[i×HOP : i×HOP + N] += imdct(coeffs[i])

# Reverse
for i in N−1, N−2, ..., 0:
    out[j×HOP : j×HOP + N] += imdct(coeffs[i])
    j += 1
```

The sine window satisfies `w[n] = w[N−1−n]`, so TDAC cancellation operates identically in reverse. The result is the true time-reversed audio signal at frame-level temporal granularity (23.2 ms per frame).

---

## 8. Variable-Speed Playback

Speed multiplier `s` relative to nominal (s = 1.0 at standard playback):

```
s > 1.0   faster, pitch rises  (vinyl scratch forward)
s < 1.0   slower, pitch drops  (vinyl slow-down)
s = 0.0   stopped: exponential amplitude decay, τ = 60 ms
s < 0.0   reverse at |s|
```

Pitch is coupled to speed (natural vinyl feel). Output sample count:

```
n_output = round(n_decoded / s)
```

Implemented via linear resampling. For real-time DJ use, SoundTouch (LGPL) is recommended for latency < 50 ms. For playback-only use, RubberBand (GPL) provides higher quality.

---

## 9. Entropy Coding

### 9.1 Codebook

One static Huffman codebook shared across all layers and bands. The codebook is fixed at encode and decode time and is not transmitted in the bitstream.

**Symbol set (34 symbols):**

```
0 – 31   coefficient magnitude, coded directly
32       SYM_ESC — magnitude ≥ 32; followed by 12-bit unsigned literal
33       SYM_EOB — end-of-band; all remaining coefficients in this band are zero
```

**Sign bit:** one bit immediately following each nonzero magnitude symbol. `0` = positive, `1` = negative. Not emitted for zero magnitudes.

**End-of-band:** after the last nonzero coefficient in a band, emit `SYM_EOB`. Coefficients after `EOB` are implicitly zero. All-zero bands emit no symbols (zero-length Huffman payload).

**Escape:** for |q| ≥ 32, emit `SYM_ESC` followed by a 12-bit unsigned integer encoding the exact magnitude. Maximum storable magnitude = 2^12 − 1 = 4095. Encoder MUST clip quantized values to this range before entropy coding.

**Bit order:** MSB-first. The Huffman bitstream is padded to a byte boundary with zero bits at the end.

### 9.2 Codebook derivation

```
LAMBDA = 0.4

P(k) = exp(−LAMBDA × k)    for k = 0, 1, ..., 31
P(SYM_ESC) = exp(−LAMBDA × 32) / (1 − exp(−LAMBDA))   (geometric tail)
P(SYM_EOB) = exp(−LAMBDA × 4)                          (empirical weight)
```

Probabilities are normalized to sum to 1, then a standard Huffman tree is built by minimum-probability merging. The resulting code for magnitude 0 is ≤ 3 bits; `SYM_ESC` is ≥ 8 bits.

### 9.3 Band wire format

Each band within a layer block is serialized as:

```
[step:     float32 LE]     quantization step (raw IEEE 754 float, 4 bytes)
[huff_n:   uint16 LE]      byte count of Huffman payload (0 = all-zero band)
[huff_data: huff_n bytes]  MSB-first Huffman bitstream, byte-padded
```

`huff_n = 0` indicates an all-zero band — no `huff_data` bytes follow.

---

## 10. Bitstream Format — DSA1

### 10.1 File layout

```
DSA1 File
├── Header              (32 bytes)
├── Frame index         (6 bytes × n_frames)
├── Layer 0 block       (per-frame: [size:u16][Huffman layer data])
├── Layer 1 block       (same structure)
├── Layer 2 block       (same structure)
└── CRC32               (4 bytes)
```

Layer blocks are read sequentially. Each frame's entry in a layer block begins with a 2-byte `uint16 LE` size field giving the byte count of that frame's Huffman layer data. `size = 0` for S-frames (no data follows). This allows an L0-only reader to seek to `layer0_offset` and read only that block, ignoring L1 and L2 entirely.

### 10.2 Header (32 bytes)

| Offset | Size | Field | Value |
|--------|------|-------|-------|
| 0 | 4 | Magic | `0x44 0x53 0x41 0x31` ("DSA1") |
| 4 | 1 | Version | `0x01` |
| 5 | 1 | Mode | `0x01` = discrete, `0x02` = gradient |
| 6 | 4 | Sample rate | uint32 LE (e.g. 44100) |
| 10 | 4 | Frame count | uint32 LE |
| 14 | 2 | Nominal bitrate | uint16 LE (kbps) |
| 16 | 4 | L0 block offset | uint32 LE (bytes from file start) |
| 20 | 4 | L1 block offset | uint32 LE |
| 24 | 4 | L2 block offset | uint32 LE |
| 28 | 4 | CRC32 offset | uint32 LE (bytes from file start) |

The L0 block offset equals `32 + 6 × n_frames` (header + frame index). A reader SHOULD verify this against the header field. The CRC32 offset equals the start of the 4-byte CRC32 checksum at the end of the file.

### 10.3 Mode byte values

| Value | Name | Description |
|-------|------|-------------|
| `0x01` | MODE_DISCRETE | Discrete dot encoding. Visual decoder produces α ∈ {0, 1}. |
| `0x02` | MODE_GRADIENT | Gradient dot encoding. Visual decoder produces α ∈ [0, 1]. |
| `0x04` | MODE_PHYSICS | Physics-integrated v4 (reserved, not yet specified). |

### 10.4 Frame index

One entry per frame, 6 bytes each, stored consecutively immediately after the header (at file offset 32):

```
[frame_type: uint8]    0x00 = K-frame, 0x01 = B-frame, 0x02 = S-frame
[gop_pos:    uint8]    position within GOP (0 = K, 1–7 = B)
[frame_idx:  uint32 LE]  frame sequence number (0-based, monotonically increasing)
```

The frame index describes frame metadata only. Layer data is stored in the layer blocks and accessed sequentially using per-frame `[size:u16]` prefix entries — there is no random-access offset table for layer positions.

### 10.5 Frame record layout

Frame metadata is stored in the frame index (§10.4). Layer data for each frame is stored in the layer blocks as a variable-length record prefixed by a 2-byte size field:

```
Layer block entry (per frame, per layer):
  [size:      uint16 LE]      byte count of the Huffman layer data below
  [layer data: size bytes]    band records (see §9.3), one per band in the layer

  For S-frames:  size = 0, no layer data follows.
```

**K-frame layer data** — one band record per band (see §9.3 for wire format):
```
For each band b in the layer:
  [step:    float32 LE]    quantization step for band b
  [huff_n:  uint16 LE]     Huffman payload byte count (0 = all-zero band)
  [huff_data: huff_n bytes] Huffman-coded magnitudes, EOB, signs (§9.1)
```

**B-frame layer data** — identical structure to K-frame, but the Huffman payload encodes quantized *residuals* relative to the linear interpolation between the surrounding K-frames:

```
residual[k] = actual_coeffs[k] − interp[k]
interp[k]   = (1 − α) × K0_coeffs[k] + α × K1_coeffs[k]
α           = gop_pos / GOP_SIZE
```

The surrounding K-frames are identified by:
- `k0`: the K-frame at `frame_idx − gop_pos` (preceding K-frame boundary)
- `k1`: the K-frame at `frame_idx − gop_pos + GOP_SIZE`

k1 is implicit — not stored in the bitstream. A conforming encoder MUST ensure k1 always exists by emitting a K-frame at the final frame position if the last GOP is incomplete. A decoder where `k1_frame_idx ≥ frame_count` SHOULD apply K-frame loss recovery (§6.4) rather than producing corrupted output.

**S-frame** — no layer data in any block. The frame index entry `frame_type = 0x02` is the complete record. All coefficients are implicitly zero.

### 10.6 CRC32

CRC32 (ISO 3309 polynomial, same as `zlib.crc32`) of all bytes from file offset 0 through the end of the L2 block. Stored as 4-byte little-endian uint32 immediately after the L2 block. The header's `crc32_offset` field (offset 28) gives the absolute file position of this checksum. A conforming decoder MUST verify the CRC32 before decoding.

---

## 11. Disc Layout — GradientDot Interface

DSA disc encoding maps MDCT coefficients to physical gradient dots on the Digilog disc surface.

### 11.1 GradientDot record

Each coefficient produces one GradientDot:

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | Source frame index |
| `band_idx` | int | Band index 0–47 |
| `coeff_idx` | int | Coefficient position within band |
| `layer` | int | 0, 1, or 2 |
| `color_a` | str | Primary color (layer-assigned) |
| `color_b` | str | Secondary color (layer-assigned) |
| `steepness` | float | Gradient steepness ∈ [0, 1] — encodes coefficient magnitude |
| `direction` | int | +1 or −1 — encodes coefficient sign |

### 11.2 Color pair assignment

Layer-specific color pairs for maximum visual discriminability under expected reading conditions:

```
Layer 0 (inner rings, always readable):
  High-contrast pairs: Black↔White, Black↔Yellow, Black↔Cyan
  Cycling over band index within layer.

Layer 1 (middle rings, phone camera):
  Complementary pairs: Red↔Cyan, Blue↔Yellow, Green↔Purple
  Cycling over band index within layer.

Layer 2 (outer rings, Digilog Rig):
  Full 8-color palette, all pairs available.
  8 pairs cycling over band index within layer.
```

### 11.3 Steepness encoding

**Mode 1 (discrete):**

```
steepness = |q_int| / 31
```

32 discrete levels (0, 1/31, 2/31, ..., 1). Layer 2 pre-emphasis: `min(1.0, steepness × 1.15)`.

**Mode 2 (gradient):**

```
steepness = |c_float| / band_peak
```

where `band_peak` = maximum absolute coefficient value in that band across all frames.
Layer 2 pre-emphasis: `min(1.0, steepness × 1.15)`.

### 11.4 Layer 2 pre-emphasis

Factor: 1.15×. Compensates for lens PSF blur at 15 cm focal distance under Digilog Rig LED illumination. Layer 0 and Layer 1 dots are large enough that PSF blur is negligible.

### 11.5 Disc geometry — standard 12-inch disc

```
Disc diameter:           290 mm
Outer audio ring radius: 141 mm
Inner audio ring radius:  62 mm
Audio zone width:         79 mm   (141 − 62)
Ring width per band:       1.65 mm (79mm / 48 bands)
Clock track:             141 mm – 143.5 mm (2.5 mm wide)
Label area:              < 55 mm radius
Spindle hole:            < 7 mm radius
Reference markers:       8 dots on clock track, 45° spacing
```

### 11.6 Disc capacity limits

The binding constraint is the **innermost audio ring** (band 0, L0 bass) at radius 62 mm. Its circumference (389 mm) is less than half the outer ring (886 mm), so it determines the maximum number of arc segments before individual arcs become too small to print or read.

```
Reader type                Min readable arc   Max frames   Max duration
─────────────────────────────────────────────────────────────────────────
Digilog Rig (standard)     0.3 mm             1298         ~30 s
Phone camera               0.5 mm              779         ~18 s
High-res inkjet + Rig      0.2 mm             1947         ~45 s
```

These values apply to a standard 290 mm disc at 44100 Hz sample rate (1024-sample hop).

**Defined format constants:**

```
DISC_MAX_FRAMES_STANDARD = 1298   # Digilog Rig, 0.3mm min arc
DISC_MAX_FRAMES_PHONE    =  779   # phone camera, 0.5mm min arc
DISC_MAX_DURATION_STANDARD_S ≈ 30.1 s
DISC_MAX_DURATION_PHONE_S    ≈ 18.1 s
```

A conforming disc encoder SHOULD warn when the input audio exceeds `DISC_MAX_FRAMES_STANDARD`. It MUST NOT silently produce a disc layout that exceeds the physical readability limits of the target reader type.

---

## 12. Conformance

A conforming DSA decoder MUST:

1. Implement MDCT/IMDCT with N=2048, M=1024, sine window
2. Decode K-frames without reference to any other frame
3. Decode B-frames using the two surrounding K-frames
4. Apply the confidence-weighted dequantization model: `C̃[k] = q[k] × step[b] × α[b]`
5. Implement K-frame loss recovery with τ ≈ 60 ms exponential decay
6. Decode in reverse order when `reverse=True`
7. Verify CRC32 before decoding

A conforming DSA decoder SHOULD:

- Support layer selection (L0 only, L0+L1, all layers)
- Support variable-speed playback
- Support analog degradation input (continuous α ∈ [0, 1])

A conforming DSA encoder MUST:

1. Produce valid DSA1 file headers with correct magic bytes and CRC32
2. Align K-frames to GOP boundaries (multiples of 8)
3. Encode K-frames as self-contained records
4. Encode B-frames with residuals relative to surrounding K-frame interpolation
5. Apply the perceptual masking model for quantization step selection
6. Enforce layer priority degradation order (L2 first, L0 last)

---

## 13. Reference Implementation

Python reference implementation: https://github.com/pisdronio/dsa

```
dsa_analyzer.py   — MDCT, band layout, frame analysis
dsa_quantizer.py  — perceptual quantizer, masking model
dsa_encoder.py    — K/B-frame encoders
dsa_huffman.py    — static Huffman table
dsa_bitstream.py  — DSA1 file format
dsa_decoder.py    — decoder, reverse playback, variable speed
dsa_disc.py       — GradientDot disc layout interface
dsa_cli.py        — unified command-line tool
```

Test suite: `tests/` — 74 tests, all pass on the reference implementation.

---

## 14. License

This specification is licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0).

You are free to implement this specification in any software, hardware, or physical medium. Implementations may be proprietary. Modifications to the specification itself must be shared under the same license.

Reference implementation: GPL v3 (https://github.com/pisdronio/dsa/blob/main/LICENSE)

---

*DSA is part of the Digilog open physical audio format.*
*github.com/pisdronio/dsa*
*Scan the groove.*
