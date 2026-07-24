> **Context:** Read [`architectural_decisions.md`](architectural_decisions.md) and [`review_protocol.md`](review_protocol.md) before starting this milestone.

## Milestone 4 — MX Block-Scaled Types

**Executed by:** fresh implementor agent
**Reviewed before:** Milestone 5 begins

### Goal

Implement E8/UE8M0 and E5M3 scale decoders, Float4/Float6/BFloat6 unpackers, and the
block-scaled GEMM reference.

### Tasks

**4.1 — Scale decoders in `Tensile/client/mx_types.py`**
- `decode_e8(scale_bytes: np.ndarray) -> np.ndarray`:
  `2.0 ** (scale_bytes.astype(np.float32) - 127)`, with `scale_bytes == 0xFF → np.nan`.
  Matches `DataTypes_E8.hpp:101–115`.
- `decode_e5m3(scale_bytes: np.ndarray) -> np.ndarray`: 5-bit exponent + 3-bit mantissa,
  standard IEEE float interpretation. ~10 lines of numpy bit manipulation.
- Unit tests: for each decoder, test at least 10 hand-computed (byte → float) pairs
  including 0x00, 0x7F, 0x80, 0xFE, 0xFF. Round-trip tests are insufficient on their own.

**4.2 — Float4/6/BFloat6 unpackers**
- `unpack_float4(packed: np.ndarray) -> np.ndarray`: two E2M1 nibbles per byte → float32.
  Nibble layout: `[s][e1][e0][m]`. Low nibble = element 0, high nibble = element 1.
  Unit test: for each of the 16 possible nibble values, verify against the known float
  (hand-computed from E2M1 spec).
- `unpack_float6_e2m3(packed: np.ndarray, n_elements: int) -> np.ndarray`: 32 6-bit
  elements per 24-byte group. Test byte-boundary positions: elements at indices 0, 3, 4, 7,
  15, 16, 31 (the ones that straddle 3-byte boundaries). Unit test: at least 32 hand-
  computed (packed_bytes, expected_float) pairs covering all byte-boundary positions.
- `unpack_bfloat6_e3m2(...)`: same layout, different format constants.

**4.3 — MX GEMM reference in `reference.py`**
- `gemm_mx(A_packed, B_packed, scale_a, scale_b, block_k, dtype_a, dtype_b, scale_type)`:
  for each K-block: unpack operand tiles, decode scales, multiply partial dot-product by
  `scale_a[m, k//block_k] * scale_b[n, k//block_k]`, accumulate in float32. Matches
  `Reference.cpp:1390–1436` and `Reference.cpp:1878–1879`.

**4.4 — Extend `build_kernel_args`** for MX flag branches (`mxBlockA`, `mxBlockB`).

**4.5 — Write `test_gemm_mx.py`**
- MX-Float8 + E8 scale: sizes (256,256,256), (512,512,512).
- MX-Float4 + E8 scale: size (256,256,256).
- MX-Float6 E2M3 and BFloat6 E3M2: size (256,256,256).
- Edge-case: zero scale factor → zero output row/column.
- Edge-case: NaN scale (E8 byte 0xFF) → NaN output.
- Poison-input test for each MX variant.

### Acceptance criteria
- All decoder unit tests pass in pure Python (no GPU) against hand-computed reference values.
- Round-trip tests alone are not accepted as sufficient — reviewer checks for explicit
  byte→float ground-truth assertions.
- All `test_gemm_mx.py` tests pass on gfx950.
- No regressions.

---
