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
- `decode_e5m3(scale_bytes: np.ndarray) -> np.ndarray`: E5M3 is an **unsigned** 8-bit float
  (**no sign bit**): bits `[e4 e3 e2 e1 e0][m2 m1 m0]` — 5 exponent bits + 3 mantissa bits,
  exponent **bias 15**. `data == 0x00` → 0.0; `data == 0xFF` → NaN. Subnormals are supported
  (exponent field 0, nonzero mantissa). This is NOT E5M2 and has no sign bit — every E5M3
  value is non-negative, correct for a block scale factor. Verified against
  `tensilelite/include/Tensile/DataTypes_E5M3.hpp` (`struct E5M3` + `cast_from_uf8(data,
  wm=3, we=5)`, which uses "uf8" = unsigned f8 and forces `sign = 0`). Decode:
  `exp = data >> 3`, `mant = data & 0x7`; for `exp != 0`, `value = 2**(exp-15) * (1 + mant/8)`;
  for `exp == 0`, `value = 2**(1-15) * (mant/8)` (subnormal). ~10–15 lines of numpy bit
  manipulation.
- Unit tests: for each decoder, test at least 10 hand-computed (byte → float) pairs
  including 0x00, 0x7F, 0x80, 0xFE, 0xFF. Round-trip tests are insufficient on their own.

**4.2 — Float4/6/BFloat6 unpackers**
- `unpack_float4(packed: np.ndarray) -> np.ndarray`: two E2M1 nibbles per byte → float32.
  Each nibble is `[s][e1][e0][m]` (1 sign, 2 exponent, 1 mantissa; bias 1). Assumed packing:
  low nibble = element 0, high nibble = element 1.
  **The nibble order (which nibble is element 0) MUST be verified before writing tests — do
  not assume it.** Cross-check against the AMD MX / OCP MXFP4 specification AND against the
  C++ unpacking in `client/src/Reference.cpp:75–95` (`loadPackedFloat4To`), which converts a
  packed `Float4x2` word via the hardware intrinsic
  `__amd_cvt_fp4x2_to_floatx2_scale(word.data, __AMD_OCP_E2M1, 0)` and stores `v.x` to element
  `2*word+0` and `v.y` to element `2*word+1`. That intrinsic (not visible bit-math) defines
  which nibble maps to `v.x` vs `v.y`; validate a byte round-trip through both paths before
  relying on the Python bit extraction. Verify these line numbers against the current source.
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

**4.4 — Extend `build_kernel_args`** for MX flag branches.
The MX-specific argument slots in `singleCallArgs` are gated by the compile-time flags
`problemType.mxBlockA` / `problemType.mxBlockB` (these are NOT emitted as runtime arg values;
they only decide whether the following slots appear). When set, they add, in order:
- `mxsa` — the MX scale-A device pointer (`inputs.mxsa`), appended right after the `a`
  pointer (`ContractionSolution.cpp:635–636`, inside the `stridedBatched` branch).
- `mxsb` — the MX scale-B device pointer (`inputs.mxsb`), appended right after the `b`
  pointer (`ContractionSolution.cpp:639–640`).
- `strideMXSA` — one `uint32_t` per MX-scale-A stride dimension
  (`ContractionSolution.cpp:709–711`).
- `strideMXSB` — one `uint32_t` per MX-scale-B stride dimension
  (`ContractionSolution.cpp:719–721`).
Port exactly these conditional appends, preserving order relative to the surrounding operand
and stride slots. Verify all line numbers against the current `ContractionSolution.cpp`
(grep for `mxsa`/`strideMXSA`) before use. Note: MX scale tensors must be E8 (asserted in the
reference at `Reference.cpp:1232–1235`).

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
