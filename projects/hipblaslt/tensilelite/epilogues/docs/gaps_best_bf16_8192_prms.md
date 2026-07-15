# Gaps: best_bf16_8192 → PartialRMS support

Tracks what must be addressed to run
`Tensile/Tests/common/gemm/gfx950/best_bf16_8192.yaml`
with `UsePartialRMS: True` / `PartialRMS: True`.

Reference config: MT320×320, DU=64, MIWT 10×10, WG 2×2,
PGR=1, PLR=1, StreamK=3, TransposeLDS=1, WGM=16, XCC=4, 1323.5 TFLOPS.

---

## Fixed

### MT multiple-of-64 constraint (commit b746b9de)

The validator previously rejected any tile where MacroTile0 or MacroTile1
was not a power of two. The actual requirement is a multiple of 64 — the
butterfly reduction only requires `MIWaveGroup[0]` to be a power of two
(enforced separately). The check was relaxed accordingly, unblocking the
MT320×320 tile.

---

## Open gaps

### Gap 1 — Missing solution/problem-type parameters in the tuning YAML

Not a code gap; requires adding fields to the benchmark YAML.

| Parameter | Source YAML | Required for PartialRMS |
|---|---|---|
| `UsePartialRMS` (ProblemType) | absent | `True` |
| `PartialRMSResidualAdd` (ProblemType) | absent | `False` |
| `PartialRMS` (solution ForkParam) | absent | `[True]` |
| `StreamKForceDPOnly` (solution ForkParam) | absent | `[1]` |

**Action:** add these four fields when deriving a PartialRMS tuning YAML
from best_bf16_8192. See `epilogues/yaml/tune_subtile_bf16_prms_8192.yaml`
for the pattern.

---

### Gap 2 — ProblemType epilogue conflicts

best_bf16_8192 uses:
- `UseBias: 1` / `BiasDataTypeList: [b]`
- `UseScaleAlphaVec: 1`
- `Activation: True` / `ActivationType: hipblaslt_all`

The PartialRMS validator does **not** explicitly reject these combinations,
but the combined kernarg layout (bias + scale-alpha-vec + activation args
followed by the PartialRMS args `RMSNormGamma` + `PartialBuf`) has never
been validated end-to-end. No test or client code exercises
`UseBias=1 + UsePartialRMS=1` simultaneously.

**Options:**

1. **Drop for tuning** (recommended short-term): set `UseBias: 0`,
   `UseScaleAlphaVec: 0`, `Activation: False` in the PartialRMS variant.
   The epilogue stands alone; bias/activation are a separate concern.

2. **Validate the combined layout**: audit `Tensile/Components/Signature.py`
   and the client's `UserArgumentsInfo` paths to confirm the kernarg offsets
   are consistent when all three are active, then add an e2e test.

---

### Gap 3 — `TransposeLDS: 1` with PartialRMS untested

best_bf16_8192 sets `TransposeLDS: 1`. The existing PartialRMS YAML configs
do not fork over this parameter. The validator does not reject it. Whether
`TransposeLDS=1 + PartialRMS=1` produces correct code (no AGPR layout
mismatch in the emitter) has not been exercised.

**Action:** run a targeted e2e test at a small shape (e.g. 512×512×1×512)
with `TransposeLDS=1` and `PartialRMS=1` before trusting any benchmark
results from that combination.

---

## LDS budget — not a gap

For `[10,10,2,2]` (MT320×320, wg_m=2, wg_n=2):

```
mma_n    = MT1 / mfma_n / wg_n = 320 / 16 / 2 = 10
LDS scratch = wg_m * wg_n * WF * mma_n * 4 B
            = 2 * 2 * 64 * 10 * 4 = 10 240 B
MaxLDS = 65 536 B  →  passes
```

The validator checks this at runtime; no code change needed.
