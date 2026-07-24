# GEMM+RMSNorm Performance Analysis: Tensile Pipeline vs PyTorch torch.compile

**Shape:** M=N=K=8192, dtype=bf16, GPU=MI300X (gfx950)

## Measured timings

| Component | PyTorch (`torch.compile`) | Tensile pipeline |
|---|---|---|
| GEMM | 884.9 µs | 1020.7 µs (+15%) |
| RMSNorm | 50.4 µs | 231.5 µs (+4.6×) |
| **Total** | **935.3 µs** | **1252.2 µs (+34%)** |

---

## Root cause 1 — K1 GEMM overhead (+136 µs)

K1 fuses a PartialRMS epilogue into the GEMM kernel (MT0=MT1=256, wg_n=2, 512 threads/block). After the MFMA compute finishes, every wave executes ~1000+ extra instructions sequentially:

1. **`_squareAndLaneSum`**: reads all 128 ACC VGPRs (`VAccvgprReadB32`) to compute per-row Σx². AGPR reads carry non-trivial latency and cannot overlap with anything.

2. **`_butterflyReduce`**: 4 butterfly stages × 16 rows = 64 `DSBPermuteB32` instructions, each followed by `SWaitCnt(dscnt=0)` — **64 full LDS-wait stalls per wave**.

3. **`_crossWaveReduce`** (wg_n=2 → 8 waves must rendezvous): **3 `s_barrier` synchronization points** + 16 `ds_store_b32` + 32 `ds_load_b32` LDS round-trips. Any scheduling skew across the 8 waves stalls the entire block at each barrier.

4. **`_applyGammaOnly`**: 8 gamma loads each followed by `SWaitCnt(vlcnt=0)` — **8 serial VMEM stalls** before the scaled output is written.

The partialBuf writes (~1 MB total) are negligible in cost. All overhead is serialized instruction execution and barrier latency. The 23 extra VGPRs for epilogue temporaries (16 for `partials` + 7 scratch) also reduce occupancy relative to a plain GEMM.

PyTorch's `torch.mm` dispatches a plain Tensile kernel (`Cijk_Ailk_Bljk_BBS_BH_...`) with no epilogue, so none of this overhead applies.

---

## Root cause 2 — `partial_rms_epilogue` poor bandwidth utilization (+181 µs)

### Data volume

| Access | Size |
|---|---|
| Read D (bf16, col-major 8192×8192) | 134 MB |
| Write D (in-place) | 134 MB |
| Read partialBuf (fp32, 8192×32) | 1 MB |
| **Total** | **269 MB** |

**Achieved bandwidth:** 269 MB / 231.5 µs ≈ **1163 GB/s — 22% of MI300X peak (~5300 GB/s).**

### Why only 22% of peak

The bottleneck is Phase 2's column loop in `epilogues/tensilelite/partial_rms_epilogue_generator.py`:

```asm
.Lpartial_rms_epilogue_col:
    global_load_b16         ; 2 bytes/thread = 128 B/wave
    s_waitcnt vmcnt(0)      ; FULL stall — drains ALL pending loads
    ; 3 ALU instructions
    global_store_b16
    s_add_u32   ++col
    s_cmpk_lt / s_cbranch
```

Two compounding problems:

1. **`s_waitcnt vmcnt(0)` after every single load** — zero memory-level parallelism. With ~100–300 ns DRAM latency per cache-miss transaction, 256 fully serialized load→wait→compute→store cycles per block means each block is almost entirely stalled on memory. The hardware has no opportunity to hide latency.

2. **16-bit load granularity** — `global_load_b16` moves 128 B/wave vs 1024 B/wave for `global_load_dwordx4`. This means 8× more loop iterations, 8× more instruction issue slots, and 8× more `vmcnt(0)` stalls for the same data volume.

### Why col-major layout is not the cause

It might seem that col-major D would hurt coalescing, but Phase 2 accesses column `c` for all 256 rows simultaneously. In col-major layout those 256 elements are contiguous in memory — fully coalesced. The layout is not contributing to the slowdown.

### Triton comparison

Triton's `triton_red_fused__fused_rms_norm_0` achieves **269 MB in 50.4 µs ≈ 5340 GB/s (~100% peak)**. The actual compiled TTGIR (8192³ run) confirms the block layout:

```
#blocked = {sizePerThread=[1,8], threadsPerWarp=[1,64], warpsPerCTA=[1,8]}
```

The CTA has 8 warps × 64 threads = 512 threads, all laid out along the N dimension. Each thread owns 8 consecutive bf16 elements, so one CTA covers a full row of 8192 elements (512 × 8 × 2 B = 8 KB) in a single vectorized load (`amdg.buffer_load` with `contiguity=8`).

The `tt.reduce` on `axis=1` lowers to a mix of DPP and LDS at the amdgcn level. There is no `ds_bpermute` anywhere. The full reduction proceeds in four phases:

#### Phase 1 — per-thread serial accumulation (no inter-thread communication)

Each thread holds 16 bf16 elements from two `buffer_load_dwordx4` loads (512 threads × 16 elements = 8192 total). Each thread unpacks its bf16 pairs, squares them with `v_fma_f32 v, v, v, 0`, and accumulates into a single f32 register `v6` using plain `v_add_f32_e32`. No inter-thread communication — purely serial within each thread. After this phase `v6` holds each thread's local sum of squares.

#### Phase 2 — intra-warp reduction via DPP only (64 lanes → 1)

gfx950 wavefronts are 64 lanes wide. DPP `row_shr` operates within 16-lane rows, so the 64-lane warp is treated as 4 rows of 16. No LDS, no barriers — DPP reads a neighbour lane's VGPR directly in the register file.

```asm
v_add_f32_dpp v6, v6, v6 row_shr:8  row_mask:0xf bank_mask:0xf  ; lane[i] += lane[i-8]  (within each row of 16)
v_add_f32_dpp v6, v6, v6 row_shr:4  row_mask:0xf bank_mask:0xf  ; lane[i] += lane[i-4]
v_add_f32_dpp v6, v6, v6 row_shr:2  row_mask:0xf bank_mask:0xf  ; lane[i] += lane[i-2]
v_add_f32_dpp v6, v6, v6 row_shr:1  row_mask:0xf bank_mask:0xf  ; lane[i] += lane[i-1]
; lanes 15, 31, 47, 63 now hold the sum of their respective 16-lane row

v_mov_b32_dpp v7, v7 row_bcast:15  row_mask:0xa bank_mask:0xf   ; broadcast lane 15 into rows 1 and 3
v_add_f32_e32 v6, v7, v6                                         ; rows 1,3 accumulate rows 0+1 and 2+3
v_add_f32_dpp v6, v6, v6 row_bcast:31 row_mask:0xf bank_mask:0xf ; broadcast lane 31 to all rows

v_readlane_b32 s3, v6, 63   ; extract full warp sum from lane 63 into scalar s3
```

After this phase lane 63 (and via `row_bcast:31`, all lanes) holds the warp-level sum.

#### Phase 3 — cross-warp reduction via LDS (8 warp sums → 1)

Lane 0 of each warp (`exec` masked by `vcc = (lane == 0)`) writes its warp sum to LDS, indexed by warp ID:

```asm
ds_write_b32 v5, v6       ; warp leader writes warp sum to LDS[warp_id * 4]
s_waitcnt lgkmcnt(0)
s_barrier                 ; all 8 warps synchronise
ds_read_b32 v6, v5        ; first 8 threads read back: thread i gets LDS[i*4] = warp i's sum
s_waitcnt lgkmcnt(0)
```

Now 8 partial sums sit in `v6` across lanes 0–7. These are reduced with DPP again (no further LDS needed for the tree itself):

```asm
v_mov_b32_dpp v7, v7 row_shr:4       row_mask:0xf bank_mask:0xa   ; lane[i] gets lane[i-4]
v_add_f32_e32 v6, v6, v7             ; pairs: (0+4), (1+5), (2+6), (3+7)
v_mov_b32_dpp v7, v7 quad_perm:[2,3,0,1]                          ; swap pairs within quads
v_add_f32_e32 v6, v6, v7             ; (0+4+2+6), (1+5+3+7)
v_mov_b32_dpp v7, v7 quad_perm:[1,0,3,2]                          ; swap adjacent lanes
; only lane 0 executes (exec masked):
v_add_f32_e32 v0, v6, v7             ; final grand total in lane 0
ds_write_b32 v5, v0                  ; write final sum to LDS[0]
```

#### Phase 4 — broadcast final result to all threads

```asm
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v0, v0    ; ALL 512 threads read LDS[0] — the grand total
```

#### Summary table

| Phase | Threads involved | Mechanism | LDS? |
|---|---|---|---|
| Per-thread accumulation (16 elements → 1 f32) | All 512, independently | Serial `v_add_f32_e32` | No |
| Intra-warp tree (64 lanes → 1) | Per warp | DPP `row_shr:8/4/2/1` + `row_bcast:15/31` | No |
| Cross-warp round 1 (8 warp sums → LDS) | 1 lane per warp | `ds_write_b32` + `s_barrier` + `ds_read_b32` | Yes |
| Cross-warp round 2 (8 LDS values → 1, DPP) | 8 threads | DPP `row_shr:4` + `quad_perm:[2,3,0,1]` + `quad_perm:[1,0,3,2]` | No |
| Final broadcast to all threads | All 512 | `ds_write_b32` (lane 0) + `s_barrier` + `ds_read_b32` (all) | Yes |

Total LDS operations: **2 `ds_write` + 2 `ds_read` + 2 `s_barrier`**. No `ds_bpermute` is used anywhere — all inter-lane data movement is via DPP `v_mov_b32_dpp` / `v_add_f32_dpp`.

The kernel makes **two passes** over the row — once to compute Σx² (line 28 in ttgir: `amdg.buffer_load`), then again to apply the normalization (line 46: second `amdg.buffer_load`) — with a `ttg.barrier` between them (line 42) to write the rsqrt scalar. So D is read twice, written once: **3 × 134 MB = 402 MB** total. At 50.4 µs that's ~7976 GB/s, which exceeds the nominal HBM peak — this is likely L2-cache assisted for the second read since both passes touch the same 134 MB within the same kernel invocation.

There is no partial accumulation stage. The `partial_rms_epilogue` kernel, by contrast, must read partialBuf (produced by K1) and then traverse D column-by-column to apply the normalization — the second traversal is where bandwidth is lost due to the `vmcnt(0)` per 16-bit load.

---

## Timing methodology

The `reset_buffers()` memsets (134 MB D + 1 MB partialBuf) are correctly excluded from the timed region. `hip_event_record(ev0, Ptr(0))` is submitted to the null stream after the memsets complete, so the `ev0→ev2` window is clean and contains only K1 and `partial_rms_epilogue`.

---

## Summary

| Component | Time | Root cause |
|---|---|---|
| K1 (GEMM+PartialRMS) | 1020.7 µs | ~1000 extra instructions/wave after MFMA: 64 LDS-wait stalls (butterfly), 3 cross-wave barriers, 8 gamma vlcnt stalls; reduced occupancy from 23 extra VGPRs |
| `partial_rms_epilogue` | 231.5 µs | Column-serial loop with `vmcnt(0)` fence per 16-bit load → zero memory-level parallelism → 22% of peak HBM bandwidth |
| Triton `fused_rms_norm` | 50.4 µs | Single-pass, 128-bit vectorized loads, warp-shuffle reduction, fully pipelined → ~100% HBM bandwidth |

## Suggested fixes

- **`partial_rms_epilogue`** (dominant gap): replace `global_load_b16` + `vmcnt(0)` per column with `global_load_dwordx4` (128-bit) and issue multiple loads before waiting. This recovers memory-level parallelism and should bring bandwidth close to peak.
- **K1 epilogue overhead**: the butterfly+barrier+gamma-load chain is inherent to the current fused design. Reducing `wg_n` to 1 eliminates the cross-wave barrier stage (`_crossWaveReduce`) at the cost of a wider partialBuf, which may reduce K1 time at the expense of a slightly larger `partial_rms_epilogue` input.
