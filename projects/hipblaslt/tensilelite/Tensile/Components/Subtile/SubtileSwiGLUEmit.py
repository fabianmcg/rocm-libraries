# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SwiGLU fused gated-linear-unit epilogue for the Subtile kernel (gfx950, bf16).

Global-split model: B has shape (K, N_gemm = 2*N_out).  Gate cols [0, N_out) and
up cols [N_out, N_gemm) are read into the same wave accumulator by the B-addressing
fix: WG t reads gate from B[:,t*NT_out:(t+1)*NT_out] and up from
B[:,t*NT_out+N_out:(t+1)*NT_out+N_out], where NT_out = MT1/2.  Both halves land in
a single wave's AGPRs with gate in n < mma_n/2 and up in n >= mma_n/2.

This emitter computes y = up * silu(gate) for each (gate, up) AGPR pair and
compacts y into the lower N half of the accumulator so the store path writes
only (M, NT_out) values to D per WG.

Acc VGPR ordering (N-outer, M-inner):
  acc_idx(base, m, n, k) = base + (n*mma_m + m)*4 + k,  k in [0,4)
  gate half: n in [0, mma_n/2); up half: n in [mma_n/2, mma_n)

silu(x) = x * sigmoid(x),  sigmoid(x) = 1/(1+exp(-x)).
v_exp_f32 computes 2^x, so exp(-x) = v_exp_f32(-x * log2(e)).

alpha=1, beta=0 must be passed by the host (hook runs before alpha/beta apply).

Tile-size selection
-------------------
The epilogue holds both the gate and up halves of a tile in registers at once,
so a SwiGLU tile costs roughly the accumulator footprint of a plain GEMM whose
N equals the full N_gemm (2*N_out), not the half-width D output.  The supported
geometry is constrained to keep the AGPR indexing valid and the split on a
sub-tile boundary:
  - 16x16 MFMA only (MatrixInstM == MatrixInstN == 16), 4 output rows per lane.
  - MacroTile1 even (D output width is MacroTile1/2).
  - mma_n = MacroTile1 // 16 // MIWaveGroup[1] must be even and >= 2 so the
    gate/up split falls on an MFMA-N-subtile boundary within each wave.
  - Arbitrary MIWaveGroup[1] (wg_n) and arbitrary N-tile count are supported.
Validated configuration on gfx950: MI 16x16x32 bf16, MIWaveTile [4,4],
MacroTile0 = 64, MacroTile1 = 64*wg_n (N_out = 32*wg_n), DepthU = 64.  Larger
MacroTile1 raises N_out per tile but widens the live accumulator span; MT1 in
{64, 128} is the recommended starting range.

Performance note (deferred)
---------------------------
The epilogue is scalar per element: each (m, n, k) does two AGPR reads, ~6 VALU
ops with two trans-op wait states (v_exp_f32, v_rcp_f32), and one AGPR write,
after a single full s_waitcnt.  This is correctness-first and unoptimized;
vectorization and wait-state hiding are intentionally deferred to a follow-up.
"""
import math

from rocisa.code import Module
from rocisa.container import vgpr, accvgpr
from rocisa.instruction import (
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddF32,
    VExpF32,
    VMulF32,
    VRcpF32,
)

_MFMA_M        = 16
_ROWS_PER_LANE = 4
_LOG2E         = math.log(math.e, 2)


class SubtileSwiGLUEmitter:
    """Emit the fused SwiGLU epilogue for the Subtile gfx950 bf16 kernel."""

    def __init__(self, writer, kernel):
        self.writer   = writer
        self.kernel   = kernel
        self.archCaps = writer.states.archCaps

        assert kernel["MatrixInstM"] == _MFMA_M and kernel["MatrixInstN"] == _MFMA_M, \
            "swiglu emitter assumes a 16x16 mfma"
        assert (kernel["MatrixInstM"] * kernel["MatrixInstN"]) // kernel["WavefrontSize"] \
            == _ROWS_PER_LANE, "swiglu emitter assumes 4 output rows per lane"

        mt0 = kernel["MacroTile0"]
        mt1 = kernel["MacroTile1"]
        wg  = kernel["MIWaveGroup"]
        self.mma_m  = (mt0 // _MFMA_M) // wg[0]
        self.mma_n  = (mt1 // _MFMA_M) // wg[1]
        self.half_n = self.mma_n // 2
        assert self.mma_n >= 2 and self.mma_n % 2 == 0, \
            "swiglu requires an even per-wave N-tile count (mma_n >= 2)"

    def _acc_idx(self, accVgprBase, m, n, k):
        return accVgprBase + (n * self.mma_m + m) * _ROWS_PER_LANE + k

    def emit(self, accVgprBase):
        module = Module("SwiGLU epilogue")
        module.addComment1("SwiGLU: y = up * silu(gate); [gate|up] = split(acc, N)")
        module.addComment0(
            "  mma_m=%d mma_n=%d half_n=%d (gate n in [0,%d), up n in [%d,%d))"
            % (self.mma_m, self.mma_n, self.half_n, self.half_n,
               self.half_n, self.mma_n)
        )

        gate_v = self.writer.vgprPool.checkOut(1, tag="swiglu_gate")
        up_v   = self.writer.vgprPool.checkOut(1, tag="swiglu_up")
        sig_v  = self.writer.vgprPool.checkOut(1, tag="swiglu_sig")

        module.add(SWaitCnt(waitAll=True, comment="flush MFMA pipeline before SwiGLU"))

        for g in range(self.half_n):
            u = g + self.half_n
            module.addComment0("  N-tile pair gate=%d up=%d" % (g, u))
            for m in range(self.mma_m):
                for k in range(_ROWS_PER_LANE):
                    module.add(self._gateOne(accVgprBase, m, g, u, k,
                                             gate_v, up_v, sig_v))

        self.writer.vgprPool.checkIn(sig_v)
        self.writer.vgprPool.checkIn(up_v)
        self.writer.vgprPool.checkIn(gate_v)
        return module

    def _gateOne(self, accVgprBase, m, g, u, k, gate_v, up_v, sig_v):
        module = Module("swiglu elem")
        gate_acc = self._acc_idx(accVgprBase, m, g, k)
        up_acc   = self._acc_idx(accVgprBase, m, u, k)

        module.add(VAccvgprReadB32(vgpr(gate_v), accvgpr(gate_acc),
                                   comment="gate = acc[m=%d,n=%d,k=%d]" % (m, g, k)))
        module.add(VAccvgprReadB32(vgpr(up_v), accvgpr(up_acc),
                                   comment="up = acc[m=%d,n=%d,k=%d]" % (m, u, k)))

        module.add(VMulF32(dst=vgpr(sig_v), src0=-1.0, src1=vgpr(gate_v),
                           comment="-gate"))
        module.add(VMulF32(dst=vgpr(sig_v), src0=_LOG2E, src1=vgpr(sig_v),
                           comment="-gate * log2(e)"))
        module.add(VExpF32(dst=vgpr(sig_v), src=vgpr(sig_v),
                           comment="exp(-gate) = 2^(-gate*log2e)"))
        if self.archCaps.get("TransOpWait"):
            module.add(SNop(waitState=0, comment="1 wait state after v_exp_f32"))
        module.add(VAddF32(dst=vgpr(sig_v), src0=1.0, src1=vgpr(sig_v),
                           comment="1 + exp(-gate)"))
        module.add(VRcpF32(dst=vgpr(sig_v), src=vgpr(sig_v),
                           comment="sigmoid(gate) = 1/(1+exp(-gate))"))
        if self.archCaps.get("TransOpWait"):
            module.add(SNop(waitState=0, comment="1 wait state after v_rcp_f32"))

        module.add(VMulF32(dst=vgpr(sig_v), src0=vgpr(gate_v), src1=vgpr(sig_v),
                           comment="silu(gate) = gate * sigmoid"))
        module.add(VMulF32(dst=vgpr(gate_v), src0=vgpr(up_v), src1=vgpr(sig_v),
                           comment="y = up * silu(gate)"))

        module.add(VAccvgprWriteB32(accvgpr(gate_acc), vgpr(gate_v),
                                    comment="compact y -> acc[m=%d,n=%d,k=%d]" % (m, g, k)))
        return module
