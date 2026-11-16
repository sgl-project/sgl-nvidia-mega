# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KDA cores at the K3.1 TP8 per-rank shapes (12 heads x 128, packed projection [T, 6288],
conv BF16 [S, 3, 4608], FP32 state [S, 12, 128, 128], FP32 conv weights; ``K31_TEST_HEADS=6``: the TP16
shapes, 6 heads, [T, 3216], conv [S, 3, 2304]):

- decode ``kda.prepared.PreparedKda`` with the per-batch gate modes and schedules of
  ``providers.build_resources`` (B1 mode 1, B2-8 mode 2, B16/32/64 mode 0 with the staged
  early-3 schedules), padding rows included;
- ReplaySSM TARGET_VERIFY ``kimi_k31.verify_kda.PreparedKdaVerify`` (the K31 adapter over
  ``kda.verify.PreparedKdaVerify``) at Q8 with Breq 1/2/4 (mode 0, as ``verify_spec``),
  one padded request at Breq 4;

against the independent FP32 torch reference ``_ref_attnkda`` (stock ``kda_fused_decode``
math and the ReplaySSM ring contract).

Tolerances (stated before any run):
- output: |out - ref| <= 0.02 + 0.02 |ref| (the K3.1 multireq KDA gate, atol/rtol .02/.02);
- new FP32 state: |s - ref| <= 1e-3 + 1e-3 |ref| (one step of FP32 math with fast
  exp/rcp/rsqrt differs by ~1e-5 relative);
- conv window / conv state: bit-exact (pure BF16 copies);
- rings: raw K/V |x - ref| <= 1e-3 + 2^-7 |ref| (one BF16 rounding of an FP32 value that
  differs by ~1e-6); log-g and beta |x - ref| <= 1e-4 + 1e-4 |ref|;
- padding rows: output exactly 0; their slots' state/conv untouched; verify leaves the
  persistent state and conv bit-identical.

Mutations (each must make the suite fail): ``no_decay`` (state not decayed), ``no_beta``
(delta-rule update without beta), ``conv_shift`` (conv state keeps the oldest tap instead of
shifting), ``ring_k`` (the raw-K ring stores q).
"""

import logging
import os

import pytest
import torch

import _harness as H
import _ref_attnkda as R

MUTATIONS = {
    "no_decay": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "history[p, j, e] *= decays[j][e]",
        "history[p, j, e] *= Float32(1.0)",
    ),
    "no_beta": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "v_new = (shared_v[safe_row] - dots[p]) * beta",
        "v_new = shared_v[safe_row] - dots[p]",
    ),
    "conv_shift": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                        taps[2, 1],\n                        taps[2, 2],",
        "                        taps[2, 0],\n                        taps[2, 2],",
    ),
    "ring_k": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "args[20][Int64(slot), head, gt, ch] = BFloat16(k)",
        "args[20][Int64(slot), head, gt, ch] = BFloat16(q)",
    ),
    # Opt-in "wide" verify (run with K31MK_KDA_VERIFY_OPT=wide): each must fail it.
    "wide_tap_row": (  # second half's conv taps one token late
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "row = token + half * TH - 3 + t",
        "row = token + half * TH - 2 + t",
    ),
    "wide_beta_slot": (  # betas staged at the half-local token index
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "shared_beta[gt] = betas[t]",
        "shared_beta[t] = betas[t]",
    ),
    "wide_norm_slot": (  # q/k norm partials of half 1 written over half 0's
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "reductions[warp % 4 + 8 * gu] = squares[2 * u]",
        "reductions[warp % 4 + 8 * u] = squares[2 * u]",
    ),
    "pk_no_decay": (  # opt "pk": packed decay multiplies by 1
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                            (decays[j][e], decays[j][e + 1]),",
        "                            (Float32(1.0), Float32(1.0)),",
    ),
    "rs_index": (  # opt "rs": q and k sums of squares swapped
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "which, gu = index % 2, index // 2",
        "which, gu = (index + 1) % 2, index // 2",
    ),
    "la_no_carry": (  # opt "la": look-ahead k-dot without the vn_{t-1} c_{t-1} term
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "kd[p] = fma(vn_prev[p], c_prev, a_now[p], ftz=self.fast_math)",
        "kd[p] = a_now[p]",
    ),
    # opt "wlf" (KDA17.2; test_kda_verify_wlf): each must fail it.
    "wlf_pass_row": (  # every pass reads pass 0's row of v (only 2+ rows per thread differ)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "safe_row = (p * self.groups + group) % self.rows\n            v_new = (shared_v[safe_row] - dots[p]) * beta",
        "safe_row = group % self.rows\n            v_new = (shared_v[safe_row] - dots[p]) * beta",
    ),
    "wlf_off": (  # the floor never engages (wide's 4 lanes at 64 rows)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        'floor = self.lanes_per_row if "wlf" in self.opt else 1',
        "floor = 1",
    ),
    "wlf32_entry": (  # with wlf, a 2-CTA grid over one wave keeps the 2-CTA clusters (WLF_ONE_WAVE never chosen)
        "kimi_k31/k3/cutedsl/kda/verify.py",
        'if "wlf" in opt and heads is not None and schedule == (0, 8, 4, 2):',
        "if False:",
    ),
    "wlf32_no_opt": (  # the prepare call does not pass the opts, so the schedule never sees "wlf"
        "kimi_k31/k3/cutedsl/kda/verify.py",
        "verify_schedule(requests, queries, self.device, verify_opt(), h),",
        "verify_schedule(requests, queries, self.device, frozenset(), h),",
    ),
    "wave_ctas": (  # the one-wave test counts 64-row CTAs per head as 1, not 2 (TP8 Breq 16 / TP16 Breq 32 stay 2-CTA)
        "kimi_k31/k3/cutedsl/kda/verify.py",
        "if 2 * heads * requests > torch.cuda.get_device_properties(device).multi_processor_count:",
        "if heads * requests > torch.cuda.get_device_properties(device).multi_processor_count:",
    ),
    "wave_no_heads": (  # the prepare call does not pass the heads, so WLF_ONE_WAVE is never chosen
        "kimi_k31/k3/cutedsl/kda/verify.py",
        "verify_schedule(requests, queries, self.device, verify_opt(), h),",
        "verify_schedule(requests, queries, self.device, verify_opt()),",
    ),
    # Timing-only ablation (outputs garbage, never a gate): the recurrence math removed,
    # pushes kept so the output barriers still complete.
    "abl_noloop": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                            self.update(\n                                history, dots, current, lanes, (start, stop), source\n                            )",
        "                            dots.fill(0.0)",
    ),
    "gm1_drop_chain": (  # opt "gm1": the multicast's F_B sum drops chain 3 (K 96-127)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "value = Float32(0) + (((value + parts[1]) + parts[2]) + parts[3])\n"
        "                raw_g = Float32(BFloat16(value))\n"
        "                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])\n"
        "                gate = self.activation(gate)\n"
        "                log_decay = lower_bound * gate\n"
        "                decay = float_bits(",
        "value = Float32(0) + ((value + parts[1]) + parts[2])\n"
        "                raw_g = Float32(BFloat16(value))\n"
        "                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])\n"
        "                gate = self.activation(gate)\n"
        "                log_decay = lower_bound * gate\n"
        "                decay = float_bits(",
    ),
    "ctl_early_cw_first": [  # sanitizer control, not a bug: early's main-warp cluster wait moved before the gate wait
        (  # re-anchored: the line now skips operands(0) under opt "chunk"
            "kimi_k31/k3/cutedsl/kda/kernel.py",
            "                ready = None if self.chunked else operands(0)\n",
            "                if cutlass.const_expr(self.early and not self.wy):\n                    cute.arch.cluster_wait()\n"
            "                ready = None if self.chunked else operands(0)\n",
        ),
        (  # re-anchored: the wait now sits under a "not self.tsp" guard (same effect)
            "kimi_k31/k3/cutedsl/kda/kernel.py",
            "                        if cutlass.const_expr(self.early and start == 0 and t == 0):  # noqa: SIM102\n",
            "                        if cutlass.const_expr(False):  # noqa: SIM102\n",
        ),
        (  # opt "wy": its cluster wait (before its first DSMEM send) moves before its gate wait
            "kimi_k31/k3/cutedsl/kda/kernel.py",
            "        wait(gate_barrier)\n        if thread < 128:\n",
            "        if cutlass.const_expr(self.early):\n            cute.arch.cluster_wait()\n"
            "        wait(gate_barrier)\n        if thread < 128:\n",
        ),
        (
            "kimi_k31/k3/cutedsl/kda/kernel.py",
            "        if cutlass.const_expr(self.early):  # before this CTA's first DSMEM send\n"
            "            cute.arch.cluster_wait()\n        # P_t (t < T)",
            "        # P_t (t < T)",
        ),
    ],
    "abl_nomma": (  # timing-only: gate warp's F_B MMA chain (and so its f_a loads) removed
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                    mma_into(acc[chain, tile, None], wa, ar[chunk], bf16=True)\n            if cutlass.const_expr(self.tokens > 1):",
        "                    pass\n            if cutlass.const_expr(self.tokens > 1):",
    ),
    "dmc_packet_quad": (  # opt "dmc": packets sent to the next quad's rows
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                feature = value_base + 4 * quad\n",
        "                feature = value_base + 4 * ((quad + 1) % per_token)\n",
    ),
    "rs_half_slot": (  # opt "rs" (live under wide+rs+pre): half-1 norm totals into half 0's slots
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                                gu = gu + half * TH\n",
        "                                gu = gu\n",
    ),
    "wide_owner_half": (  # output norm of every token by half 0 (wrong gate registers)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "owner = owner & (half == t // TH)",
        "owner = owner & (half == 0)",
    ),
    # Opt "chunk": each must fail test_kda_verify_chunk.
    "chunk_ratio_self": (  # E_st includes d_s (one decay too many on every intra-chunk term)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "later = t > source",
        "later = t >= source",
    ),
    "chunk_no_gram": (  # k-dot loses the earlier tokens' updates (A_ts dropped)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "acc[t] = fma(g[t], u, acc[t], ftz=self.fast_math)",
        "acc[t] = acc[t]",
    ),
    "chunk_no_diag": (  # o_t misses its own token's update (B_tt dropped)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "for t in cutlass.range_constexpr(s, T):",
        "for t in cutlass.range_constexpr(s + 1, T):",
    ),
    "chunk_prefix_d0": (  # decay prefixes start after token 0's decay
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "prefix[j, e] = d4[e] if t == 0 else prefix[j, e] * d4[e]",
        "prefix[j, e] = Float32(1.0) if t == 0 else prefix[j, e] * d4[e]",
    ),
    "chunk_no_beta": (  # delta-rule update without beta
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "u = (shared_v[self.rows * s + group] - acc[s]) * shared_beta[s]",
        "u = shared_v[self.rows * s + group] - acc[s]",
    ),
    "chunk_scatter_slot": (  # reduce-scatter totals stored at the lane-local index (all lanes on slot 0..)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "products[W * group + selected + i] = sums[i]",
        "products[W * group + i] = sums[i]",
    ),
    # Bitwise knobs: each must fail test_kda_verify_bitwise_opts.
    "lean_gate_row": (  # opt "lean": the owned gate read from the next token's row
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "gate_owned = gate[token + partition, qk_channel]",
        "gate_owned = gate[token + (partition + 1) % 8, qk_channel]",
    ),
    "lean_beta_raw": (  # opt "lean": staged betas not activated
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                            shared_beta[gt] = beta_act",
        "                            shared_beta[gt] = Float32(betas[t])",
    ),
    "bal_none": (  # opt "bal": no CTA writes the q / k side rings and windows
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "qk_writer = partition == gt % self.cluster",
        "qk_writer = partition == gt % self.cluster + self.cluster",
    ),
    # Opt "tsp" (token-split phase 1): each must fail test_kda_verify_tsp.
    "tsp_factor_scale": (  # q not scaled by 128^-0.5 (k's factor used for q)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "            factor[0] = factor[0] * scale",
        "            factor[0] = factor[0]",
    ),
    "tsp_packet_lane": (  # channels rotated inside each 16-byte q / k packet
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "cute.arch.shuffle_sync(normalized, (lane & 28) + e)",
        "cute.arch.shuffle_sync(normalized, (lane & 28) + (e + 1) % 4)",
    ),
    "tsp_rawk_section": (  # raw-K ring row written by the q threads
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "        if section == 1:\n            args[20][Int64(slot), head, partition, ch] = BFloat16(qk)",
        "        if section == 0:\n            args[20][Int64(slot), head, partition, ch] = BFloat16(qk)",
    ),
    "tsp_window_entry": (  # q / k conv window entries one tap early
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "            args[18][Int64(scratch_row), partition, column, i] = window[i + 1]",
        "            args[18][Int64(scratch_row), partition, column, i] = window[i]",
    ),
    "tsp_v_slot": (  # shared v stored token-major instead of row-major
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "            shared_v[v_row + rows * v_token] = v",
        "            shared_v[v_token + T * v_row] = v",
    ),
    "tsp_beta_raw": (  # staged betas not activated
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "            shared_beta[thread - 128] = beta_act",
        "            shared_beta[thread - 128] = Float32(beta_raw[0])",
    ),
    "tsp_no_wait": (  # sync: readers skip the q / k barrier (read before the peers' packets land)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "        wait(qk_barrier)  # every token's normalized q / k rows (acquire.cluster)",
        "        pass  # (mutant: no wait on the q / k barrier)",
    ),
    "tsp_short_tx": (  # sync: barrier armed for 7 of 8 senders' bytes (phase completes one sender early)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "armed = ((qk_barrier, True, 1024 * T),)",
        "armed = ((qk_barrier, True, 1024 * T - 1024),)",
    ),
    "tsp_tap_cut": (  # conv tap 2 of tokens 0-2 never loaded (stays 0)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "            if partition + i < 3:\n                window[i] = conv_page[column, partition + i]",
        "            if partition + i < 2:\n                window[i] = conv_page[column, partition + i]",
    ),
    # Merged-string sync probes (reported, never part of a VALID set): with tsp + chunk the main warps
    # wait on the cluster barrier twice after one arrive; cg's gate-warp rendezvous back on tsp's barrier 2.
    "m1_double_wait": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "if cutlass.const_expr(self.early and not self.tsp):  # before this CTA's first DSMEM send (tsp: waited)",
        "if cutlass.const_expr(self.early):  # MERGE-1 probe: second wait under tsp",
    ),
    "m1_cg_barrier2": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "CG_BARRIER = 4",
        "CG_BARRIER = 2",
    ),
    # FX-5 / FX-8 caps: each must fail test_kda_verify_m1_engagement.
    "fx5_uncap": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        'self.dmc = "dmc" in self.opt and batch <= (B3_MAX_REQUESTS if b3w or b3c else DMC_MAX_REQUESTS)',
        'self.dmc = "dmc" in self.opt',
    ),
    "fx8_uncap": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        'self.chunk = "chunk" in self.opt and batch <= (B3_MAX_REQUESTS if b3c else CHUNK_MAX_REQUESTS)',
        'self.chunk = "chunk" in self.opt',
    ),
    "k23_tx_double": (  # K-23 knob path: setup() arms twice the bytes the peers send (phase never completes)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "                    arm_cluster(bar, size)\n",
        "                    arm_cluster(bar, 2 * size)\n",
    ),
    "fx58_overcap": (  # caps at 0: Breq 1 / 2 lose dmc and the chunk family (bitwise for dmc: only engagement sees it)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "DMC_MAX_REQUESTS = CHUNK_MAX_REQUESTS = TSP_MAX_REQUESTS",
        "DMC_MAX_REQUESTS = CHUNK_MAX_REQUESTS = 0",
    ),
    # Opts "cg" / "cs": each must fail test_kda_verify_chunk2.
    "cg_span_self": (  # opt "cg": the gate warp's E_st includes d_s (one decay too many)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "after = t > own",
        "after = t >= own",
    ),
    "cg_entry_slot": (  # opt "cg": B entries sent onto the A slots (the B slots never written)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "destination = gram.iterator + (W * own + index)",
        "destination = gram.iterator + (W * own + index % T)",
    ),
    "cg_solve_no_a": (  # opts "cg" / "cs" solver: k-dot loses the earlier tokens' updates (A_ts dropped)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "pending[t] = fma(coef[t], u, pending[t], ftz=self.fast_math)",
        "pending[t] = pending[t]",
    ),
    "cs_row_swap": (  # opt "cs": the remap gives each lane its row pair swapped
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "row = 2 * (lane // 4) + (h ^ ((lane >> 1) & 1))",
        "row = 2 * (lane // 4) + (h ^ ((lane >> 1) & 1) ^ 1)",
    ),
    "cs_prefix_d0": (  # opt "cs": decay prefixes start after token 0's decay
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "running[e] = d4[e] if t == 0 else running[e] * d4[e]",
        "running[e] = Float32(1.0) if t == 0 else running[e] * d4[e]",
    ),
    "cs_partial_slot": (  # opt "cs": the second row of each pair stores its partials over the first's
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "first = 16 * W * warp + 2 * W * (lane // 4) + picked",
        "first = 16 * W * warp + 2 * W * (lane // 4) + picked % 16",
    ),
    "cs_warp_drop": (  # opt "cs": key columns 112..127 (warp 7) dropped, warp 6 counted twice
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "(parts[6] + parts[7])",
        "(parts[6] + parts[6])",
    ),
    "csp_pair_swap": (  # opt "csp": the second column pair's keys / queries swapped within the pair
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "acc2 = fma2_ftz(high, (v4[2], v4[3]), acc2, ftz=ftz)",
        "acc2 = fma2_ftz(high, (v4[3], v4[2]), acc2, ftz=ftz)",
    ),
    "gb_half_drop": (  # opt "gb": the second column half's Gram parts dropped (the first counted twice)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "gram[thread] = gram_parts[thread] + gram_parts[W * T + thread]",
        "gram[thread] = gram_parts[thread] + gram_parts[thread]",
    ),
    "gb_slot_a": (  # opt "gb": A entries stored one token early
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "slot = row + 1 + (k - count_b) if k >= count_b else T + row + k",
        "slot = row + (k - count_b) if k >= count_b else T + row + k",
    ),
    "gb_b_keys": (  # opt "gb": B entries dot keys instead of queries
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "entries[slot[\"B\", s, t]] = dot(weighted, q_t, ftz=fz)",
        "entries[slot[\"B\", s, t]] = dot(weighted, k_t, ftz=fz)",
    ),
    "gb_row_split": (  # opt "gb": the first entry of row b mapped to row a
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "in_b = index >= first_two",
        "in_b = index > first_two",
    ),
    "cs_solver_row": (  # opt "cs": the solver lane reads its neighbour row's P/R
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "row_products = row_products + 20 * thread",
        "row_products = row_products + 20 * (thread ^ 1)",
    ),
    # Must FAIL test_kda_gate_boundary_control (K31_KDA_GATE_DIAG=1): the verify gate input a is not rounded
    # to BF16, so log_g misses are no longer BF16 boundary flips of a.
    "a_round": (  # re-anchored: the plain anchor also matched multicast_packets (2x)
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "raw_g = Float32(BFloat16(value))\n                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])\n"
        "                gate = self.activation(gate)\n                log_decay = lower_bound * gate\n"
        "                decay = float_bits(",
        "raw_g = value\n                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])\n"
        "                gate = self.activation(gate)\n                log_decay = lower_bound * gate\n"
        "                decay = float_bits(",
    ),
    # MKB-0: must FAIL test_kda_verify_mkb on SM107 (the Breq 8 / 16 entries dropped: 8-CTA default).
    "mkb_no_entry": (
        "kimi_k31/k3/cutedsl/kda/verify.py",
        "(4, 8): (0, 8, 2, 4), (8, 8): (0, 8, 4, 2), (16, 8): (0, 8, 4, 2), (32, 8): (0, 8, 4, 2)}",
        "(4, 8): (0, 8, 2, 4)}",
    ),
    # tsp at 12 heads. tsp_fastmath = the old gate (tsp engages at fast math and asserts at build):
    # must FAIL test_kda_verify_tsp_heads12 (and test_kda_verify_wlf[1] at 12 heads); tsp_off_all (tsp off at every head
    # count) must FAIL test_kda_verify_tsp at 6 heads; sched_log_cluster (the schedule log line prints the CTAs per SM as
    # the cluster) must FAIL test_kda_verify_tsp_heads12's log check.
    "tsp_fastmath": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "self.tsp = self.tsp and batch <= (B3_MAX_REQUESTS if b3w or b3c else TSP_MAX_REQUESTS) and not self.fast_math",
        "self.tsp = self.tsp and batch <= (B3_MAX_REQUESTS if b3w or b3c else TSP_MAX_REQUESTS)",
    ),
    "tsp_off_all": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "self.tsp = self.tsp and batch <= (B3_MAX_REQUESTS if b3w or b3c else TSP_MAX_REQUESTS) and not self.fast_math",
        "self.tsp = self.tsp and batch <= (B3_MAX_REQUESTS if b3w or b3c else TSP_MAX_REQUESTS) and False",
    ),
    "sched_log_cluster": (
        "kimi_k31/k3/cutedsl/kda/verify.py",
        "requests, queries, heads, kernel.cluster, kernel.occupancy,",
        "requests, queries, heads, kernel.occupancy, kernel.occupancy,",
    ),
    # L3: must FAIL test_kda_verify_l3 on SM107 (Breq 3 / 5-7 keep GB300's 8-CTA default instead of the
    # schedule of the Breq 4 / 8 graph they were padded to).
    "l3_kda_pad": (
        "kimi_k31/k3/cutedsl/kda/verify.py",
        "padded = {3: 4, 5: 8, 6: 8, 7: 8}.get(requests, requests) if queries == 8 else requests",
        "padded = requests",
    ),
    #  (wide tier): must FAIL test_kda_verify_wide on SM107 (Breq 9-15 drop to GB300's 8-CTA default
    # instead of the schedule of the Breq 16 graph).
    "wide_kda_pad": (
        "kimi_k31/k3/cutedsl/kda/verify.py",
        "        padded = 16 if queries == 8 and 8 < requests < 16 else padded\n",
        "",
    ),
    # Opts "b3w" / "b3c" (engagement mutants; the kb3 checks engagement rows must flag each):
    # b3_nosched: Breq 3 keeps the padded 4-CTA schedule (tsp cannot engage); b3_cap4: the caps reach Breq 4 (dmc engages there).
    "b3_nosched": (
        "kimi_k31/k3/cutedsl/kda/verify.py",
        '        if requests == 3 and queries == 8 and ("b3w" in opt or "b3c" in opt):\n            return B3_8CTA\n',
        "",
    ),
    "b3_cap4": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "B3_MAX_REQUESTS = 3\n",
        "B3_MAX_REQUESTS = 4\n",
    ),
}
# Opt "wy" (KdaSplit.wy_outputs) mutants, each one plausible slip; test_kda_verify_wy must fail on each.
_WYK = "kimi_k31/k3/cutedsl/kda/kernel.py"
MUTATIONS.update({
    "wy_diag_b": (_WYK, "                for t in cutlass.range_constexpr(s, T):\n                    out[t] = fma(g[T + t], u, out[t], ftz=fz)",
                  "                for t in cutlass.range_constexpr(s + 1, T):\n                    out[t] = fma(g[T + t], u, out[t], ftz=fz)"),
    "wy_basis_lane": (_WYK, "(SLAB * m + (j * 4 + e) * L + group_lane)", "(SLAB * m + (j * 4 + e) * L + (group_lane ^ 1))"),
    "wy_prefix_d0": (_WYK, "            cumulative = shared_decay[thread]\n", "            cumulative = Float32(1.0)\n"),
    "wy_stale_row": (_WYK, "                    h0 = history[p, j, e]", "                    h0 = history[0, j, e]"),
    "wy_scatter_idx": (_WYK, "                products[W * row + selected + i] = sums[i]", "                products[W * row + i] = sums[i]"),
    "wy_gram_late": (_WYK, "                    later = t > source\n                    for e in cutlass.range_constexpr(4):\n"
                     "                        ratio[e] = ratio[e] * d4[e] if later else ratio[e]\n                    scaled",
                     "                    later = t >= source\n                    for e in cutlass.range_constexpr(4):\n"
                     "                        ratio[e] = ratio[e] * d4[e] if later else ratio[e]\n                    scaled"),
})
# Sanitizer positive controls for "wy" (bugs, with the racecheck control's edits); racecheck must report hazards on each.
# A short token-barrier count is not usable as one: each solver warp's arrive is warp-aggregated (count 32), so a count
# below rows completes several phases per launch and the parity-0 wait hangs instead of racing.
# san_pos_wy_arrive_first: the one-CTA solver arrives on token t's barrier before storing o_t (count stays rows: one
# completion), so the readers' loads of shared_o are unordered with that store.
MUTATIONS["san_pos_wy_arrive_first"] = MUTATIONS["ctl_early_cw_first"] + [(
    _WYK,
    "                    put_float(dst, out[t])\n                    arrive_release_cta(advance(barrier, t))\n",
    "                    arrive_release_cta(advance(barrier, t))\n                    put_float(dst, out[t])\n",
)]
# san_pos_wy_nobar1: the barrier after the bases / Gram rows is dropped, so warps 4-7 read bases that threads 0-127
# may not have stored yet (the second barrier keeps every thread's barrier count equal: no deadlock).
MUTATIONS["san_pos_wy_nobar1"] = MUTATIONS["ctl_early_cw_first"] + [(
    _WYK,
    "        cute.arch.barrier(barrier_id=1, number_of_threads=MAIN)\n"
    "        if cutlass.const_expr(self.early and not self.tsp2):  # before this CTA's first DSMEM send (tsp2: waited)\n",
    "        if cutlass.const_expr(self.early and not self.tsp2):  # before this CTA's first DSMEM send (tsp2: waited)\n",
)]

_LAST: dict = {}  # the last verify case's tensors (diagnostics only)
DECODE_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
VERIFY_REQUESTS = (1, 2, 4)
Q = 8
D, HEADS = R.D, R.HEADS
LOWER, EPS = -5.0, 1e-6

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def decode_mode(batch):
    """The gate mode providers.build_resources selects on this device (SM107 retunes B2/4/8)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import decode_mode as plugin_mode

    return plugin_mode(batch, torch.device("cuda", torch.cuda.current_device()))


def _layer(g):
    conv_w = (torch.randn((3 * D, 1, 4), generator=g, device="cuda") * 0.5).float()
    return _NS(
        conv1d=_NS(weight=conv_w, bias=None),
        dt_bias=(torch.randn((D,), generator=g, device="cuda") * 0.5).float(),
        A_log=(torch.randn((HEADS,), generator=g, device="cuda") * 0.5 - 1.0).float(),
        f_b_proj=_NS(weight=H.randn((D, 128), 0.1, g)),
        o_norm=_NS(weight=(1.0 + H.randn((128,), 0.2, g)).to(torch.bfloat16), eps=EPS),
        gate_lower_bound=LOWER,
    )


def _pools(g, slots):
    conv = H.randn((slots, 3, 3 * D), 1.0, g)
    state = (torch.randn((slots, HEADS, 128, 128), generator=g, device="cuda") * 0.1).float()
    return conv, state


def _proj(g, rows):
    return H.randn((rows, R.PITCH), 1.0, g)


def _ref_kwargs(layer):
    return dict(
        conv_w=layer.conv1d.weight, f_b=layer.f_b_proj.weight, a_log=layer.A_log,
        dt_bias=layer.dt_bias, norm_w=layer.o_norm.weight, lower_bound=LOWER, eps=EPS,
    )


@needs_gpu
@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_kda_decode(rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import PreparedKda

    g = H.generator(3000 + rows)
    layer = _layer(g)
    cap = 2 * rows + 2
    conv0, state0 = _pools(g, cap + 1)
    proj = _proj(g, rows)
    perm = torch.randperm(cap, generator=torch.Generator().manual_seed(rows))[:rows] + 1
    slots = perm.to(torch.int32).cuda()
    if rows >= 4:
        slots[rows // 2] = 0  # one padding row (slot <= 0)
    core = PreparedKda(layer, {rows: (R.PITCH, decode_mode(rows))}, False)
    conv, state = conv0.clone(), state0.clone()
    core.prepare_caches((conv, state))
    out = torch.empty((rows, D), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    core.run(proj, out, caches=(conv, state), slots=slots)
    torch.cuda.synchronize()
    ref_out, ref_conv, ref_state = R.kda_decode(proj, conv0, state0, slots, **_ref_kwargs(layer))
    item = f"kda.decode.b{rows}.mode{decode_mode(rows)}"
    H.record({"kind": "resources", "item": item, **H.function_resources(core.kernels[rows]), **H.device_facts()})
    H.check(item + ".out", out, ref_out, 0.02, 0.02)
    H.check(item + ".state", state, ref_state, 1e-3, 1e-3)
    assert torch.equal(conv.view(torch.int16), ref_conv.view(torch.int16)), f"{item}: conv state not bit-exact"
    pad = (slots <= 0).nonzero().flatten().tolist()
    for t in pad:
        assert int(out[t].view(torch.int16).abs().sum()) == 0, f"{item}: padding row {t} not zero"

    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION"):
        state_bytes = HEADS * 128 * 128 * 4
        nbytes = rows * (2 * state_bytes + 2 * 3 * 3 * D * 2 + R.PITCH * 2 + D * 2) + D * 128 * 2 + 3 * D * 4 * 4
        copies = H.copies_for(nbytes, cap=64)
        tconv, tstate = _pools(g, copies * rows + 1)
        core.prepare_caches((tconv, tstate))
        outs = [torch.empty((rows, D), dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
        tslots = [torch.arange(1 + i * rows, 1 + (i + 1) * rows, dtype=torch.int32, device="cuda") for i in range(copies)]
        calls = [lambda o=o, s=s: core.run(proj, o, caches=(tconv, tstate), slots=s) for o, s in zip(outs, tslots)]
        t = H.cold_time_us(calls)
        fl = H.floor_us(nbytes)
        H.record({"kind": "time", "item": item, **t, **fl})
        print("TIME", item, t, fl, flush=True)


def _boundary_flip(x, got, layer, h, f):
    """test_kda_gate_boundary_control's rule for one log_g cell: the FP32 a = f_a . f_b (``x``) sits within
    2^-16 |a| of a BF16 rounding midpoint and a neighbouring BF16 value of a reproduces the kernel's log_g within
    2e-5 (the kernel's MMA order and torch's order rounded a to neighbours: a reduction-order difference)."""
    lo = x.to(torch.bfloat16)
    bits = lo.view(torch.int16)
    up, dn = (bits + 1).view(torch.bfloat16), (bits - 1).view(torch.bfloat16)

    def g_of(av):
        scale = torch.exp(layer.A_log[h].float())
        return LOWER * torch.sigmoid(scale * (av.float() + layer.dt_bias[h * 128 + f].float()))

    mid = min(abs(float(x) - (float(lo) + float(v)) / 2) for v in (up, dn))
    return any(abs(float(g_of(v)) - float(got)) <= 2e-5 for v in (up, dn)) and mid <= 2**-16 * abs(float(x))


def _log_g_check(item, log_g, got_g, ref_rings, proj, layer, slots):
    """log_g ring: |x - ref| <= 1e-4 + 1e-4 |ref| on every cell, except cells ``_boundary_flip`` explains
    (counted and recorded). Fix-stage item FX-1 (post hoc, arm-neutral): the
    plain check failed one Breq2 cell identically for the shipped kernel and every variant on Rubin;
    test_kda_gate_boundary_control already attributes such cells to this flip."""
    assert bool(torch.isfinite(got_g).all()), f"{item}.log_g: non-finite cells"
    request_of = {s: i for i, s in enumerate(slots.tolist()) if s >= 0}
    a32 = proj[:, 4 * D : 4 * D + 128].float() @ layer.f_b_proj.weight.float().T
    explained, unexplained = 0, []
    for (slot, t), (_, _, g_ref, _) in ref_rings.items():
        got = log_g[slot, :, t]
        bad = ((got - g_ref).abs() > 1e-4 + 1e-4 * g_ref.abs()).nonzero().tolist()
        for h, f in bad:
            if _boundary_flip(a32[request_of[slot] * Q + t, h * 128 + f], got[h, f], layer, h, f):
                explained += 1
            else:
                unexplained.append((slot, t, h, f, float(got[h, f]), float(g_ref[h, f])))
    H.record({"kind": "check", "item": item + ".log_g", "boundary_explained": explained, "unexplained": len(unexplained)})
    assert not unexplained, (
        f"{item}.log_g: {len(unexplained)} cells beyond 1e-4 + 1e-4|ref| not explained by a BF16 boundary flip of a "
        f"(slot, t, head, feature, kernel, ref): {unexplained[:4]}"
    )


@needs_gpu
@pytest.mark.parametrize("requests", VERIFY_REQUESTS)
def test_kda_verify(requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import decode_conv_weight
    from sgl_nvidia_mega.kimi_k31.verify_kda import PreparedKdaVerify

    g = H.generator(4000 + requests)
    layer = _layer(g)
    cap, ring_len, scratch_rows = 2 * requests + 3, 16, 2 * requests + 1
    conv_phys, state = _pools(g, cap)  # pool conv [S, 3, 3D]; host passes its transpose
    conv0, state0 = conv_phys.clone(), state.clone()
    rawv = torch.zeros((cap, HEADS, ring_len, 128), dtype=torch.bfloat16, device="cuda")
    rawk = torch.zeros_like(rawv)
    log_g = torch.zeros((cap, HEADS, ring_len, 128), dtype=torch.float32, device="cuda")
    beta = torch.zeros((cap, HEADS, ring_len), dtype=torch.float32, device="cuda")
    windows = torch.zeros((scratch_rows, Q, 3, 3 * D), dtype=torch.bfloat16, device="cuda")
    for t in (rawv, rawk, log_g, beta, windows):
        H.poison(t)
    weights = _NS(
        conv_weight=decode_conv_weight(layer.conv1d.weight), has_bias=False, f_b=layer.f_b_proj.weight,
        a_log=layer.A_log, dt_bias=layer.dt_bias, norm_weight=layer.o_norm.weight.float(),
        lower_bound=LOWER, scale=128**-0.5, norm_eps=EPS,
    )
    pools = _NS(
        conv=conv_phys.transpose(-1, -2), temporal=state, rawv=rawv, rawk=rawk, log_g=log_g, beta=beta,
        conv_windows=windows.transpose(-1, -2),
    )
    core = PreparedKdaVerify(weights, ((requests, Q, R.PITCH, 0),), torch.cuda.current_device()).prepare_caches(pools)
    slots = (torch.randperm(cap, generator=torch.Generator().manual_seed(requests))[:requests]).to(torch.int32).cuda()
    scratch = torch.randperm(scratch_rows, generator=torch.Generator().manual_seed(7 + requests))[:requests].to(torch.int32).cuda()
    if requests == 4:
        slots[2] = -1  # padded request
    if os.environ.get("K31_PAD_REQUEST") is not None:  # pad another request (opt tsp at Breq 2)
        slots[int(os.environ["K31_PAD_REQUEST"])] = -1
    rows = requests * Q
    proj = _proj(g, rows)
    out = torch.empty((rows, D), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    step = _NS(state_slots=slots, scratch_rows=scratch, query_width=Q)
    core.run(proj, out, pools=pools, step=step)
    torch.cuda.synchronize()
    ref_out, ref_rings, ref_windows = R.kda_verify(
        proj, conv0, state0, slots, scratch, Q, ring_len=ring_len, **_ref_kwargs(layer)
    )
    forced = os.environ.get("K31MK_KDA_VERIFY_SCHEDULE")
    item = f"kda.verify.breq{requests}.q{Q}" + (f".s{forced}" if forced else "")
    _LAST.clear()
    _LAST.update(log_g=log_g, ref_rings=ref_rings, proj=proj, layer=layer, slots=slots)
    H.record({"kind": "resources", "item": item, **H.function_resources(core.provider.kernels[requests, Q]), **H.device_facts()})
    H.check(item + ".out", out, ref_out, 0.02, 0.02)
    assert torch.equal(state.view(torch.int32), state0.view(torch.int32)), f"{item}: checkpoint state modified"
    assert torch.equal(conv_phys.view(torch.int16), conv0.view(torch.int16)), f"{item}: persistent conv modified"
    for r in range(requests):
        if int(slots[r]) < 0:
            assert int(out[r * Q : (r + 1) * Q].view(torch.int16).abs().sum()) == 0, f"{item}: padded request output"
    got_v = torch.stack([rawv[s, :, t] for (s, t) in ref_rings])
    got_k = torch.stack([rawk[s, :, t] for (s, t) in ref_rings])
    got_g = torch.stack([log_g[s, :, t] for (s, t) in ref_rings])
    got_b = torch.stack([beta[s, :, t] for (s, t) in ref_rings])
    ref_v, ref_k, ref_g, ref_b = (torch.stack([v[i] for v in ref_rings.values()]) for i in range(4))
    H.check(item + ".rawv", got_v, ref_v, 1e-3, 2**-7)
    H.check(item + ".rawk", got_k, ref_k, 1e-3, 2**-7)
    _log_g_check(item, log_g, got_g, ref_rings, proj, layer, slots)
    H.check(item + ".beta", got_b, ref_b, 1e-4, 1e-4)
    for row, win in ref_windows.items():
        assert torch.equal(windows[row].view(torch.int16), win.view(torch.int16)), f"{item}: conv window row {row}"
    _LAST.update(out=out, rawv=rawv, rawk=rawk, beta=beta, windows=windows, state=state, conv=conv_phys)

    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION"):
        state_bytes = HEADS * 128 * 128 * 4
        nbytes = requests * (state_bytes + Q * (R.PITCH * 2 + D * 2 + 4 * HEADS * 128 * 2 + 3 * 3 * D * 2))
        nbytes += D * 128 * 2 + 3 * D * 4 * 4
        copies = H.copies_for(nbytes, cap=48)
        tconv, tstate = _pools(g, copies * requests)
        pools.conv, pools.temporal = tconv.transpose(-1, -2), tstate
        outs = [torch.empty((rows, D), dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
        steps = [
            _NS(state_slots=torch.arange(i * requests, (i + 1) * requests, dtype=torch.int32, device="cuda"),
                scratch_rows=scratch, query_width=Q)
            for i in range(copies)
        ]
        tr = [torch.zeros((copies * requests, HEADS, ring_len, 128), dtype=dt, device="cuda") for dt in (torch.bfloat16, torch.bfloat16, torch.float32)]
        pools.rawv, pools.rawk, pools.log_g = tr
        pools.beta = torch.zeros((copies * requests, HEADS, ring_len), dtype=torch.float32, device="cuda")
        core.prepare_caches(pools)
        calls = [lambda o=o, s=s: core.run(proj, o, pools=pools, step=s) for o, s in zip(outs, steps)]
        t = H.cold_time_us(calls)
        fl = H.floor_us(nbytes)
        H.record({"kind": "time", "item": item, **t, **fl})
        print("TIME", item, t, fl, flush=True)


# Opt "chunk" (KdaSplit.chunk_outputs, the chunked WY verify step). Lines stated before any GPU run
# (expected results):
# (1) the chunk arm passes every test_kda_verify gate (FP32 reference, rings, windows, untouched state / conv);
# (2) rings (raw V / K, log-g, beta) and conv windows bit-identical to the base arm (the same knobs without chunk):
#     the chunk path writes none of them;
# (3) outputs vs the base arm: every cell |c - b| <= 2^-7 |b| + 2^-20 (two BF16 ulps of b: both arms form the same
#     quantities ~1e-6 apart in FP32, so a cell moves at most one rounding step), and at most 1% of cells differ;
# (4) engagement: kda.verify.BUILT records the chunk path for the chunk arm's (requests, Q, opt), not for the base.
# Mutations chunk_* below must each fail it.
CHUNK_BASE = "wide+rs"


@needs_gpu
@pytest.mark.parametrize("requests", VERIFY_REQUESTS)
def test_kda_verify_chunk(requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base_opt = os.environ.get("K31_CHUNK_BASE", CHUNK_BASE)
    saved, runs = os.environ.get("K31MK_KDA_VERIFY_OPT"), {}
    try:
        for name, opt in (("base", base_opt), ("chunk", base_opt + "+chunk")):
            os.environ["K31MK_KDA_VERIFY_OPT"] = opt
            test_kda_verify(requests)
            runs[name] = dict(_LAST)
            built = V.BUILT.get((requests, Q, frozenset(opt.split("+"))))
            # FX-8 (KDA-4): chunk engages up to 2 requests (Breq 4 measured slower, keeps the base).
            want = name == "chunk" and requests <= 2
            assert built is want, f"{name} arm ({opt}): chunk path compiled = {built}"
    finally:
        if saved is None:
            os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
        else:
            os.environ["K31MK_KDA_VERIFY_OPT"] = saved
    item = f"kda.verify_chunk.breq{requests}.q{Q}.{base_opt}"
    b, c = runs["base"], runs["chunk"]
    for key, dt in (("rawv", torch.int16), ("rawk", torch.int16), ("log_g", torch.int32), ("beta", torch.int32),
                    ("windows", torch.int16)):
        assert torch.equal(b[key].view(dt), c[key].view(dt)), f"{item}: {key} differs from the base arm"
    ob, oc = b["out"].float(), c["out"].float()
    over = int(((oc - ob).abs() > 2**-7 * ob.abs() + 2**-20).sum())
    changed = int((b["out"].view(torch.int16) != c["out"].view(torch.int16)).sum())
    H.record({"kind": "check", "item": item, "cells": ob.numel(), "beyond_two_ulps": over, "changed": changed,
              "max_abs": float((oc - ob).abs().max())})
    assert over == 0, f"{item}: {over} output cells beyond two BF16 ulps of the base arm"
    assert changed <= 0.01 * ob.numel(), f"{item}: {changed}/{ob.numel()} output cells differ from the base arm"


# Opts "cg" / "cs" (with "chunk"; KdaSplit.gram_row / cs_products). Lines stated before any
# GPU run:
# (1) the arm passes every test_kda_verify gate (FP32 reference, rings, windows, untouched state / conv);
# (2) rings (raw V / K, log-g, beta) and conv windows bit-identical to the base arm (the same knobs without chunk);
# (3) outputs vs the base arm: every cell |c - b| <= 2^-7 |b| + 2^-20 and at most 1% of cells differ (the chunk lines);
# (4) "chunk+cg": outputs bit-identical to the "chunk" arm's (cg computes the same Gram rows with the same arithmetic in
#     the same order on the gate warps and moves them by DSMEM: a wrong row, entry, rank or ratio changes cells);
# (5) engagement: kda.verify.PATHS records (chunk, cg, cs) for every arm run here.
# Mutations cg_* / cs_* / csp_* / gb_* must each fail it. ("chunk+cg+cs+csp" added before any GPU run of csp;
# "chunk+cs+gb" and "chunk+cs+gb+csp" added before any GPU run of gb.)
CG_ARMS = ("chunk+cg", "chunk+cs", "chunk+cg+cs", "chunk+cg+cs+csp", "chunk+cs+gb", "chunk+cs+gb+csp")


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2))
@pytest.mark.parametrize("opts", CG_ARMS)
def test_kda_verify_chunk2(opts, requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base_opt = os.environ.get("K31_CG_BASE", "early+wide+rs+lt")
    arms = [("base", base_opt), ("arm", base_opt + "+" + opts)]
    if opts == "chunk+cg":
        arms.append(("chunk", base_opt + "+chunk"))
    saved, runs = os.environ.get("K31MK_KDA_VERIFY_OPT"), {}
    try:
        for name, opt in arms:
            os.environ["K31MK_KDA_VERIFY_OPT"] = opt
            test_kda_verify(requests)
            runs[name] = dict(_LAST)
            knobs = frozenset(opt.split("+"))
            want, got = ("chunk" in knobs, "cg" in knobs, "cs" in knobs), V.PATHS.get((requests, Q, knobs))
            assert got == want, f"{name} arm ({opt}): compiled paths {got}, expected {want}"
    finally:
        if saved is None:
            os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
        else:
            os.environ["K31MK_KDA_VERIFY_OPT"] = saved
    item = f"kda.verify_chunk2.breq{requests}.q{Q}.{base_opt}+{opts}"
    b, c = runs["base"], runs["arm"]
    for key, dt in (("rawv", torch.int16), ("rawk", torch.int16), ("log_g", torch.int32), ("beta", torch.int32),
                    ("windows", torch.int16)):
        assert torch.equal(b[key].view(dt), c[key].view(dt)), f"{item}: {key} differs from the base arm"
    ob, oc = b["out"].float(), c["out"].float()
    over = int(((oc - ob).abs() > 2**-7 * ob.abs() + 2**-20).sum())
    changed = int((b["out"].view(torch.int16) != c["out"].view(torch.int16)).sum())
    rec = {"kind": "check", "item": item, "cells": ob.numel(), "beyond_two_ulps": over, "changed": changed,
           "max_abs": float((oc - ob).abs().max())}
    if "chunk" in runs:
        rec["changed_vs_chunk"] = int((runs["chunk"]["out"].view(torch.int16) != c["out"].view(torch.int16)).sum())
    H.record(rec)
    assert over == 0, f"{item}: {over} output cells beyond two BF16 ulps of the base arm"
    assert changed <= 0.01 * ob.numel(), f"{item}: {changed}/{ob.numel()} output cells differ from the base arm"
    if "chunk" in runs:
        assert rec["changed_vs_chunk"] == 0, f"{item}: {rec['changed_vs_chunk']} cells differ from the chunk arm"


# Opt "wy" (KdaSplit.wy_outputs: the chunked WY verify step for 32 / 64 / 128-row CTAs). Lines stated before
# any GPU run of this committed form:
# (1) both arms pass every test_kda_verify gate (FP32 reference, rings, windows, untouched state / conv, padded rows);
# (2) rings (raw V / K, log-g, beta) and conv windows bit-identical to the base arm (wy writes none of them);
# (3) outputs vs the base arm: every cell |w - b| <= 2^-7 |b| + 2^-20 (two BF16 ulps) and at most 1 % of cells differ;
# (4) engagement: verify.WY_BUILT records wy for the arm, not for the base; with "wlf" (the served ON string) the 2-CTA 64-row CTAs
#     run 16 lanes per row (verify.WY_TWO_CTA) and the one-wave 128-row CTAs 8 lanes.
# K31_WY_BASE overrides the base string (default the served ON verify opts incl. wlf). Mutations wy_* must each fail it.
WY_BASE = "early+wide+rs+lt+tsp+dmc+chunk+cg+cs+wlf"
WY_REQUESTS = (4, 8, 16, 32)


@needs_gpu
@pytest.mark.parametrize("requests", WY_REQUESTS)
def test_kda_verify_wy(requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base_opt = os.environ.get("K31_WY_BASE", WY_BASE)
    saved, runs, built = os.environ.get("K31MK_KDA_VERIFY_OPT"), {}, {}
    try:
        for name, opt in (("base", base_opt), ("wy", base_opt + "+wy")):
            os.environ["K31MK_KDA_VERIFY_OPT"] = opt
            test_kda_verify(requests)
            runs[name] = dict(_LAST)
            built[name] = V.WY_BUILT.get((requests, Q, frozenset(opt.split("+"))))
    finally:
        if saved is None:
            os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
        else:
            os.environ["K31MK_KDA_VERIFY_OPT"] = saved
    item = f"kda.verify_wy.breq{requests}.q{Q}.{base_opt}"
    assert built["base"] is not None and not built["base"][0], f"{item}: base arm built wy ({built['base']})"
    assert built["wy"] is not None and built["wy"][0], f"{item}: wy arm did not build wy ({built['wy']})"
    if "wlf" in base_opt.split("+"):
        cluster, lanes = built["wy"][1], built["wy"][3]
        assert lanes == (16 if cluster == 2 else 8), f"{item}: wy lanes {lanes} at cluster {cluster}"
    b, w = runs["base"], runs["wy"]
    for key, dt in (("rawv", torch.int16), ("rawk", torch.int16), ("log_g", torch.int32), ("beta", torch.int32),
                    ("windows", torch.int16)):
        assert torch.equal(b[key].view(dt), w[key].view(dt)), f"{item}: {key} differs from the base arm"
    ob, ow = b["out"].float(), w["out"].float()
    over = int(((ow - ob).abs() > 2**-7 * ob.abs() + 2**-20).sum())
    changed = int((b["out"].view(torch.int16) != w["out"].view(torch.int16)).sum())
    H.record({"kind": "check", "item": item, "cells": ob.numel(), "beyond_two_ulps": over, "changed": changed,
              "max_abs": float((ow - ob).abs().max()), "built": list(built["wy"])})
    assert over == 0, f"{item}: {over} output cells beyond two BF16 ulps of the base arm"
    assert changed <= 0.01 * ob.numel(), f"{item}: {changed}/{ob.numel()} output cells differ from the base arm"


# Bitwise knobs "lean" / "bal" (and any other opt claimed bitwise). Line stated before any GPU run
# (expected results): with the knob added to K31_BITWISE_BASE (default early+wide+rs+lt), test_kda_verify's gates pass
# AND the output, raw V / K, log-g, beta rings and conv windows are bit-identical to the base arm's. lean is Q8 / cluster-8
# only (Breq 1 and 2). Mutations lean_* / bal_* must each fail it.
BITWISE_OPTS = ("lean", "bal", "lean+bal")


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2))
@pytest.mark.parametrize("opts", BITWISE_OPTS)
def test_kda_verify_bitwise_opts(opts, requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base_opt = os.environ.get("K31_BITWISE_BASE", "early+wide+rs+lt")
    saved, runs = os.environ.get("K31MK_KDA_VERIFY_OPT"), {}
    try:
        for name, opt in (("base", base_opt), ("arm", base_opt + "+" + opts)):
            os.environ["K31MK_KDA_VERIFY_OPT"] = opt
            test_kda_verify(requests)
            runs[name] = dict(_LAST)
            assert (requests, Q, frozenset(opt.split("+"))) in V.BUILT, f"{opt}: no KdaSplit built with these opts"
    finally:
        if saved is None:
            os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
        else:
            os.environ["K31MK_KDA_VERIFY_OPT"] = saved
    item = f"kda.verify_bitwise.breq{requests}.q{Q}.{base_opt}+{opts}"
    b, a = runs["base"], runs["arm"]
    for key, dt in (("out", torch.int16), ("rawv", torch.int16), ("rawk", torch.int16), ("log_g", torch.int32),
                    ("beta", torch.int32), ("windows", torch.int16)):
        assert torch.equal(b[key].view(dt), a[key].view(dt)), f"{item}: {key} differs from the base arm"
    H.record({"kind": "check", "item": item, "bitwise": True})


# Opt "tsp" (token-split phase 1, claimed BITWISE). Line stated before any GPU run
# (expected results): with "tsp" (alone, or with the gate warp's existing bitwise "dmc") added to K31_BITWISE_BASE (default
# early+wide+rs+lt; also run on wide+rs+pre), test_kda_verify's gates pass AND the output, raw V / K, log-g, beta rings and
# conv windows are bit-identical to the base arm's (test_kda_verify_bitwise_opts), at Breq 1 and 2 (Q8, 8-CTA clusters);
# and kda.verify.TSP_BUILT shows the tsp phase 1 compiled for the arm, not for the base (engagement).
# Mutations tsp_* must each fail it. (Breq 3 and 8 ran this test with tsp engaged in an earlier run: PASS,
# bitwise; tsp now engages only up to kernel.TSP_MAX_REQUESTS = 2, see test_kda_verify_tsp_fallback.)
@needs_gpu
@pytest.mark.parametrize("requests", (1, 2))
@pytest.mark.parametrize("opts", ("tsp", "tsp+dmc"))
def test_kda_verify_tsp(opts, requests):
    if HEADS == 12:
        pytest.skip("tsp switches itself off at 12 heads (test_kda_verify_tsp_heads12)")
    test_kda_verify_bitwise_opts(opts, requests)  # loads the (possibly mutated) plugin first
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V  # after load_plugin: the module that compiled

    base_opt = os.environ.get("K31_BITWISE_BASE", "early+wide+rs+lt")
    arm = frozenset((base_opt + "+" + opts).split("+"))
    assert V.TSP_BUILT.get((requests, Q, arm)) is True, "tsp phase 1 not compiled for the arm"
    assert V.TSP_BUILT.get((requests, Q, frozenset(base_opt.split("+")))) is False, "base arm compiled tsp"


# Opt "dq" (KdaSplit.dq: the serial token loop keeps each token's q-dot partials; their butterflies and the
# output pushes run after the loop; claimed BITWISE). Line stated before any GPU run of this test
# (expected results): with "dq" added to the served verify opt (and to served + "b3w" at Breq 3, the c=8 form), test_kda_verify's gates pass
# AND the output, raw V / K, log-g, beta rings and conv windows are bit-identical to the same opt without "dq". Mutation
# dq_lastlevel (the deferred butterflies skip their last level) must fail it wherever "dq" engages (Breq 3 here; its failure is
# also the in-process engagement proof, since the mutated line only runs when dq engages). Run with K31_TEST_HEADS=6.
MUTATIONS["dq_lastlevel"] = (
    "kimi_k31/k3/cutedsl/kda/kernel.py",
    "                    for shift in reversed(range(levels)):\n"
    "                        for t in tail:\n"
    "                            for p in cutlass.range_constexpr(self.passes):\n"
    "                                qd[t, p] += cute.arch.shuffle_sync_bfly(qd[t, p], 1 << shift)\n",
    "                    for shift in reversed(range(levels)):\n"
    "                        if cutlass.const_expr(shift != 0):\n"
    "                            for t in tail:\n"
    "                                for p in cutlass.range_constexpr(self.passes):\n"
    "                                    qd[t, p] += cute.arch.shuffle_sync_bfly(qd[t, p], 1 << shift)\n",
)
DQ_SERVED = "early+wide+rs+lt+tsp+dmc+chunk+cg+cs+wlf"


@needs_gpu
@pytest.mark.parametrize("requests,base", ((3, DQ_SERVED + "+b3w"), (3, DQ_SERVED), (4, DQ_SERVED), (8, DQ_SERVED)))
def test_kda_verify_dq(requests, base):
    H.load_plugin(MUTATIONS)
    saved, runs = os.environ.get("K31MK_KDA_VERIFY_OPT"), {}
    try:
        for name, opt in (("base", base), ("arm", base + "+dq")):
            os.environ["K31MK_KDA_VERIFY_OPT"] = opt
            test_kda_verify(requests)
            runs[name] = dict(_LAST)
    finally:
        if saved is None:
            os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
        else:
            os.environ["K31MK_KDA_VERIFY_OPT"] = saved
    item = f"kda.verify_dq.breq{requests}.q{Q}.{base}"
    b, a = runs["base"], runs["arm"]
    for key, dt in (("out", torch.int16), ("rawv", torch.int16), ("rawk", torch.int16), ("log_g", torch.int32),
                    ("beta", torch.int32), ("windows", torch.int16)):
        diff = int((b[key].view(dt) != a[key].view(dt)).sum())
        assert diff == 0, f"{item}: {key} differs from the base arm in {diff} cells"
    H.record({"kind": "check", "item": item, "bitwise": True})


# Where tsp must not engage (more than kernel.TSP_MAX_REQUESTS requests, or no 8-CTA clusters: Breq 4 on
# SM107), the kernel keeps the base phase 1 and stays bitwise to the base (line stated for Breq 4; Breq 3 and 8
# added with the request cap, before any GPU run of the capped tree).
@needs_gpu
@pytest.mark.parametrize("requests", (3, 4, 8))
def test_kda_verify_tsp_fallback(requests):
    test_kda_verify_bitwise_opts("tsp", requests)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.kernel import TSP_MAX_REQUESTS

    base_opt = os.environ.get("K31_BITWISE_BASE", "early+wide+rs+lt")
    cluster8 = V.verify_schedule(requests, Q, torch.device("cuda", torch.cuda.current_device()))[3] == 8
    expected = cluster8 and requests <= TSP_MAX_REQUESTS
    built = V.TSP_BUILT.get((requests, Q, frozenset((base_opt + "+tsp").split("+"))))
    assert built is expected, f"tsp engaged={built} at Breq {requests}, expected {expected}"


# Tsp with a padded request (only Breq 4 pads in test_kda_verify, where tsp does not engage). Line stated
# before any GPU run: Breq 2 with request 1 padded: the tsp arm passes test_kda_verify_tsp's gates
# (padded output exactly 0, its slot untouched) and is bitwise to the base arm.
@needs_gpu
def test_kda_verify_tsp_padded():
    os.environ["K31_PAD_REQUEST"] = "1"
    try:
        test_kda_verify_tsp("tsp", 2)
    finally:
        os.environ.pop("K31_PAD_REQUEST", None)


# FX-5 / FX-8 on the merged string. Line stated before any GPU run:
# the merged opts M1 = tsp+dmc+chunk+cg+cs engage exactly at Breq 1 and 2 (the measured range; Breq 4 asserted in cg / cs
# before FX-8): at Breq 1 / 2 the merged string passes test_kda_verify's gates and kda.verify records tsp, dmc, chunk, cg
# and cs as compiled; at Breq 3 / 4 / 8 it builds, is BITWISE to the base string (outputs, rings, windows) and records none
# of them. Catches: an uncapped dmc or chunk family (fx5_uncap / fx8_uncap), a cap that drops them at Breq 1 / 2
# (fx58_overcap; dmc is bitwise, so only the engagement record sees it), and the Breq 4 build failure.
M1_OPTS = "tsp+dmc+chunk+cg+cs"


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2, 3, 4, 8))
def test_kda_verify_m1_engagement(requests):
    on = requests <= 2  # the spec (measured range), not the kernel's constants
    base_opt = os.environ.get("K31_BITWISE_BASE", "early+wide+rs+lt")
    arm = frozenset((base_opt + "+" + M1_OPTS).split("+"))
    if on:
        H.load_plugin(MUTATIONS)
        saved = os.environ.get("K31MK_KDA_VERIFY_OPT")
        try:
            os.environ["K31MK_KDA_VERIFY_OPT"] = "+".join(sorted(arm))
            test_kda_verify(requests)
        finally:
            if saved is None:
                os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
            else:
                os.environ["K31MK_KDA_VERIFY_OPT"] = saved
    else:
        test_kda_verify_bitwise_opts(M1_OPTS, requests)  # loads the (possibly mutated) plugin first
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V  # after load_plugin: the module that compiled

    key = (requests, Q, arm)
    got = (V.TSP_BUILT.get(key), V.DMC_BUILT.get(key), V.PATHS.get(key))
    tsp_on = on and HEADS != 12  # 12 heads (fast math): tsp switches itself off (test_kda_verify_tsp_heads12)
    assert got == (tsp_on, on, (on, on, on)), f"Breq {requests}: (tsp, dmc, (chunk, cg, cs)) compiled = {got}"


# Line stated before any run: at 12 KDA heads (TP8; fast math, which
# prologue_tsp does not build) "tsp" switches itself off instead of asserting. At Breq 1 / 2 the base + tsp string builds,
# passes test_kda_verify's gates, is BITWISE to the base string (outputs, rings, windows; same kernel), kda.verify records tsp
# as not compiled, and the schedule log line of that build names its (cluster, CTAs per SM, lanes per row) = kda.verify.CLUSTERS
# and "tsp False". Mutations tsp_fastmath (fails at build) and sched_log_cluster must each fail it.
@needs_gpu
@pytest.mark.skipif(HEADS != 12, reason="12 KDA heads only (K31_TEST_HEADS=12)")
@pytest.mark.parametrize("requests", (1, 2))
def test_kda_verify_tsp_heads12(requests, caplog):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V  # after load_plugin: the module that compiles

    V.executable.cache_clear()  # every build below runs (and logs) here
    with caplog.at_level(logging.INFO, logger=V.__name__):
        test_kda_verify_bitwise_opts("tsp", requests)
    base_opt = os.environ.get("K31_BITWISE_BASE", "early+wide+rs+lt")
    arm = frozenset((base_opt + "+tsp").split("+"))
    assert V.TSP_BUILT.get((requests, Q, arm)) is False, "tsp compiled at 12 heads"
    cluster, occupancy, lanes = V.CLUSTERS[requests, Q, arm]
    want = (f"KdaSplit verify Breq {requests} Q {Q} heads 12: cluster {cluster}, CTAs/SM {occupancy}, lanes/row {lanes}; "
            f"tsp False")
    logged = [r.getMessage() for r in caplog.records if r.name == V.__name__]
    assert any(m.startswith(want) and m.endswith("+".join(sorted(arm))) for m in logged), (want, logged)


# Rubin retune experiment (not a GB300 policy): B2-8 decode with the gate modes and staged
# schedules GB300 uses at B16+ ("early.lanes.occupancy.cluster.evict"). Same gates as
# test_kda_decode; timing recorded per variant. Run with K31_KDA_SWEEP=1.
SWEEP = {
    "m2_default": (2, None),
    "m0_s16": (0, "3.16.3.2.1"),
    "m1_s16": (1, "3.16.3.2.1"),
    "m0_e1c4": (0, "1.0.0.4.0"),
}


# B16-64 (GB300: mode 0, "3.16.3.2.1" at B16-32, "3.16.4.2.1" at B64) on VR200.
SWEEP_WIDE = {
    "m0_s16o3c2": (0, "3.16.3.2.1"),
    "m0_s16o4c2": (0, "3.16.4.2.1"),
    "m0_s16o2c2": (0, "3.16.2.2.1"),
    "m0_s8o3c2": (0, "3.8.3.2.1"),
    "m0_s16o3c4": (0, "2.16.3.4.1"),
    "m1_s16o3c2": (1, "3.16.3.2.1"),
    "m0_e1c4": (0, "1.0.0.4.0"),
    "m0_e1c2": (0, "1.0.0.2.0"),
}


@needs_gpu
@pytest.mark.skipif(os.environ.get("K31_KDA_SWEEP") != "1", reason="tuning sweep (K31_KDA_SWEEP=1)")
@pytest.mark.parametrize("rows", (16, 32, 64))
@pytest.mark.parametrize("variant", list(SWEEP_WIDE))
def test_kda_decode_schedule_wide(variant, rows):
    SWEEP[variant] = SWEEP_WIDE[variant]
    test_kda_decode_schedule(variant, rows)


@needs_gpu
@pytest.mark.skipif(os.environ.get("K31_KDA_SWEEP") != "1", reason="tuning sweep (K31_KDA_SWEEP=1)")
@pytest.mark.parametrize("rows", (2, 3, 4, 5, 6, 7, 8))
@pytest.mark.parametrize("variant", list(SWEEP))
def test_kda_decode_schedule(variant, rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import prepared as P

    mode, schedule = SWEEP[variant]
    saved = dict(P.SCHEDULE)
    retune = os.environ.get("K31MK_KDA_SM107_RETUNE")
    os.environ["K31MK_KDA_SM107_RETUNE"] = "0"  # the sweep sets each variant explicitly
    try:
        if schedule is not None:
            P.SCHEDULE[HEADS] = f"{rows}:{schedule}"
        P.executable.cache_clear()
        g = H.generator(5000 + rows)
        layer = _layer(g)
        cap = 2 * rows + 2
        conv0, state0 = _pools(g, cap + 1)
        proj = _proj(g, rows)
        slots = (torch.randperm(cap, generator=torch.Generator().manual_seed(rows))[:rows] + 1).to(torch.int32).cuda()
        core = P.PreparedKda(layer, {rows: (R.PITCH, mode)}, False)
        conv, state = conv0.clone(), state0.clone()
        core.prepare_caches((conv, state))
        out = torch.empty((rows, D), dtype=torch.bfloat16, device="cuda")
        H.poison(out)
        core.run(proj, out, caches=(conv, state), slots=slots)
        torch.cuda.synchronize()
        ref_out, ref_conv, ref_state = R.kda_decode(proj, conv0, state0, slots, **_ref_kwargs(layer))
        item = f"kda.sweep.{variant}.b{rows}"
        H.check(item + ".out", out, ref_out, 0.02, 0.02)
        H.check(item + ".state", state, ref_state, 1e-3, 1e-3)
        assert torch.equal(conv.view(torch.int16), ref_conv.view(torch.int16)), f"{item}: conv state not bit-exact"
        state_bytes = HEADS * 128 * 128 * 4
        nbytes = rows * (2 * state_bytes + 2 * 3 * 3 * D * 2 + R.PITCH * 2 + D * 2) + D * 128 * 2 + 3 * D * 4 * 4
        copies = H.copies_for(nbytes, cap=64)
        tconv, tstate = _pools(g, copies * rows + 1)
        core.prepare_caches((tconv, tstate))
        outs = [torch.empty((rows, D), dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
        tslots = [torch.arange(1 + i * rows, 1 + (i + 1) * rows, dtype=torch.int32, device="cuda") for i in range(copies)]
        calls = [lambda o=o, s=s: core.run(proj, o, caches=(tconv, tstate), slots=s) for o, s in zip(outs, tslots)]
        t = H.cold_time_us(calls)
        fl = H.floor_us(nbytes)
        H.record({"kind": "time", "item": item, **t, **fl, **H.function_resources(core.kernels[rows])})
        print("TIME", item, t, fl, flush=True)
    finally:
        P.SCHEDULE.clear()
        P.SCHEDULE.update(saved)
        P.executable.cache_clear()
        if retune is None:
            os.environ.pop("K31MK_KDA_SM107_RETUNE", None)
        else:
            os.environ["K31MK_KDA_SM107_RETUNE"] = retune


# SM107 verify schedule sweep (lanes per state row . minimum CTAs per SM . CTAs per head);
# same gates as test_kda_verify. Run with K31_KDA_VSWEEP=1 (K31_TIME=1 for timing).
VSWEEP = ("8.2.8", "16.2.8", "8.1.8", "16.1.8", "8.2.4", "16.2.4", "8.3.4", "16.3.4", "8.4.2", "16.4.2")


@needs_gpu
@pytest.mark.skipif(os.environ.get("K31_KDA_VSWEEP") != "1", reason="tuning sweep (K31_KDA_VSWEEP=1)")
@pytest.mark.parametrize("requests", VERIFY_REQUESTS)
@pytest.mark.parametrize("schedule", VSWEEP)
def test_kda_verify_schedule(schedule, requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    saved = os.environ.get("K31MK_KDA_VERIFY_SCHEDULE")
    os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = schedule
    try:
        test_kda_verify(requests)
    finally:
        if saved is None:
            os.environ.pop("K31MK_KDA_VERIFY_SCHEDULE", None)
        else:
            os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = saved
        V.executable.cache_clear()


# MKB-0: the verify MK at Breq 8 (contract) and the Breq 16 table entry, on the served table (no forced
# schedule): every test_kda_verify gate, and on SM107 the (8, 8) / (16, 8) entries build 2-CTA clusters (GB300 or
# K31MK_KDA_SM107_RETUNE=0: the 8-CTA default).
MKB_REQUESTS = (8, 16)


@needs_gpu
@pytest.mark.parametrize("requests", MKB_REQUESTS)
def test_kda_verify_mkb(requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    forced = os.environ.pop("K31MK_KDA_VERIFY_SCHEDULE", None)
    V.executable.cache_clear()
    V.CLUSTERS.clear()
    try:
        test_kda_verify(requests)
    finally:
        if forced is not None:
            os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = forced
    built = [v for (b, q, _), v in V.CLUSTERS.items() if (b, q) == (requests, Q)]
    assert len(built) == 1, f"Breq {requests}: {len(built)} verify kernels built"
    retune = os.environ.get("K31MK_KDA_SM107_RETUNE", "1") != "0"
    want = 2 if retune and torch.cuda.get_device_capability() == (10, 7) else 8
    assert built[0][0] == want, f"Breq {requests}: built (cluster, occupancy, lanes) {built[0]}, want cluster {want}"


# Opt "wlf" (default off). Lines stated before any GPU run of this test
# (expected results), base = the served verify opts (K31_WLF_BASE), arm = base + "wlf", served table (no forced schedule):
# (1) the arm passes every test_kda_verify gate (FP32 reference, rings, windows, untouched state / conv);
# (2) rings (raw V / K, log-g, beta) and conv windows bit-identical to the base arm (wlf changes only the recurrence);
# (3) outputs vs the base arm: Breq 1 / 4 bit-identical (same kernel); Breq 8 / 16 / 32 every cell |a - b| <= 2^-7 |b| +
#     2^-20 (two BF16 ulps: the row dots reduce over 8 lanes instead of 4, FP32 values ~1e-6 apart) and at most 1% differ;
# (4) engagement (SM107, retune on): 64-row CTAs (2-CTA clusters, Breq 8 / 16 / 32) build 8 lanes per row with wlf and 4
#     without; Breq 1 / 4 build the same lanes as the base.
# (5) (added before any run of it) SM107, retune on: Breq 32 with wlf builds the WLF_ONE_WAVE
#     entry, cluster 1 / occupancy 1 / 8 lanes (128-row CTAs, 4 rows per thread), while the base stays on 2-CTA clusters;
#     (3)'s two-ulp / 1 % lines hold for it unchanged (the same 8-lane row dots).
# (5b) (added before any run of it) the same at TP8 shapes (K31_TEST_HEADS=12) for Breq 16 and 32:
#     WLF_ONE_WAVE's 1-CTA clusters wherever WAVE_ONE lists (heads, Breq).
# Mutations wlf_* and pk_no_decay (wlf turns the packed path on for the 2-row threads) must each fail it.
WLF_REQUESTS = (1, 4, 8, 16, 32)
WLF_BASE = "early+wide+rs+lt+tsp+dmc+chunk+cg+cs"
# (heads per rank, Breq) where the 128-row 1-CTA form beat the 2-CTA form (same GPU, ABBA x 2, cold graph):
# TP16 Breq 32; TP8 Breq 16 and 32. It lost where the 2-CTA grid is one wave:
# TP16 Breq 16, TP8 Breq 8. Independent of verify_schedule's rule (taken from those timings).
WAVE_ONE = {(6, 32), (12, 16), (12, 32)}
WAVE_TWO_CTA = {(6, 8), (6, 16), (12, 8)}


@needs_gpu
def test_kda_verify_wlf_waves():
    """verify_schedule with "wlf" picks (0, 8, 1, 1) exactly at WAVE_ONE and keeps (0, 8, 4, 2) at WAVE_TWO_CTA; without
    "wlf", or without the head count, every Breq 8 / 16 / 32 keeps (0, 8, 4, 2). SM107 (VR200) only: the
    measured cells are that GPU's."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    dev = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(dev) != (10, 7) or os.environ.get("K31MK_KDA_SM107_RETUNE", "1") == "0":
        pytest.skip("SM107 retune table")
    assert not os.environ.get("K31MK_KDA_VERIFY_SCHEDULE"), "a forced schedule overrides the table"
    wlf, served = frozenset(WLF_BASE.split("+")) | {"wlf"}, frozenset(WLF_BASE.split("+"))
    for heads, requests in sorted(WAVE_ONE | WAVE_TWO_CTA):
        want = (0, 8, 1, 1) if (heads, requests) in WAVE_ONE else (0, 8, 4, 2)
        assert V.verify_schedule(requests, Q, dev, wlf, heads) == want, (heads, requests)
        assert V.verify_schedule(requests, Q, dev, served, heads) == (0, 8, 4, 2), (heads, requests)
        assert V.verify_schedule(requests, Q, dev, wlf) == (0, 8, 4, 2), (heads, requests)


@needs_gpu
@pytest.mark.parametrize("requests", WLF_REQUESTS)
def test_kda_verify_wlf(requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base_opt = os.environ.get("K31_WLF_BASE", WLF_BASE)
    forced = os.environ.pop("K31MK_KDA_VERIFY_SCHEDULE", None)
    saved, runs, lanes = os.environ.get("K31MK_KDA_VERIFY_OPT"), {}, {}
    try:
        for name, opt in (("base", base_opt), ("arm", base_opt + "+wlf")):
            os.environ["K31MK_KDA_VERIFY_OPT"] = opt
            test_kda_verify(requests)
            runs[name] = dict(_LAST)
            built = V.CLUSTERS.get((requests, Q, frozenset(opt.split("+"))))  # set by the (cached) build
            assert built is not None, f"{name} arm ({opt}): no verify kernel built with these opts"
            lanes[name] = built
    finally:
        if saved is None:
            os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
        else:
            os.environ["K31MK_KDA_VERIFY_OPT"] = saved
        if forced is not None:
            os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = forced
    item = f"kda.verify_wlf.breq{requests}.q{Q}.{base_opt}"
    b, a = runs["base"], runs["arm"]
    for key, dt in (("rawv", torch.int16), ("rawk", torch.int16), ("log_g", torch.int32), ("beta", torch.int32),
                    ("windows", torch.int16)):
        assert torch.equal(b[key].view(dt), a[key].view(dt)), f"{item}: {key} differs from the base arm"
    ob, oa = b["out"].float(), a["out"].float()
    over = int(((oa - ob).abs() > 2**-7 * ob.abs() + 2**-20).sum())
    changed = int((b["out"].view(torch.int16) != a["out"].view(torch.int16)).sum())
    H.record({"kind": "check", "item": item, "cells": ob.numel(), "beyond_two_ulps": over, "changed": changed,
              "max_abs": float((oa - ob).abs().max()), "base_built": list(lanes["base"]), "arm_built": list(lanes["arm"])})
    sm107 = torch.cuda.get_device_capability() == (10, 7) and os.environ.get("K31MK_KDA_SM107_RETUNE", "1") != "0"
    if sm107 and lanes["base"][0] == 2:
        assert (lanes["base"][2], lanes["arm"][2]) == (4, 8), f"{item}: lanes per row base / arm {lanes['base']} / {lanes['arm']}"
        assert over == 0, f"{item}: {over} output cells beyond two BF16 ulps of the base arm"
        assert changed <= 0.01 * ob.numel(), f"{item}: {changed}/{ob.numel()} output cells differ from the base arm"
    else:
        assert lanes["arm"] == lanes["base"], f"{item}: built {lanes['arm']} with wlf vs {lanes['base']} without"
        assert changed == 0, f"{item}: {changed} output cells differ from the base arm (same kernel expected)"
    if sm107:
        assert (lanes["base"][0] == 2) == (requests >= 8), f"{item}: cluster {lanes['base'][0]} at Breq {requests}"
        if (R.HEADS, requests) in WAVE_ONE:
            assert tuple(lanes["arm"]) == (1, 1, 8), f"{item}: wlf built {lanes['arm']}, want WLF_ONE_WAVE's (1, 1, 8)"


# L3: Breq 3 / 5 / 6 / 7 (Q8) run their own verify graphs. Every test_kda_verify gate (FP32 reference, rings,
# windows, untouched state / conv) at those request counts, and on SM107 the cluster of the graph they were padded to:
# Breq 3 -> the (4, 8) entry's 4-CTA clusters, Breq 5-7 -> the (8, 8) entry's 2-CTA clusters (values copied from
# SM107_VERIFY); GB300 or K31MK_KDA_SM107_RETUNE=0: the 8-CTA default.
L3_CLUSTER_SM107 = {3: 4, 5: 2, 6: 2, 7: 2}


@needs_gpu
@pytest.mark.parametrize("requests", sorted(L3_CLUSTER_SM107))
def test_kda_verify_l3(requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    forced = os.environ.pop("K31MK_KDA_VERIFY_SCHEDULE", None)
    V.executable.cache_clear()
    V.CLUSTERS.clear()
    try:
        test_kda_verify(requests)
    finally:
        if forced is not None:
            os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = forced
    built = [v for (b, q, _), v in V.CLUSTERS.items() if (b, q) == (requests, Q)]
    assert len(built) == 1, f"Breq {requests}: {len(built)} verify kernels built"
    retune = os.environ.get("K31MK_KDA_SM107_RETUNE", "1") != "0"
    want = L3_CLUSTER_SM107[requests] if retune and torch.cuda.get_device_capability() == (10, 7) else 8
    assert built[0][0] == want, f"Breq {requests}: built (cluster, occupancy, lanes) {built[0]}, want cluster {want}"


#  (wide decode tier): Breq 9-15 (Q8, 72-120 rows) run their own verify graphs. Every test_kda_verify gate
# (FP32 reference, rings, windows, untouched state / conv) at those ragged request counts, and on SM107 the cluster of the
# Breq 16 graph: SM107_VERIFY[(16, 8)] = (0, 8, 4, 2) -> 2-CTA clusters (value copied from the table, not from a run);
# GB300 or K31MK_KDA_SM107_RETUNE=0: the 8-CTA default.
WIDE_CLUSTER_SM107 = {b: 2 for b in range(9, 16)}


@needs_gpu
@pytest.mark.parametrize("requests", sorted(WIDE_CLUSTER_SM107))
def test_kda_verify_wide(requests):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    forced = os.environ.pop("K31MK_KDA_VERIFY_SCHEDULE", None)
    V.executable.cache_clear()
    V.CLUSTERS.clear()
    try:
        test_kda_verify(requests)
    finally:
        if forced is not None:
            os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = forced
    built = [v for (b, q, _), v in V.CLUSTERS.items() if (b, q) == (requests, Q)]
    assert len(built) == 1, f"Breq {requests}: {len(built)} verify kernels built"
    retune = os.environ.get("K31MK_KDA_SM107_RETUNE", "1") != "0"
    want = WIDE_CLUSTER_SM107[requests] if retune and torch.cuda.get_device_capability() == (10, 7) else 8
    assert built[0][0] == want, f"Breq {requests}: built (cluster, occupancy, lanes) {built[0]}, want cluster {want}"


# Per-token cost of the ReplaySSM verify recurrence on this GPU: Breq 1 at Q 2/4/8/16 (same
# gates); the slope of us vs Q is the serial per-token cost, the intercept the fixed part.
@needs_gpu
@pytest.mark.skipif(os.environ.get("K31_KDA_QSCAN") != "1", reason="per-token scan (K31_KDA_QSCAN=1)")
@pytest.mark.parametrize("queries", (2, 4, 8, 16))
def test_kda_verify_qscan(queries):
    global Q
    saved = Q
    Q = queries
    try:
        test_kda_verify(1)
    finally:
        Q = saved


# Control for the 1-cell log_g misses (Q16 Breq1; cluster-2 Q8 Breq1): hypothesis = a 1-ulp
# BF16 flip of a = bf16(f_a @ f_b^T) whose FP32 value sits at a BF16 rounding midpoint, so the
# kernel's MMA order and torch's order round to neighbours (reduction order, allowed by the
# accuracy rule). The control recomputes log_g from the neighbouring BF16 value of a: every
# violating cell must match the kernel within 2e-5 (fast-math sigmoid) that way, with |a_fp32 - midpoint| <=
# 2^-16 |a|; otherwise it is a kernel bug. Run with K31_KDA_GATE_DIAG=1. breq2 / breq4: the default Q8
# schedule at Breq 2 / 4 (the 6-head misses on GB300).
@needs_gpu
@pytest.mark.skipif(os.environ.get("K31_KDA_GATE_DIAG") != "1", reason="control (K31_KDA_GATE_DIAG=1)")
@pytest.mark.parametrize("case", ("q16", "c2", "breq2", "breq4"))
def test_kda_gate_boundary_control(case):
    global Q
    saved_q, saved_s = Q, os.environ.get("K31MK_KDA_VERIFY_SCHEDULE")
    requests = int(case[4:]) if case.startswith("breq") else 1
    if case == "q16":
        Q = 16
    elif case == "c2":
        os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = "8.4.2"
    try:
        try:
            test_kda_verify(requests)
            missed = False
        except AssertionError:
            missed = True
    finally:
        Q = saved_q
        if saved_s is None:
            os.environ.pop("K31MK_KDA_VERIFY_SCHEDULE", None)
        else:
            os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = saved_s
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

        V.executable.cache_clear()
    log_g, ref_rings, proj, layer = (_LAST[k] for k in ("log_g", "ref_rings", "proj", "layer"))
    request_of = {s: i for i, s in enumerate(_LAST["slots"].tolist()) if s >= 0}  # ring slot -> request (proj rows)
    a32 = proj[:, 4 * D : 4 * D + 128].float() @ layer.f_b_proj.weight.float().T  # [rows, D]
    explained, total = 0, 0
    for (slot, t), (_, _, g_ref, _) in ref_rings.items():
        got = log_g[slot, :, t]
        bad = ((got - g_ref).abs() > 1e-4 + 1e-4 * g_ref.abs()).nonzero().tolist()
        for h, f in bad:
            total += 1
            x = a32[request_of[slot] * Q + t, h * 128 + f]
            lo = x.to(torch.bfloat16)
            bits = lo.view(torch.int16)  # BF16 neighbours: +-1 in the bit pattern (same sign)
            up, dn = (bits + 1).view(torch.bfloat16), (bits - 1).view(torch.bfloat16)
            def g_of(av):
                return LOWER * torch.sigmoid(torch.exp(layer.A_log[h].float()) * (av.float() + layer.dt_bias[h * 128 + f].float()))
            alt = [g_of(v) for v in (up, dn)]
            mid = min(abs(float(x) - (float(lo) + float(v)) / 2) for v in (up, dn))
            ok = any(abs(float(gv) - float(got[h, f])) <= 2e-5 for gv in alt) and mid <= 2**-16 * abs(float(x))
            print("GATE_CELL", case, slot, t, h, f, "a32=%.9g bf16=%.9g kernel=%.7g ref=%.7g mid_dist=%.3g ok=%s"
                  % (float(x), float(lo), float(got[h, f]), float(g_ref[h, f]), mid, ok), flush=True)
            explained += ok
    H.record({"kind": "control", "item": f"kda.gate_boundary.{case}", "missed": missed, "cells": total, "explained": explained})
    assert total == explained, f"{case}: {total - explained}/{total} log_g misses not explained by a BF16 boundary flip"


# Opt "tsp4" (KdaSplit.prologue_tsp4: tsp's token split on the 4-CTA clusters of 32 rows that SM107 runs at Breq 4,
# and at Breq 5 / 6 with K31MK_KDA_VERIFY_B57=1; claimed BITWISE). Line stated before any GPU run of this test:
# with "tsp4" added to the served c=8 verify opt (TSP4_BASE; K31_TSP4_BASE overrides), test_kda_verify's gates pass AND the
# output, raw V / K, log-g, beta rings and conv windows are bit-identical to the same opt without "tsp4" (test_kda_verify_bitwise_opts),
# and kda.verify.TSP4_BUILT shows tsp4 compiled for the arm exactly where the schedule has 4-CTA clusters and Breq <= 6 (not for the
# base, not at 8- / 2-CTA Breq). Mutations tsp4_* below must each fail it, except the control tsp4_late (delayed peer packets, wait
# kept: must PASS); tsp4_late_nowait (the same delay, the q / k wait dropped) must fail it (race catch). A mutant counts as caught
# only by an assertion or a hang, never by a build / import error.
TSP4_BASE = "early+wide+rs+lt+tsp+dmc+chunk+cg+cs+wlf+b3c+dq"  # served c=8 K31MK_KDA_VERIFY_OPT
_K = "kimi_k31/k3/cutedsl/kda/kernel.py"
MUTATIONS["tsp4_off"] = (  # engagement: tsp4 never builds -> bitwise passes trivially, TSP4_BUILT must catch it
    _K, 'self.tsp4 = "tsp4" in self.opt and tokens == 8 and self.cluster == 4 and self.rows == 32\n',
    'self.tsp4 = "tsp4" in self.opt and tokens == 8 and self.cluster == 16 and self.rows == 32\n',
)
MUTATIONS["tsp4_pick"] = (  # packet gather: tsp2's 4-token pick (token 1's lanes take token 0 for half the channels)
    _K, "            pick = ((lane + k) & 3) % TP\n", "            pick = (lane + k) & 3\n",
)
MUTATIONS["tsp4_addr"] = (  # destination slot by replica instead of token: right byte count, slots missed / doubled
    _K, "4 * (128 * (first + (lane & 3) % TP) + (ch & 124))", "4 * (128 * (first + (lane & 3) // TP) + (ch & 124))",
)
MUTATIONS["tsp4_vwin"] = (  # v window: the next input row
    _K, "\n                v_window[i] = x[token + v_token + i - 3, v_column]\n",
    "\n                v_window[i] = x[token + v_token + i - 2, v_column]\n",
)
MUTATIONS["tsp4_scale"] = (  # q / k factor: the 1/sqrt(128) scale on k instead of q
    _K, "            if section == 0:\n                factor[0] = factor[0] * scale\n            normalized[u] = qk[u] * factor[0]\n"
    "        # 16-byte packets: lane 4j + d gathers token first + d % TP's",
    "            if section == 1:\n                factor[0] = factor[0] * scale\n            normalized[u] = qk[u] * factor[0]\n"
    "        # 16-byte packets: lane 4j + d gathers token first + d % TP's",
)
MUTATIONS["tsp4_beta"] = (  # beta ring: wrong writer CTA set (entries of tokens 2, 3, 6, 7 never written: poisoned)
    _K, "\n            args[18][Int64(scratch_row), v_token, v_column, i] = v_window[i + 1]\n        if is_beta:\n"
    "            beta_act = self.activation(Float32(beta_raw[0]))\n            shared_beta[thread - 128] = beta_act\n"
    "            if (thread - 128) // TP == partition:\n",
    "\n            args[18][Int64(scratch_row), v_token, v_column, i] = v_window[i + 1]\n        if is_beta:\n"
    "            beta_act = self.activation(Float32(beta_raw[0]))\n            shared_beta[thread - 128] = beta_act\n"
    "            if (thread - 128) // TP == partition * 2:\n",
)
_TSP4_IMPORT = (_K, "    scatter_reduce,\n    shared_address,\n", "    scatter_reduce,\n    shared_address,\n    sleep_ns,\n")
_TSP4_DELAY = (  # CTAs 1-3 send their q / k packets ~2 us late
    _K, "        replica = (lane & 3) // TP  # which copy of the packet this lane holds (0 .. REP - 1)\n",
    "        replica = (lane & 3) // TP  # which copy of the packet this lane holds (0 .. REP - 1)\n"
    "        if partition != 0:\n            sleep_ns(2000)\n",
)
# Test aid for mutants whose miss depends on stale shared memory (a CTA that ran the same (head, request) before on that SM leaves the
# right q / k there): every main thread fills its CTA's shared q / k with NaN before the CTA can arrive on the cluster barrier.
_TSP4_POISON = (
    _K, "            if cutlass.const_expr(self.trigger_at_entry):\n                cute.arch.griddepcontrol_launch_dependents()\n"
    "            if cutlass.const_expr(not self.early):\n",
    "            if cutlass.const_expr(self.tsp4):\n                if thread < MAIN:\n"
    "                    for i in cutlass.range_constexpr(128 * T // MAIN):\n"
    "                        shared_q[thread + MAIN * i] = bits_float(Uint32(0x7FC00000))\n"
    "                        shared_k[thread + MAIN * i] = bits_float(Uint32(0x7FC00000))\n"
    "            if cutlass.const_expr(self.trigger_at_entry):\n                cute.arch.griddepcontrol_launch_dependents()\n"
    "            if cutlass.const_expr(not self.early):\n",
)
MUTATIONS["tsp4_addr"] = [MUTATIONS["tsp4_addr"], _TSP4_POISON]
MUTATIONS["tsp4_late"] = [_TSP4_IMPORT, _TSP4_DELAY, _TSP4_POISON]  # CONTROL: must PASS (the q / k wait covers late packets)
MUTATIONS["tsp4_late_nowait"] = [  # race: must FAIL (readers use shared q / k before the peers' packets land)
    _TSP4_IMPORT, _TSP4_DELAY, _TSP4_POISON,
    (_K, "        wait(qk_barrier)  # every token's normalized q / k rows (acquire.cluster; tsp4)\n", ""),
]


# Opt "g2" (two gate warps of 16 rows on the 32-row CTAs; claimed BITWISE). Line stated before any GPU run of this test
# (expected results): with "g2" added to the base (K31_G2_BASE, default TSP4_BASE + tsp4), test_kda_verify's gates pass AND the
# six tensors are bit-identical to the base (test_kda_verify_bitwise_opts), and kda.verify.D3G_BUILT records 2 gate warps for the arm exactly
# where the CTA has 32 rows (4-CTA clusters: Breq 4; Breq 5 / 6 with B57) and 1 elsewhere and for the base. Mutations g2_* must fail it.
MUTATIONS["g2_off"] = (_K, '        if "g2" in self.opt and tokens == 8 and self.rows == 32 and not self.local and self.gate_warps == 1:\n',
                       '        if "g2" in self.opt and tokens == 8 and self.rows == 48 and not self.local and self.gate_warps == 1:\n')
MUTATIONS["g2_rows"] = (  # gate warp 1 projects rows 8-23 instead of 16-31 (rows 8-15 twice, 24-31 never: log-g / decays missing)
    _K, "coords = (head, token, value_base + gate_rows_n * ((thread - MAIN) // 32), lane)",
    "coords = (head, token, value_base + (gate_rows_n // 2) * ((thread - MAIN) // 32), lane)",
)


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2, 3, 4, 5, 6, 8))
def test_kda_verify_g2(requests):
    if HEADS == 12:
        pytest.skip("TP16 (6-head) 32-row CTAs only")
    base_opt = os.environ.get("K31_G2_BASE", TSP4_BASE + "+tsp4")
    saved = os.environ.get("K31_BITWISE_BASE")
    os.environ["K31_BITWISE_BASE"] = base_opt
    try:
        test_kda_verify_bitwise_opts("g2", requests)
    finally:
        if saved is None:
            os.environ.pop("K31_BITWISE_BASE", None)
        else:
            os.environ["K31_BITWISE_BASE"] = saved
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base, arm = frozenset(base_opt.split("+")), frozenset((base_opt + "+g2").split("+"))
    dev = torch.device("cuda", torch.cuda.current_device())
    rows32 = V.verify_schedule(requests, Q, dev, arm, HEADS)[3] == 4  # 4-CTA clusters = 32-row CTAs (the spec)
    want = 2 if rows32 else 1
    got = V.D3G_BUILT.get((requests, Q, arm), (None, None))[1]
    assert got == want, f"g2: {got} gate warps at Breq {requests}, expected {want}"
    assert V.D3G_BUILT.get((requests, Q, base), (None, None))[1] == 1, "base arm built more than one gate warp"
    H.record({"kind": "check", "item": f"kda.verify_g2.breq{requests}", "gate_warps": want, "bitwise": True})


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2, 3, 4, 5, 6, 8))
def test_kda_verify_tsp4(requests):
    if HEADS == 12:
        pytest.skip("tsp4 is built for the TP16 (6-head) 4-CTA Breq 4 form; at 12 heads fast math switches it off")
    base_opt = os.environ.get("K31_TSP4_BASE", TSP4_BASE)
    saved = os.environ.get("K31_BITWISE_BASE")
    os.environ["K31_BITWISE_BASE"] = base_opt
    try:
        test_kda_verify_bitwise_opts("tsp4", requests)  # loads the (possibly mutated) plugin first
    finally:
        if saved is None:
            os.environ.pop("K31_BITWISE_BASE", None)
        else:
            os.environ["K31_BITWISE_BASE"] = saved
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V  # after load_plugin: the module that compiled

    base, arm = frozenset(base_opt.split("+")), frozenset((base_opt + "+tsp4").split("+"))
    dev = torch.device("cuda", torch.cuda.current_device())
    cluster4 = V.verify_schedule(requests, Q, dev, arm, HEADS)[3] == 4
    expected = cluster4 and requests <= 6  # the spec: 4-CTA clusters at Breq 1-6
    assert V.TSP4_BUILT.get((requests, Q, arm)) is expected, f"tsp4 engaged={V.TSP4_BUILT.get((requests, Q, arm))}, expected {expected}"
    assert V.TSP4_BUILT.get((requests, Q, base)) is False, "base arm compiled tsp4"
    H.record({"kind": "check", "item": f"kda.verify_tsp4.breq{requests}", "tsp4": expected, "bitwise": True})

# Opt "pkb" (f32x2 FMUL2 / FFMA2 for the decay multiply and the delta-rule update of ``update`` on the 32-row CTAs; the row
# dots stay scalar; claimed BITWISE). Line stated before any GPU run of this test: with "pkb" added to
# the base (K31_PKB_BASE, default TSP4_BASE + tsp4 + g2), test_kda_verify's gates pass AND the six tensors are bit-identical to the base
# (test_kda_verify_bitwise_opts), and kda.verify.PKB_BUILT is True for the arm exactly where the CTA has 32 rows (4-CTA clusters: Breq 4;
# Breq 5 / 6 with B57) and False elsewhere and for the base. Mutations pkb_* must fail it.
MUTATIONS["pkb_off"] = (_K, '        self.pkb = "pkb" in self.opt and self.rows == 32 and not self.packed\n',
                        '        self.pkb = "pkb" in self.opt and self.rows == 48 and not self.packed\n')
MUTATIONS["pkb_pair"] = (  # packed decay with the pair's decays swapped (element e scaled by e + 1's decay and vice versa)
    _K, "                            (decays[j][e], decays[j][e + 1]),\n", "                            (decays[j][e + 1], decays[j][e]),\n",
)


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2, 3, 4, 5, 6, 8))
def test_kda_verify_pkb(requests):
    if HEADS == 12:
        pytest.skip("TP16 (6-head) 32-row CTAs only")
    base_opt = os.environ.get("K31_PKB_BASE", TSP4_BASE + "+tsp4+g2")
    saved = os.environ.get("K31_BITWISE_BASE")
    os.environ["K31_BITWISE_BASE"] = base_opt
    try:
        test_kda_verify_bitwise_opts("pkb", requests)
    finally:
        if saved is None:
            os.environ.pop("K31_BITWISE_BASE", None)
        else:
            os.environ["K31_BITWISE_BASE"] = saved
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base, arm = frozenset(base_opt.split("+")), frozenset((base_opt + "+pkb").split("+"))
    dev = torch.device("cuda", torch.cuda.current_device())
    rows32 = V.verify_schedule(requests, Q, dev, arm, HEADS)[3] == 4  # 4-CTA clusters = 32-row CTAs (the spec)
    assert V.PKB_BUILT.get((requests, Q, arm)) is rows32, f"pkb engaged={V.PKB_BUILT.get((requests, Q, arm))} at Breq {requests}, expected {rows32}"
    assert V.PKB_BUILT.get((requests, Q, base)) is False, "base arm compiled pkb"
    H.record({"kind": "check", "item": f"kda.verify_pkb.breq{requests}", "pkb": rows32, "bitwise": True})


# Opt "occ1" (1 CTA / SM launch bound on the TP16 32-row CTAs; same code; claimed BITWISE). Line stated before any GPU run of this test
# (expected results): with "occ1" added to the base (K31_OCC1_BASE, default TSP4_BASE + tsp4 + g2 + pkb), test_kda_verify's gates
# pass AND the six tensors are bit-identical to the base, and kda.verify.CLUSTERS records occupancy 1 for the arm exactly where the CTA has 32 rows (4-CTA
# clusters: Breq 4; Breq 5 / 6 with B57) and the base's occupancy everywhere else; the base keeps 2 there. Mutation occ1_off must fail it.
MUTATIONS["occ1_off"] = (_K, '        if "occ1" in self.opt and self.cluster == 4 and self.rows == 32 and heads == 6:\n',
                         '        if "occ1" in self.opt and self.cluster == 4 and self.rows == 48 and heads == 6:\n')


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2, 3, 4, 5, 6, 8))
def test_kda_verify_occ1(requests):
    if HEADS == 12:
        pytest.skip("TP16 (6-head) 32-row CTAs only")
    base_opt = os.environ.get("K31_OCC1_BASE", TSP4_BASE + "+tsp4+g2+pkb")
    saved = os.environ.get("K31_BITWISE_BASE")
    os.environ["K31_BITWISE_BASE"] = base_opt
    try:
        test_kda_verify_bitwise_opts("occ1", requests)
    finally:
        if saved is None:
            os.environ.pop("K31_BITWISE_BASE", None)
        else:
            os.environ["K31_BITWISE_BASE"] = saved
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base, arm = frozenset(base_opt.split("+")), frozenset((base_opt + "+occ1").split("+"))
    dev = torch.device("cuda", torch.cuda.current_device())
    rows32 = V.verify_schedule(requests, Q, dev, arm, HEADS)[3] == 4  # 4-CTA clusters = 32-row CTAs (the spec)
    occ_base, occ_arm = V.CLUSTERS[(requests, Q, base)][1], V.CLUSTERS[(requests, Q, arm)][1]
    want = 1 if rows32 else occ_base
    assert occ_arm == want, f"occ1: occupancy {occ_arm} at Breq {requests}, expected {want}"
    assert not rows32 or occ_base == 2, f"base occupancy {occ_base} at Breq {requests}, expected 2"
    H.record({"kind": "check", "item": f"kda.verify_occ1.breq{requests}", "occupancy": want, "bitwise": True})


# Opt "dql" (token t - 1's q-dot butterfly + output push inside the token loop on the 32-row CTAs; same levels and order per value; claimed
# BITWISE). Line stated before any GPU run of this test: with "dql" added to the base (K31_DQL_BASE, default
# TSP4_BASE + tsp4 + g2 + pkb + occ1), test_kda_verify's gates pass AND the six tensors are bit-identical to the base, and kda.verify.DQL_BUILT is True for the
# arm exactly where the CTA has 32 rows (Breq 4; Breq 5 / 6 with B57) and False elsewhere and for the base. Mutations dql_* must fail it.
MUTATIONS["dql_off"] = (_K, '        self.dql = "dql" in self.opt and self.dq and self.rows == 32\n',
                        '        self.dql = "dql" in self.opt and self.dq and self.rows == 48\n')
MUTATIONS["dql_level"] = (  # the in-loop reduce skips its first butterfly level (tokens 0 .. T - 2 get partial sums)
    _K, "                        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):\n",
    "                        for shift in reversed(range(self.lanes_per_row.bit_length() - 2)):\n",
)


@needs_gpu
@pytest.mark.parametrize("requests", (1, 2, 3, 4, 5, 6, 8))
def test_kda_verify_dql(requests):
    if HEADS == 12:
        pytest.skip("TP16 (6-head) 32-row CTAs only")
    base_opt = os.environ.get("K31_DQL_BASE", TSP4_BASE + "+tsp4+g2+pkb+occ1")
    saved = os.environ.get("K31_BITWISE_BASE")
    os.environ["K31_BITWISE_BASE"] = base_opt
    try:
        test_kda_verify_bitwise_opts("dql", requests)
    finally:
        if saved is None:
            os.environ.pop("K31_BITWISE_BASE", None)
        else:
            os.environ["K31_BITWISE_BASE"] = saved
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    base, arm = frozenset(base_opt.split("+")), frozenset((base_opt + "+dql").split("+"))
    dev = torch.device("cuda", torch.cuda.current_device())
    rows32 = V.verify_schedule(requests, Q, dev, arm, HEADS)[3] == 4  # 4-CTA clusters = 32-row CTAs (the spec)
    assert V.DQL_BUILT.get((requests, Q, arm)) is rows32, f"dql engaged={V.DQL_BUILT.get((requests, Q, arm))} at Breq {requests}, expected {rows32}"
    assert V.DQL_BUILT.get((requests, Q, base)) is False, "base arm compiled dql"
    H.record({"kind": "check", "item": f"kda.verify_dql.breq{requests}", "dql": rows32, "bitwise": True})


#  (default off): K31MK_KDA_VERIFY_SHORTQ=1 swaps GB300's (0, 8, 2, 8) for the SM107_SHORTQ entry at Q < 8.
# Lines (written before any run of this test): (1) knob off, every table key keeps GB300's form (the served short-Q build is unchanged);
# (2) knob on, verify_schedule returns the entry and the KdaSplit is built with its cluster / CTAs per SM; (3) the knob-on build passes
# every test_kda_verify gate (independent FP32 reference) at that (Breq, Q). Mutations no_decay / no_beta / pk_no_decay must fail (3).


def _shortq_cases():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    return sorted(k[1:] for k in V.SM107_SHORTQ if k[0] == HEADS)


@needs_gpu
def test_kda_verify_shortq_table():
    H.load_plugin(MUTATIONS)
    global Q
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V

    dev = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(dev) != (10, 7) or os.environ.get("K31MK_KDA_SM107_RETUNE", "1") == "0":
        pytest.skip("SM107 retune table")
    assert not os.environ.get("K31MK_KDA_VERIFY_SCHEDULE"), "a forced schedule overrides the table"
    cases = _shortq_cases()
    if not cases:
        pytest.skip(f"no SM107_SHORTQ entry for {HEADS} heads")
    saved_q, saved_k = Q, os.environ.get("K31MK_KDA_VERIFY_SHORTQ")
    opt = V.verify_opt()
    try:
        for requests, queries in cases:
            want = V.SM107_SHORTQ[HEADS, requests, queries]
            o = V.verify_opt_for(queries, opt)
            os.environ["K31MK_KDA_VERIFY_SHORTQ"] = "0"
            assert V.verify_schedule(requests, queries, dev, o, HEADS) == (0, 8, 2, 8), (requests, queries)
            os.environ["K31MK_KDA_VERIFY_SHORTQ"] = "1"
            assert V.verify_schedule(requests, queries, dev, o, HEADS) == want, (requests, queries)
            Q = queries
            V.executable.cache_clear()
            test_kda_verify(requests)
            built = V.CLUSTERS.get((requests, queries, frozenset(o)))
            assert built is not None and built[0] == want[3] and built[1] == want[2], (requests, queries, built, want)
    finally:
        Q = saved_q
        if saved_k is None:
            os.environ.pop("K31MK_KDA_VERIFY_SHORTQ", None)
        else:
            os.environ["K31MK_KDA_VERIFY_SHORTQ"] = saved_k
        V.executable.cache_clear()
