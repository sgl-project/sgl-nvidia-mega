// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MoE-LD-W: routed NVFP4 experts of one expert-parallel rank for verify batches of 1..512 token rows, in one launch of one
// CTA per SM that streams every active local expert's weights once per pass of up to 128 routed rows.
//
// Prologue (every CTA, after griddepcontrol.wait): the local routed slots are counted per expert (per 256-slot chunk), giving
// expert bases (rows in expert-major, token order, as the stock expanded-index map orders them) and passes of at most 128
// rows; each CTA builds its work list (pass, weight tile) of FC1 (128 gate/up rows, full K) and FC2 (128 output rows, full K)
// items, dealt per locality domain as in moe_ld.cuh (each domain streams the expert row halves placed on it); the weights
// of its first ring stages go out at once. Warps 4-11 copy the token rows of slots b, b + G, ... into the expert-major
// buffer with their per-token scales and token-scale images (the 32-lane tensor-memory layout of block scale factors: row
// t of a pass in lane t % 32, column t / 32 of each 64-wide K chunk's 4-column group, PTX ISA tcgen05 block scaling), write
// the expanded-index map and top-k weights of those slots, and release one count of the grid's rows-ready counter; token-row
// copies of any CTA follow the acquire of that counter (all CTAs) and a proxy fence.
// Every stage carries 32 KiB of weights + their scales and the pass's token rows (16 / 32 / 64 / 128 rows, the next bucket
// above the pass's row count) + their scale images; the MMA is M = 128 x N = that bucket; every output element is accumulated
// over the full K in K order, as the stock kernel does (tests/test_moe_ldw.py: every routed FC2 row bitwise).
// FC1 epilogue: the 16x256b tensor-memory fragment gives each thread the up and the gate row of a pair (no shuffles);
// SwiGLU (stock arithmetic) -> activation rows. FC1 -> FC2: per pass, the rows of the FC2 operand are dealt over the CTAs
// holding the pass's FC2 tiles (tile c: rows c, c + 32, c + 64, c + 96) and round-robin over their four builder warps; each
// row is built (FlashInfer's per-token NVFP4 recipe) once the pass's FC1 tiles are published, then one count is released;
// FC2 stages of the pass are copied after that count is acquired and a proxy fence. Builders wait only on FC1, which waits
// on nothing outside its CTA, so the waits are acyclic. Counters come in two sets used by alternating launches; a launch
// zeroes the other set for the next one. Code that runs once per launch (prologue, row copies) is kept small and out of line.
// Every wait is bounded (poll count, sticky error word); outstanding copies and MMAs are drained before exit, and a CTA
// whose wait expired then traps (MUSE_VR_LDW_TRAP, default 1), so an expired wait is never a silent result.
// Residency: start tickets as in moe_ld.cuh. Slots: claimed exactly as in moe_ld.cuh (slot_claim.cuh: the SM's slot is
// only the preferred one; one claim per CTA, kept in shared memory for every role; the previous form only with
// -DMUSE_VR_SLOT_UNSAFE=1). The claim area is words kClaimWord.. of the control tensor, which the launches that share
// it already order by completion.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "ld_queue.cuh"
#include "slot_claim.cuh"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

// Warp roles are inlined into the kernel body (measured faster in chain A/B than separate role functions);
// -DMUSE_VR_LDW_ROLE=__noinline__ keeps them separate (e.g. to read per-role code size).
// weight stages (beyond the ring) whose bytes are prefetched into L2 right after the first stages go out: the ring cannot
// take them before the token rows of the first pass are copied, so their DRAM reads would otherwise wait (0: off)
#ifndef MUSE_VR_LDW_HEAD_PF
#define MUSE_VR_LDW_HEAD_PF 0
#endif
#ifndef MUSE_VR_LDW_ROLE
#define MUSE_VR_LDW_ROLE __forceinline__
#endif
#ifndef MUSE_VR_LDW_IDLE_NS
#define MUSE_VR_LDW_IDLE_NS 256
#endif
#ifndef MUSE_VR_LDW_TRAP
#define MUSE_VR_LDW_TRAP 1
#endif
// Default off. 1 = publish the slot table for the dependent grid: after the copy phase (the CTA's share of the expanded-index map
// and top-k weights stored, the workers' named barrier), worker 0 issues fence.acq_rel.gpu then griddepcontrol.launch_dependents,
// and the tail trigger is not used, so every CTA triggers only after its share is stored and fenced; the dependent may then read
// the map and the weights before its griddepcontrol.wait. Outputs are unchanged. Non-landed, non-balanced builds only.
#ifndef MUSE_VR_LDW_EARLY_SLOTTAB
#define MUSE_VR_LDW_EARLY_SLOTTAB 0
#endif
// Default off. 1 (with EARLY_SLOTTAB): rows-ready flags for a dependent grid that consumes FC2 rows while this grid runs. Each
// launch takes an epoch from a 64-bit ticket (every CTA adds 1; epoch = old / gridDim.x + 1, the same in every CTA, strictly
// increasing); the copy phase also stores each routed slot's pass index (e2pass) and CTA 0 stores the epoch (rr_pub), both
// before the fence + trigger; every FC2 epilogue warp, after its last row stores of an item, stores flag [pass][tile][warp] =
// epoch with release semantics. Flags are never reset: an earlier launch's flag holds a smaller epoch. Outputs are unchanged.
#ifndef MUSE_VR_LDW_ROWS_READY
#define MUSE_VR_LDW_ROWS_READY 0
#endif
#if MUSE_VR_LDW_ROWS_READY && !MUSE_VR_LDW_EARLY_SLOTTAB
#error "MUSE_VR_LDW_ROWS_READY needs MUSE_VR_LDW_EARLY_SLOTTAB"
#endif
#if MUSE_VR_LDW_ROWS_READY && MUSE_VR_LDW_TEST_PREWAIT_CLAIM
#error "MUSE_VR_LDW_ROWS_READY: not with the pre-wait claim test form"
#endif
// Tests only (rows-ready builds; default 0): MUSE_VR_LDW_TEST_RR_DELAY_NS = CTA 0's epilogue warps sleep this long before their
// first FC2 item's row stores (rows and flags late); MUSE_VR_LDW_TEST_RR_PLANT = 1 / 2: CTA 0's first FC2 item's flags carry
// epoch - 1 / epoch + 1 (a stale / a newer flag the consumer must never accept)
#ifndef MUSE_VR_LDW_TEST_RR_DELAY_NS
#define MUSE_VR_LDW_TEST_RR_DELAY_NS 0
#endif
#ifndef MUSE_VR_LDW_TEST_RR_PLANT
#define MUSE_VR_LDW_TEST_RR_PLANT 0
#endif
#if (MUSE_VR_LDW_TEST_RR_DELAY_NS || MUSE_VR_LDW_TEST_RR_PLANT) && !MUSE_VR_LDW_ROWS_READY
#error "MUSE_VR_LDW_TEST_RR_*: rows-ready builds only"
#endif
// Tests only: 1 = the start ticket and the slot claim are issued BEFORE griddepcontrol.wait (the earlier placement; the
// negative control of tests/test_moe_ldw_order.py). Default 0: both are issued right after the wait (prologue).
#ifndef MUSE_VR_LDW_TEST_PREWAIT_CLAIM
#define MUSE_VR_LDW_TEST_PREWAIT_CLAIM 0
#endif
// Tests only (sanitizer runs): launch this many CTAs (two domain halves) instead of one per SM, so a sanitizer's host memory,
// which grows with the resident grid, stays bounded. The slot claim treats SM hints outside the smaller domains as misses.
#ifndef MUSE_VR_LDW_TEST_GRID
#define MUSE_VR_LDW_TEST_GRID 0
#endif
#ifndef MUSE_VR_LDW_MAX_POLLS
#define MUSE_VR_LDW_MAX_POLLS (1u << 24)
#endif
#ifndef MUSE_VR_LDW_K128
#define MUSE_VR_LDW_K128 0  // 1: K = 128 block-scaled MMAs (two 64-wide chunks per instruction)
#endif
// Development build (default 0): 1 = the routed ids and the token rows come straight from the landing buffers of the
// hop transfers that carry them to this rank (no stock tensors in between). Ids: 8 aligned 64-bit words per row, the
// transfer's 32-bit tag in bits 63..32 and the packed id in bits 31..0. Token rows: aligned 64-bit words of 6 payload
// bytes (bits 47..0, row byte 6 w + b in bits 8 b..8 b + 7 of word w) and the transfer's 16-bit tag in bits 63..48;
// row bytes [0, 2048) FP4, [2048, 2304) block scales, [2304, 2308) the per-token scale. Both landings use the
// [copy][source rank][row][pitch] layout of the hop sites; the copy and both tags derive from the forward sequence word
// on the device. Each word is accepted on its own matching 64-bit relaxed load (bounded polls; a word carrying a newer
// tag or an expired poll sets the error word). The first FC1 item's token operand is written by warps 4-7 straight
// into its ring stages; the copy phase (warps 8-11) builds the permuted rows of the later items.
#ifndef MUSE_VR_LDW_LANDED
#define MUSE_VR_LDW_LANDED 0
#endif
// Tests only (landed build): 1 = a head loader waits for an impossible tag on one word of stage 0 (forced expiry at the
// head item), 2 = the copy phase does so on one word (forced expiry at a later item); MUSE_VR_LDW_TEST_HEAD_DELAY = ns the
// head loaders sleep before their last stage's stores; MUSE_VR_LDW_TEST_SKIP_HEAD_DONE = 1 drops the producer's head barrier
// wait (the non-vacuity twin of that edge). All default 0.
#ifndef MUSE_VR_LDW_TEST_EXPIRE
#define MUSE_VR_LDW_TEST_EXPIRE 0
#endif
#ifndef MUSE_VR_LDW_TEST_HEAD_DELAY
#define MUSE_VR_LDW_TEST_HEAD_DELAY 0
#endif
#ifndef MUSE_VR_LDW_TEST_SKIP_HEAD_DONE
#define MUSE_VR_LDW_TEST_SKIP_HEAD_DONE 0
#endif
// landed build: the named barrier (head loader warps 4-7 arrive, the producer warp waits) that orders the loaders' stores into
// the ring before the producer's later copies into those slots; a fused kernel that owns the barrier table may set another id
#ifndef MUSE_VR_LDW_HEAD_BAR
#define MUSE_VR_LDW_HEAD_BAR 8
#endif
// Landed build, start order (development, default 0 = the form above). MUSE_VR_LDW_HEAD_PRIO: the head loaders issue their
// first stage's token words (payload and scales) as ONE batch per thread, and the producer issues the weights of ring stages
// 1.. only after that batch is issued (1) or has landed (2); stage 0's weights go out at once. Kept for diagnostics: holding
// the later stages measured slower end to end than issuing them at once (start diagnostics), so it stays off.
// MUSE_VR_LDW_START = 1PGC (four decimal digits, a leading 1): HEAD_PRIO = P, HEAD_GO = G, PRO_COMPACT = C in one short flag
#ifdef MUSE_VR_LDW_START
#if MUSE_VR_LDW_START < 1000 || MUSE_VR_LDW_START > 1999
#error "MUSE_VR_LDW_START: 1PGC"
#endif
#define MUSE_VR_LDW_HEAD_PRIO ((MUSE_VR_LDW_START / 100) % 10)
#define MUSE_VR_LDW_HEAD_GO ((MUSE_VR_LDW_START / 10) % 10)
#define MUSE_VR_LDW_PRO_COMPACT (MUSE_VR_LDW_START % 10)
#endif
#ifndef MUSE_VR_LDW_HEAD_PRIO
#define MUSE_VR_LDW_HEAD_PRIO 0
#endif
// Tests only (timing diagnostics of the head batch; default 0): MUSE_VR_LDW_TEST_HEAD_TWICE = 1 runs the stage-0 batch twice
// (same code, same data, same stores); MUSE_VR_LDW_TEST_HEAD_LDGPU = 1 issues its loads at .gpu scope (same-GPU writers only)
#ifndef MUSE_VR_LDW_TEST_HEAD_TWICE
#define MUSE_VR_LDW_TEST_HEAD_TWICE 0
#endif
#ifndef MUSE_VR_LDW_TEST_HEAD_LDGPU
#define MUSE_VR_LDW_TEST_HEAD_LDGPU 0
#endif
// MUSE_VR_LDW_HEAD_GO = 1: the head loaders start on a CTA-local flag that warp 3 publishes with the head-row list (release /
// acquire, CTA scope) instead of the end-of-prologue barrier, and the producer issues stage 0's weights as soon as its work
// list is written; both join the other roles through barriers they only arrive at, before anything those roles wrote is read.
#ifndef MUSE_VR_LDW_HEAD_GO
#define MUSE_VR_LDW_HEAD_GO 0
#endif
// the barrier the head loaders wait at (the other warps arrive) before the rest of their roles; settable like HEAD_BAR
#ifndef MUSE_VR_LDW_LIST_BAR
#define MUSE_VR_LDW_LIST_BAR 10
#endif
// MUSE_VR_LDW_PRO_COMPACT = 1: a smaller serial prologue: the routed ids are read in a rolled loop of groups of words in
// flight (instead of every possible word unrolled), per-expert prefixes run over the used chunks only, and the FC2 operand's
// padded pass rows are a warp scan off the work list's path (the start is bound by the code it runs once; measured in the
// start diagnostics; tunable).
#ifndef MUSE_VR_LDW_PRO_COMPACT
#define MUSE_VR_LDW_PRO_COMPACT 0
#endif
#if (MUSE_VR_LDW_HEAD_PRIO || MUSE_VR_LDW_HEAD_GO) && (!MUSE_VR_LDW_LANDED || MUSE_VR_LDW_BALB)
#error "MUSE_VR_LDW_HEAD_PRIO / HEAD_GO need the landed build without the board build"
#endif
#if MUSE_VR_LDW_PRO_COMPACT && MUSE_VR_LDW_BALB
#error "MUSE_VR_LDW_PRO_COMPACT is not built for the board build"
#endif
#if MUSE_VR_LDW_HEAD_PRIO < 0 || MUSE_VR_LDW_HEAD_PRIO > 2
#error "MUSE_VR_LDW_HEAD_PRIO: 0, 1 or 2"
#endif
// Tests only: 1 = every head loader warp writes the per-token scales into warp 0's copy (a planted cross-warp race, the
// racecheck control; results are not checked)
#ifndef MUSE_VR_LDW_TEST_PLANT_RACE
#define MUSE_VR_LDW_TEST_PLANT_RACE 0
#endif
// Development build (default 0; with the landed build and a ring build): N = 4 or 8 expert-parallel ranks over a fixed replica
// pool. The step's choice of holder for each replicated expert arrives as a board written on this GPU by the front stage: 32
// tagged words of 4-bit holder codes (8 experts per word) and a digest word, all tagged with the step's ids tag. Pinned experts
// (one holder) are local from the ids alone and start at once; warp 3 reads the board once the pinned work list is out and
// appends this rank's assigned replicated experts (pinned-first form). The routed math per (token, expert) is unchanged.
#ifndef MUSE_VR_LDW_BALB
#define MUSE_VR_LDW_BALB 0
#endif
// Development build (default 0; with the landed build): the expert stage also finalizes. The FC2 epilogue stores, per routed
// row and per (FC2 tile, epilogue warp), the partial sum of squares of the post-expert norm's scaled input (weight-first
// RMSNorm: (w x y)^2 over the warp's 32 columns, y the bf16 FC2 output); then one release per row on its token's completion
// counter. The builder warps, after their FC2-operand rows, own the tokens t = blockIdx.x mod grid: once a token's counter shows
// all of its local routes' tiles, they reduce each route's partials in a fixed order, and write the token's weighted sum over its
// local routes in slot order (fp32, one rounding to bf16) to tok_out (zero rows for tokens with no local route). The moments'
// reduction order differs from the stock finalize kernel's (a different column partition): reduction-order class.
#ifndef MUSE_VR_LDW_EPI
#define MUSE_VR_LDW_EPI 0
#endif
#if MUSE_VR_LDW_EPI && !MUSE_VR_LDW_LANDED
#error "MUSE_VR_LDW_EPI needs the landed build"
#endif
// Development build (default 0; served builds): MUSE_VR_LDW_HB0_PUSH = W (2, 4 or 8, the tensor-parallel world). The expert
// stage also sends its routed rows to the other ranks (the route all-gather form of the expert-output hop, c3_hb0.cuh). After
// the FC2 epilogue of each of this CTA's FC2 items (one pass x one 128-column tile), the builder warps store that tile of every
// row of the pass into every OTHER rank's receive region of a hop site at linear row t * 8 + k (the row's routed slot): 22
// six-value groups per tile (the last one: columns 126..127 and four zero pads) = 44 tagged words in the tile's own three
// lines (c3_hb0.cuh layout; the pad words are never stored). The values are the bf16 output rows unchanged; out is
// written as before (the consumer reads this rank's own slots from it). Ring sites:
// the slot's credits are acquired before the first store. An aborted launch stores nothing more.
#ifndef MUSE_VR_LDW_HB0_PUSH
#define MUSE_VR_LDW_HB0_PUSH 0
#endif
#if MUSE_VR_LDW_HB0_PUSH && (MUSE_VR_LDW_LANDED || MUSE_VR_LDW_BALB || MUSE_VR_LDW_EPI || \
                             (MUSE_VR_LDW_HB0_PUSH != 2 && MUSE_VR_LDW_HB0_PUSH != 4 && MUSE_VR_LDW_HB0_PUSH != 8))
#error "MUSE_VR_LDW_HB0_PUSH = the world (2, 4 or 8), served builds only"
#endif
// Served builds (development, default 0): MUSE_VR_LDW_SPLIT_REL = 1 splits the copy phase's release in two. The rows-ready count
// is raised once every worker's token-row and scale-image stores are done (barrier 7), before the per-slot metadata stores
// (expanded-index map, top-k weight, per-row token scale); a second grid count (kRowsReady + 1) is raised after those
// (barrier 7 again). The producer's token copies acquire the first count, every reader of the metadata the second.
#ifndef MUSE_VR_LDW_SPLIT_REL
#define MUSE_VR_LDW_SPLIT_REL 0
#endif
// Served ring builds (development, default 0): MUSE_VR_LDW_THEAD = 1 gives the CTA's first FC1 item (when its pass has at most
// MUSE_VR_LDW_HEAD_ROWS rows) a token path of its own, without the copy warps: the producer gathers the item's token rows from
// the token tensor with tensor-copy gathers of four rows (a 2D map of the rows, 128-byte swizzle; a destination at row 4 of a
// swizzle atom takes the swizzle base offset of that row, so the rows land where the row map's box puts them), and warp 3,
// before its FC1 publication, writes the item's stages' token-scale images (the copy phase's image layout) and arrives on each
// stage's full barrier (the producer gives that second arrival on every other stage). The copy phase is unchanged.
#ifndef MUSE_VR_LDW_THEAD
#define MUSE_VR_LDW_THEAD 0
#endif
#if MUSE_VR_LDW_SPLIT_REL && (MUSE_VR_LDW_LANDED || MUSE_VR_LDW_BALB)
#error "MUSE_VR_LDW_SPLIT_REL: served builds only"
#endif
// Served builds (development, default 0): MUSE_VR_LDW_COPY_DIET = 1 gives the copy phase its operands by value. The calling
// roles pass the kernel's pointers and the shared-window address of the CTA's slot tables, and the one out-of-line copy
// reads the tables with shared loads (no load through a reference to the parameters or to shared memory). Same work, same
// stores, same barriers and releases.
#ifndef MUSE_VR_LDW_COPY_DIET
#define MUSE_VR_LDW_COPY_DIET 0
#endif
#if MUSE_VR_LDW_COPY_DIET && (MUSE_VR_LDW_LANDED || MUSE_VR_LDW_BALB)
#error "MUSE_VR_LDW_COPY_DIET: served builds only"
#endif
#if MUSE_VR_LDW_EARLY_SLOTTAB && (MUSE_VR_LDW_COPY_DIET || MUSE_VR_LDW_SPLIT_REL || MUSE_VR_LDW_THEAD)
#error "MUSE_VR_LDW_EARLY_SLOTTAB: not with the by-value copy phase, the split release or the tensor-copy head (their copy phases publish differently)"
#endif
// HB0 push: the barrier the four builder warps meet at after the credit wait (settable like HEAD_BAR)
#ifndef MUSE_VR_LDW_HB0_BAR
#define MUSE_VR_LDW_HB0_BAR 13
#endif
// Tests only (HB0 push; default 0): TEST_HB0_PARTIAL = 1 stores only the first half of each CTA's FC2 items (then marks the
// launch failed: a producer that aborts mid-way), 2 only the first half of each tile's groups; TEST_HB0_DELAY_NS = epilogue warp 1 waits
// that long before its FC2 row stores of each item; TEST_HB0_SKIP_ACQ = 1 the push warps do not wait for the epilogue's release
// (the planted race of the non-vacuity pair).
#ifndef MUSE_VR_LDW_TEST_HB0_PARTIAL
#define MUSE_VR_LDW_TEST_HB0_PARTIAL 0
#endif
#ifndef MUSE_VR_LDW_TEST_HB0_DELAY_NS
#define MUSE_VR_LDW_TEST_HB0_DELAY_NS 0
#endif
#ifndef MUSE_VR_LDW_TEST_HB0_SKIP_ACQ
#define MUSE_VR_LDW_TEST_HB0_SKIP_ACQ 0
#endif
// Tests only (board build): > 0 enables an emulated board arrival: the board phase first waits until Params.test_delay_ns ns after
// the CTA's dependency wait returned (the board's arrival time in a one-kernel front + expert launch); default 0 (no wait)
#ifndef MUSE_VR_LDW_TEST_BOARD_DELAY_NS
#define MUSE_VR_LDW_TEST_BOARD_DELAY_NS 0
#endif
// (defined before the guards below that test it)
// 1: the token rows and scale images of each CTA's first FC1 items (up to kHeadItems, while their pass has at most
// kHeadRows rows) are copied straight from the token tensors by row index (asynchronous copies tracked by each stage's
// barrier), so those items do not wait for the copy phase and its grid-wide rows-ready count; later items use the copy
// phase's permuted rows as before; 0: every item waits for the copy phase
#ifndef MUSE_VR_LDW_GATHER
#define MUSE_VR_LDW_GATHER 1
#endif
#if MUSE_VR_LDW_EARLY_SLOTTAB && (MUSE_VR_LDW_LANDED || MUSE_VR_LDW_BALB)
#error "MUSE_VR_LDW_EARLY_SLOTTAB: non-landed, non-balanced builds only"
#endif
#if MUSE_VR_LDW_BALB
#if MUSE_VR_LDW_BALB != 4 && MUSE_VR_LDW_BALB != 8
#error "MUSE_VR_LDW_BALB is the expert-parallel world: 4 or 8"
#endif
#if !MUSE_VR_LDW_LANDED || MUSE_VR_LDW_GATHER
#error "MUSE_VR_LDW_BALB needs the landed build and a ring build (MUSE_VR_LDW_GATHER=0)"
#endif
#endif
#if MUSE_VR_LDW_LANDED
#define MUSE_VR_LDW_HEADS 1            // leading FC1 items with a token path of their own
#define MUSE_VR_LDW_FULL_ARRIVALS 1    // the producer warp (weights); the head loaders arrive on tok_full
#else
#define MUSE_VR_LDW_HEADS (MUSE_VR_LDW_GATHER || MUSE_VR_LDW_THEAD)
#define MUSE_VR_LDW_FULL_ARRIVALS (MUSE_VR_LDW_THEAD ? 2 : 1)  // THEAD: + warp 3 (head stages) / the producer (others)
#endif
#if MUSE_VR_LDW_THEAD && (MUSE_VR_LDW_LANDED || MUSE_VR_LDW_BALB || MUSE_VR_LDW_GATHER)
#error "MUSE_VR_LDW_THEAD: served ring builds only (MUSE_VR_LDW_GATHER=0)"
#endif
// Served builds without head copies (development, default 0): MUSE_VR_LDW_EARLY_COPY = 1 starts the copy phase before the work
// list. Warp 0 publishes the expert bases, the pass bases and the counter parity right after its expert loop by arriving on a named
// barrier (MUSE_VR_LDW_ECG_BAR, 32 arrivals + 256 waiters); warps 4-11 arrive at the pass barrier instead of waiting there, wait on
// that barrier, compute their slot positions (warps 8-11) or zero the other parity's counters (warps 4-7) as before, arrive at the
// end-of-prologue barrier 3, meet at MUSE_VR_LDW_EC_BAR (256 threads), run the unchanged copy phase, then wait at the list barrier
// (MUSE_VR_LDW_LIST_BAR) that warps 0-3 arrive at after barrier 3. The same bytes go to the same addresses with the same single
// release per CTA; every access stays after griddepcontrol.wait, and every new edge is a CTA barrier. Not with the forms whose
// prologue or copy phase differs (landed, board, finalize, start order, head copies, tensor-copy head, early slot table, rows-ready
// flags, expert-output hop push): each is refused below.
#ifndef MUSE_VR_LDW_EARLY_COPY
#define MUSE_VR_LDW_EARLY_COPY 0
#endif
#if MUSE_VR_LDW_EARLY_COPY
#if MUSE_VR_LDW_LANDED
#error "MUSE_VR_LDW_EARLY_COPY: not with the landed build (MUSE_VR_LDW_LANDED)"
#endif
#if MUSE_VR_LDW_BALB
#error "MUSE_VR_LDW_EARLY_COPY: not with the board build (MUSE_VR_LDW_BALB)"
#endif
#if MUSE_VR_LDW_EPI
#error "MUSE_VR_LDW_EARLY_COPY: not with the finalize build (MUSE_VR_LDW_EPI)"
#endif
#if defined(MUSE_VR_LDW_START) || MUSE_VR_LDW_HEAD_GO || MUSE_VR_LDW_HEAD_PRIO
#error "MUSE_VR_LDW_EARLY_COPY: not with the start-order forms (MUSE_VR_LDW_START / HEAD_GO / HEAD_PRIO)"
#endif
#if MUSE_VR_LDW_THEAD
#error "MUSE_VR_LDW_EARLY_COPY: not with the tensor-copy head (MUSE_VR_LDW_THEAD)"
#endif
#if MUSE_VR_LDW_EARLY_SLOTTAB
#error "MUSE_VR_LDW_EARLY_COPY: not with the early slot table (MUSE_VR_LDW_EARLY_SLOTTAB)"
#endif
#if MUSE_VR_LDW_ROWS_READY
#error "MUSE_VR_LDW_EARLY_COPY: not with the rows-ready flags (MUSE_VR_LDW_ROWS_READY)"
#endif
#if MUSE_VR_LDW_HB0_PUSH
#error "MUSE_VR_LDW_EARLY_COPY: not with the expert-output hop push (MUSE_VR_LDW_HB0_PUSH)"
#endif
#if MUSE_VR_LDW_HEADS
#error "MUSE_VR_LDW_EARLY_COPY: builds without head copies only (MUSE_VR_LDW_GATHER=0)"
#endif
#endif
// early copy: the hand-off barrier (warp 0 arrives, warps 4-11 wait) and the copy warps' barrier; settable like HEAD_BAR
#ifndef MUSE_VR_LDW_ECG_BAR
#define MUSE_VR_LDW_ECG_BAR 14
#endif
#ifndef MUSE_VR_LDW_EC_BAR
#define MUSE_VR_LDW_EC_BAR 15
#endif
// Tests only (early-copy builds; default 0). MUSE_VR_LDW_TEST_EC_COUNT = 1: every participant of the two early-copy barriers
// also counts itself in shared memory and thread 0 checks the counts at the end (error bit kErrEcCount on a mismatch).
// MUSE_VR_LDW_TEST_EC_TWIN = 1: the hand-off barrier is dropped (no arrival, no wait); 2: the copy warps' barrier is dropped
// (the two racecheck controls; in its place the dropping warps wait MUSE_VR_LDW_TEST_EC_WAIT_CYCLES clock cycles, default 0, so
// the reads usually find the values though nothing orders them). With the hand-off dropped (TWIN = 1), every value warps 4-11
// take from warp 0's publication (expert bases, pass bases, counter parity, and the positions and pass indices computed with
// them) is clamped into its valid range, so the launch completes (its rows may be wrong) and racecheck can report the unordered
// accesses. MUSE_VR_LDW_TEST_EC_PLANT = 1: warps 8-11 arrive at barrier 3
// BEFORE their slot positions (the last reads of the prologue scratch on the copy's path) and then wait
// MUSE_VR_LDW_TEST_EC_WAIT_CYCLES clock cycles (default 200000), so the producer's copy into the last ring stage lands under
// those reads (the ordering control; outputs are expected to differ). MUSE_VR_LDW_TEST_EC_PLANT = 2: the hand-off barrier is
// dropped on both sides and warp 0 waits MUSE_VR_LDW_TEST_EC_WAIT_CYCLES clock cycles before it writes its publication, while
// warps 4-11 proceed without waiting (the hand-off control; no clamps; outputs are expected to differ or the launch to fault).
#ifndef MUSE_VR_LDW_TEST_EC_COUNT
#define MUSE_VR_LDW_TEST_EC_COUNT 0
#endif
#ifndef MUSE_VR_LDW_TEST_EC_TWIN
#define MUSE_VR_LDW_TEST_EC_TWIN 0
#endif
#ifndef MUSE_VR_LDW_TEST_EC_PLANT
#define MUSE_VR_LDW_TEST_EC_PLANT 0
#endif
#ifndef MUSE_VR_LDW_TEST_EC_WAIT_CYCLES
#define MUSE_VR_LDW_TEST_EC_WAIT_CYCLES (MUSE_VR_LDW_TEST_EC_PLANT ? 200000 : 0)
#endif
#if (MUSE_VR_LDW_TEST_EC_COUNT || MUSE_VR_LDW_TEST_EC_TWIN || MUSE_VR_LDW_TEST_EC_PLANT) && !MUSE_VR_LDW_EARLY_COPY
#error "MUSE_VR_LDW_TEST_EC_*: early-copy builds only"
#endif
#if MUSE_VR_LDW_TEST_EC_TWIN < 0 || MUSE_VR_LDW_TEST_EC_TWIN > 2 || (MUSE_VR_LDW_TEST_EC_TWIN && MUSE_VR_LDW_TEST_EC_PLANT)
#error "MUSE_VR_LDW_TEST_EC_TWIN: 0, 1 or 2, one control per build"
#endif
#if MUSE_VR_LDW_TEST_EC_PLANT < 0 || MUSE_VR_LDW_TEST_EC_PLANT > 2
#error "MUSE_VR_LDW_TEST_EC_PLANT: 0, 1 or 2, one control per build"
#endif

// Every build variant (compile flags) gets its own symbol names (the ops layer sets MUSE_VR_LDW_ABI per variant), so
// variants loaded in one process never share a kernel or a launch-configuration static.
#ifndef MUSE_VR_LDW_ABI
#define MUSE_VR_LDW_ABI v0
#endif
#if MUSE_VR_LDW_HB0_PUSH
#include "c3_hb0.cuh"  // expert-output hop, route all-gather form: tile-aligned row layout and word helpers
#endif
namespace muse_vr_ldw {
inline namespace MUSE_VR_LDW_ABI {

constexpr uint32_t kMaxPolls = MUSE_VR_LDW_MAX_POLLS;
// ---------------------------------------------------------------- MoE-LD-W problem shape
constexpr int kHidden = 4096;  // FC1 K, FC2 N
constexpr int kInter = 4096;   // FC1 N / 2, FC2 K
constexpr int kTopK = 8;
// Rows per call: 512 (served default) or 1024 (-DMUSE_VR_LDW_MAX_TOKENS=1024, the wide build; its scratch is sized for it)
#ifndef MUSE_VR_LDW_MAX_TOKENS
#define MUSE_VR_LDW_MAX_TOKENS 512
#endif
constexpr int kMaxTokens = MUSE_VR_LDW_MAX_TOKENS;
static_assert(kMaxTokens == 512 || kMaxTokens == 1024, "rows per call: 512 or 1024");
constexpr int kMaxSlots = kMaxTokens * kTopK;  // routed slots per launch (all ranks' ids are read; local ones kept)
constexpr int kMaxPool = 128;
#ifdef MUSE_VR_LDW_PASS_ROWS
// rows of one expert pass = largest MMA N used, set by the ring builds (-DMUSE_VR_LDW_PASS_ROWS=32 / 48 with
// -DMUSE_VR_LDW_STAGES=4: a smaller token region per stage, so one more stage of weights is in flight)
constexpr int kPassRows = MUSE_VR_LDW_PASS_ROWS;
static_assert(kPassRows == 32 || kPassRows == 48 || kPassRows == 64 || kPassRows == 128, "pass rows");
#else
constexpr int kPassRows = 128;                 // rows of one expert pass = largest MMA N used
#endif
#if MUSE_VR_LDW_MAX_TOKENS > 512
// wide build: at most 256 passes (the work item's pass index has 8 bits); a call that needs more fails loudly (kErrShape).
// A pool of at most 64 local experts never needs more (64 + 8192 / 48 < 256).
constexpr int kMaxPasses = (kMaxPool + kMaxSlots / kPassRows) < 256 ? (kMaxPool + kMaxSlots / kPassRows) : 256;
#else
constexpr int kMaxPasses = kMaxPool + kMaxSlots / kPassRows;
#endif
// FC2 token rows (hs2) are stored at bucket-padded per-pass bases: pass q starts at the sum of bucket_rows(bucket_of(rows)) of the
// passes before it, so an FC2 operand box (bucket_rows rows from the pass base) never covers a row of another pass. Padding of a
// pass is at most max(15, its rows) rows, hence at most 2 kMaxSlots + 16 kMaxPasses rows in all.
constexpr int kHs2Rows = 2 * kMaxSlots + 16 * kMaxPasses;
#ifdef MUSE_VR_LDW_PASS_ROWS
static_assert(kMaxPasses <= 256, "pass index bits of a work item");
#endif
constexpr int kTile = 128;
#if MUSE_VR_LDW_ROWS_READY
constexpr int kRrPerPass = (kHidden / kTile) * 4;                             // flags per pass: FC2 tiles x epilogue warps
constexpr int kRrFlags = kMaxPasses * kRrPerPass;
constexpr int kRrLast = 1024;                                                  // the consumer's per-CTA last epochs (not ours)
constexpr int kRrFlagBase = 2 + kRrLast;
constexpr int kRrWords = kRrFlagBase + kRrFlags + kMaxTokens * kTopK / 4;      // + e2pass (int16, 4 per word)
#endif
constexpr int kStageW = 32 * 1024;
constexpr int kChunks = 8, kStagesPerTile = kHidden / (64 * kChunks);
constexpr int kStageA = kStageW + kChunks * 512;  // weights (128 rows x 512 K) + their block scales
constexpr int kStageB = kPassRows * 256;          // token rows (up to 128) x 512 K
constexpr int kStageImg = kChunks * 512;          // token-scale images of 8 K chunks
constexpr int kStageBytes = kStageA + kStageB + kStageImg;
#ifdef MUSE_VR_LDW_STAGES
constexpr int kStages = MUSE_VR_LDW_STAGES;
#else
constexpr int kStages = 3;
#endif
#ifndef MUSE_VR_LDW_HEAD_ITEMS
#define MUSE_VR_LDW_HEAD_ITEMS 1
#endif
#ifndef MUSE_VR_LDW_HEAD_ROWS
#define MUSE_VR_LDW_HEAD_ROWS 32
#endif
#if MUSE_VR_LDW_LANDED
// landed build: the head loaders take a whole pass (any row count) of the first FC1 item. The head item is item 0, so its stages
// are the launch's stages k = 0 .. kStagesPerTile - 1, contiguous from the first use of every ring slot: every earlier use of a
// slot by a head stage is a head use, and the readers' tok_full parity (k / kStages) & 1 counts exactly those uses.
constexpr int kHeadItems = 1, kHeadRows = kPassRows;
static_assert(kHeadItems == 1, "the tok_full parity assumes the head stages are the launch's first stages");
#else
constexpr int kHeadItems = MUSE_VR_LDW_HEAD_ITEMS, kHeadRows = MUSE_VR_LDW_HEAD_ROWS;
#endif
static_assert(kHeadItems >= 1 && kHeadItems <= 4 && kHeadRows >= 1 && kHeadRows <= 128, "head items");
#if MUSE_VR_LDW_THEAD
// THEAD: warp 3 writes the images of item 0 only (its stages are the launch's first stages); the image loop takes 8 K chunks
// per stage; a stage's token region starts on a 1024-byte boundary and every bucket height is a multiple of 8 rows, so each
// K atom (a * nb * 128) starts a 128-byte-swizzle atom
static_assert(kHeadItems == 1, "MUSE_VR_LDW_THEAD: one head item");
static_assert(kChunks == 8, "MUSE_VR_LDW_THEAD: 8 K chunks per stage");
#endif
constexpr int kFc1TilesPerExpert = 2 * kInter / kTile, kFc2TilesPerExpert = kHidden / kTile;
constexpr int kFc1TilesPerDomain = kFc1TilesPerExpert / 2, kFc2TilesPerDomain = kFc2TilesPerExpert / 2;
constexpr int kMaxItems = 96;
constexpr int kActWords = kInter / 4;  // 64-bit words per activation row (4 bf16)
constexpr int kFc1Arrivals = kFc1TilesPerExpert * 4;
constexpr int kImgBytes = (kHidden / 64) * 512;  // token-scale images of one pass (64 K chunks)
constexpr int kThreads = 384;
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kAccCol = 0;     // 2 accumulators x 128 columns
constexpr uint32_t kSfaCol = 256;   // weight scales: kStages regions of 32 columns
#ifdef MUSE_VR_LDW_STAGES
constexpr uint32_t kSfbCol = kSfaCol + kStages * 32;  // token scales: kStages regions of 32 columns
static_assert(kSfbCol + kStages * 32 <= kTmemCols, "tensor memory columns");
#else
constexpr uint32_t kSfbCol = 352;   // token scales: kStages regions of 32 columns
#endif
constexpr int kErrPoll = 1, kErrMbar = 2, kErrSlot = 4, kErrStart = 8, kErrDrain = 16, kErrShape = 32;
#if MUSE_VR_LDW_LANDED
constexpr int kErrLand = 64;  // landed build: an expired landing poll, a word of a newer transfer, or a missed advance
#if MUSE_VR_LDW_BALB
constexpr int kErrBoard = 128;  // board build: the board read expired, its digest or a holder code is wrong
constexpr uint32_t kNoHolder = 0xFu;
#endif
// token row in the landing: FP4 bytes, block-scale bytes, the per-token scale; 6 payload bytes per 64-bit word
constexpr int kTokRowBytes = kHidden / 2 + kHidden / 16 + 4, kTokWords = (kTokRowBytes + 5) / 6;
constexpr int kTokSfDword = kHidden / 8, kTokPtsDword = kTokSfDword + kHidden / 64;  // 32-bit word offsets in the row
#endif
// slot claim area (slot_claim.cuh): 32-bit words kClaimWord .. kClaimWord + muse_vr_slot::kWords - 1 of the control
// tensor (word 0: start tickets)
constexpr int kClaimWord = 21;
static_assert(kClaimWord + muse_vr_slot::kWords <= 32, "claim area inside the 32-word control tensor");
constexpr unsigned kStartCycle = 1u << 12;
constexpr int kNumBuckets = 4;  // token-operand heights 16 / 32 / 64 / 128
#if MUSE_VR_LDW_PASS_ROWS == 48
__host__ __device__ constexpr int bucket_of(int rows) { return rows <= 16 ? 0 : rows <= 32 ? 1 : rows <= 48 ? 2 : 3; }
__host__ __device__ constexpr int bucket_rows(int b) { return b == 2 ? 48 : 16 << b; }
#else
__host__ __device__ constexpr int bucket_of(int rows) { return rows <= 16 ? 0 : rows <= 32 ? 1 : rows <= 64 ? 2 : 3; }
__host__ __device__ constexpr int bucket_rows(int b) { return 16 << b; }
#endif
// the kHs2Rows bound: a pass of r <= kPassRows rows is padded by at most max(15, r) rows
__host__ __device__ constexpr bool hs2_pad_bound() {
  for (int r = 1; r <= kPassRows; ++r)
    if (bucket_rows(bucket_of(r)) - r > (r > 15 ? r : 15)) return false;
  return true;
}
static_assert(hs2_pad_bound(), "hs2 padding bound (kHs2Rows)");
#if MUSE_VR_LDW_EARLY_COPY
// early copy: barrier ids and thread counts. Every named barrier of an early-copy build (the forms refused above use none
// beyond these): 3 (end of prologue), 5 (end), 6 (prologue steps), 7 (copy phase), the list barrier, the hand-off barrier and
// the copy warps' barrier; all distinct and below the 16 logical barriers of a CTA (PTX ISA barrier{.cta}: ids 0..15). The two
// new ids also avoid 0 (the CTA-wide barrier), the head-loader and hop-push barriers of the refused forms, and 11 / 12 (kept
// free for other development forms of this kernel).
constexpr int kEcgBar = MUSE_VR_LDW_ECG_BAR, kEcBar = MUSE_VR_LDW_EC_BAR;
constexpr int kEcgThreads = 32 + 8 * 32;  // warp 0 arrives, warps 4-11 wait
constexpr int kEcThreads = 8 * 32;        // warps 4-11
constexpr int kNamedBarIds = 16;
__host__ __device__ constexpr bool ec_bar_ids_ok() {
  const int live[] = {3, 5, 6, 7, MUSE_VR_LDW_LIST_BAR, kEcgBar, kEcBar};
  const int n = int(sizeof(live) / sizeof(live[0]));
  for (int i = 0; i < n; ++i) {
    if (live[i] < 0 || live[i] >= kNamedBarIds) return false;
    for (int j = i + 1; j < n; ++j)
      if (live[i] == live[j]) return false;
  }
  const int kept[] = {0, MUSE_VR_LDW_HEAD_BAR, MUSE_VR_LDW_HB0_BAR, 11, 12};
  for (int k = 0; k < int(sizeof(kept) / sizeof(kept[0])); ++k)
    if (kEcgBar == kept[k] || kEcBar == kept[k]) return false;
  return true;
}
static_assert(ec_bar_ids_ok(), "early copy: barrier ids must be distinct among the live named barriers and below 16");
static_assert(kThreads == 12 * 32, "early copy: warps 0-3 (work list, producer, MMA, control) and 4-11 (copy)");
static_assert(kEcgThreads == 288 && kEcThreads == 256, "early copy: hand-off = 32 arrivals + 256 waiters; copy barrier = 256");
static_assert(kEcgThreads % 32 == 0 && kEcThreads % 32 == 0, "named barrier thread counts are multiples of the warp size");
#if MUSE_VR_LDW_TEST_EC_COUNT
constexpr int kErrEcCount = 512;  // tests only: a participant count of an early-copy barrier differs from its declared count
#endif
#endif


// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// CTA abort flag (shared memory; set on the first expired wait of any thread of the CTA)
__shared__ int g_abort;
__device__ __forceinline__ bool aborted() { return *reinterpret_cast<volatile int*>(&g_abort) != 0; }
__device__ __noinline__ void fail(int32_t* err, int code) {
  atomicOr(err, code);
  *reinterpret_cast<volatile int*>(&g_abort) = 1;
}
__device__ __forceinline__ bool elect_one() {
  uint32_t pred;
  asm volatile("{ .reg .pred P; elect.sync _|P, 0xffffffff; selp.u32 %0, 1, 0, P; }" : "=r"(pred));
  return pred != 0;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return done != 0;
}
// Non-blocking test of a phase (warp-uniform: lane 0's observation)
__device__ __forceinline__ bool mbar_test_warp(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return __shfl_sync(0xffffffffu, done, 0) != 0;
}
// Bounded wait (every lane of the calling warp calls it with the same barrier and parity). Returns false when the CTA
// aborted or the bound ran out (which aborts the CTA).
__device__ __forceinline__ bool mbar_wait(uint64_t* bar, uint32_t parity, int32_t* err, uint32_t sleep_ns = 0) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    ++polls;
    if ((polls & 63u) == 0u && aborted()) return false;
    if (polls > kMaxPolls) {
      fail(err, kErrMbar);
      return false;
    }
    if (sleep_ns > 0) __nanosleep(sleep_ns);
  }
  return true;
}
// Waits off the weight-stream critical path (epilogue, builders, control): one inline test, then an out-of-line polling
// loop shared by every call site (hot-code budget of those roles).
__device__ __noinline__ bool mbar_wait_slow(uint64_t* bar, uint32_t parity, int32_t* err) {
  return mbar_wait(bar, parity, err, MUSE_VR_LDW_IDLE_NS);
}
__device__ __forceinline__ bool mbar_wait_idle(uint64_t* bar, uint32_t parity, int32_t* err) {
  return mbar_try(bar, parity) || mbar_wait_slow(bar, parity, err);
}
// Bounded wait for asynchronous work this CTA issued (completes on its own): not cut short by the abort flag.
__device__ __noinline__ void mbar_drain(uint64_t* bar, uint32_t parity, int32_t* err) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    if (++polls > kMaxPolls) {
      atomicOr(err, kErrDrain);
      return;
    }
  }
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t n) {
  // non-aligned form: correct even if the calling warp is not converged at this point (PTX ISA barrier.sync)
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(uint32_t id, uint32_t n) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void grid_dep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
#if MUSE_VR_LDW_TEST_EC_TWIN || MUSE_VR_LDW_TEST_EC_PLANT
__device__ __forceinline__ void ec_test_wait() {  // tests only (early-copy controls): about that many clock cycles
  const long long c0 = clock64();
  while (clock64() - c0 < (long long)(MUSE_VR_LDW_TEST_EC_WAIT_CYCLES)) __nanosleep(1000);
}
#endif
#if MUSE_VR_LDW_TEST_EC_TWIN == 1
// tests only (hand-off dropped): a value taken from warp 0's publication, clamped into [0, hi]
__device__ __forceinline__ int ec_twin_clamp(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }
#endif
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
// asynchronous copies global -> shared of 16 / 4 bytes (generic proxy); completion tracked by cp_async_arrive
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
// cp.async groups: one per head stage; before a ring slot is written again, the issuing thread waits for the group that
// last wrote it (complete by then: its stage was consumed, which the empty barrier already showed), so the reuse is
// ordered after the copy by this thread too
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_newer(int n) {  // at most n newer groups pending (n = 0, 1, 2)
  if (n <= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if (n == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 2;" ::: "memory");
}
// arrive-on of bar once every prior cp.async of this thread has completed; the pending count is raised by one first, so
// the phase cannot complete before those copies (PTX ISA cp.async.mbarrier.arrive)
__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar))
      : "memory");
}
#if MUSE_VR_LDW_THEAD
// four rows r0..r3 of a 2D map, columns x .. x + box - 1 each, as one destination tile (PTX ISA tensor copy, tile::gather4)
__device__ __forceinline__ void tma_gather4(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t r0, int32_t r1, int32_t r2,
                                            int32_t r3, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], "
      "[%7];" ::"r"(dst),
      "l"(map), "r"(x), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(smem_u32(bar))
      : "memory");
}
#endif
__device__ __forceinline__ void tma_load_3d_hint(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z,
                                                 uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void l2_prefetch_3d(const CUtensorMap* map, int32_t x, int32_t y, int32_t z) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(map), "r"(x), "r"(y), "r"(z)
               : "memory");
}
__device__ __forceinline__ void l2_prefetch(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void bulk_load_hint(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const void* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

// tcgen05 (PTX ISA, tensor core 5th generation)
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_cp_32x128b_x4(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
__device__ __forceinline__ void tc_mma_nvf4(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa,
                                            uint32_t sfb, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
      "  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; }" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate), "r"(sfa), "r"(sfb)
      : "memory");
}
// Shared-memory matrix descriptor, K-major, 128-byte swizzle, 8-row groups of 1024 bytes (SBO), version 1.
__device__ __forceinline__ uint64_t sdesc_sw128(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(1024u >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}
// Scale-factor source descriptor for tcgen05.cp: no swizzle, 8 x 16-byte rows per core matrix (SBO 128 bytes).
__device__ __forceinline__ uint64_t sdesc_sf(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(128u >> 4) << 32) | (uint64_t(1) << 46);
}
// Instruction descriptor, kind::mxf4nvf4: dense, E2M1 operands, K-major, UE4M3 scales, scale ids 0, M = 128, N = n.
__host__ __device__ constexpr uint32_t idesc_nvf4(uint32_t n) {
  return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | ((128u >> 4) << 24);
}
__device__ __forceinline__ void sts_u64(uint32_t addr, uint64_t v) {
  asm volatile("st.shared.b64 [%0], %1;" ::"r"(addr), "l"(v) : "memory");
}
#if MUSE_VR_LDW_LANDED
__device__ __forceinline__ void sts_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
#endif
__device__ __forceinline__ void sts_u16(uint32_t addr, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"(v) : "memory");
}
__device__ __forceinline__ void sts_v4(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void sts_v2(uint32_t addr, uint2 v) {
  asm volatile("st.shared.v2.b32 [%0], {%1,%2};" ::"r"(addr), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ int lds_s16(uint32_t addr) {  // sign-extended to 32 bits
  int v;
  asm volatile("ld.shared.s16 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ int lds_s32(uint32_t addr) {
  int v;
  asm volatile("ld.shared.s32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint4 lds_v4(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_cta_shared(int32_t* p, int v) {
  asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(p)), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_cta_shared(const int32_t* p) {
  int v;
  asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t shfl_idx(uint32_t v, int src) {
  uint32_t r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(r) : "r"(v), "r"(src));
  return r;
}
__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ ulonglong2 ld_relaxed_v2u64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_relaxed_s32(const int32_t* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ ulonglong2 ld_ca_v2u64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.global.ca.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t tmem_ld_x1(uint32_t taddr) {
  uint32_t r;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n\ttcgen05.wait::ld.sync.aligned;" : "=r"(r) : "r"(taddr) : "memory");
  return r;
}
__device__ __forceinline__ unsigned ld_acquire_gpu_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// fp32 arithmetic with flush-to-zero (reproduces the stock results bitwise).
__device__ __forceinline__ float fmul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fadd_ftz(float a, float b) {
  float r;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ffma_ftz(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float ex2_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float rcp_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t bf16_satfinite(float x) {
  uint32_t r;
  asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(0.0f), "f"(x));
  return r & 0xFFFFu;
}

// FC1 gated epilogue for one (gate, up) accumulator pair, with the stock operand clamp at its default limit.
__device__ __forceinline__ uint32_t swiglu_out(float acc_gate, float acc_up, float s_tok, float a_gate, float a_up) {
  constexpr float kLimit = 3.40282346638528859812e+38f;
  float x = fmul_ftz(acc_gate, s_tok);
  float u = fmul_ftz(acc_up, s_tok);
  const float x0 = x, u0 = u;
  x = x > kLimit ? kLimit : x;
  x = (x0 >= -kLimit || x0 != x0) ? x : -kLimit;
  u = u > kLimit ? kLimit : u;
  u = (u0 >= -kLimit || u0 != u0) ? u : -kLimit;
  const float k = fmul_ftz(a_gate, 1.4426950216293334961f);
  const float e = ex2_ftz(fmul_ftz(fmul_ftz(x, k), -1.0f));
  const float r = rcp_ftz(fadd_ftz(e, 1.0f));
  const float up = ffma_ftz(u, a_gate, 0.0f);
  return bf16_satfinite(fmul_ftz(fmul_ftz(up, fmul_ftz(x, r)), a_up));
}

// Byte offset of (row r, 16-byte chunk c) inside one 128-byte-swizzled K atom of rows.
__device__ __forceinline__ uint32_t sw128_off(int r, int c) {
  return uint32_t((r >> 3) * 1024 + (r & 7) * 128 + ((c ^ (r & 7)) << 4));
}


#if MUSE_VR_LDW_MAX_TOKENS > 512
constexpr int kMaxMySlots = 64;                  // routed slots whose token row one CTA copies (slots b, b + G, ...)
#else
constexpr int kMaxMySlots = 32;                  // routed slots whose token row one CTA copies (slots b, b + G, ...)
#endif
constexpr int kCntSet = 2 * kMaxPasses + 32;     // per launch parity: FC1 arrivals [kMaxPasses], FC2-operand rows
#if MUSE_VR_LDW_EPI
constexpr int kMomParts = (kHidden / kTile) * 4;  // per routed row: one partial per (FC2 tile, epilogue warp)
constexpr int kMaxOwn = 32;                       // tokens one CTA finalizes (t = blockIdx.x mod grid)
#endif
constexpr int kRowsReady = 2 * kMaxPasses;       // [kMaxPasses], rows-ready count of the grid (one per CTA)
// the count the readers of the copy phase's per-slot metadata acquire (SPLIT_REL: the second release)
constexpr int kMetaReady = kRowsReady + (MUSE_VR_LDW_SPLIT_REL ? 1 : 0);
constexpr int kChunkSlots = 256, kNumChunks = kMaxSlots / kChunkSlots;

struct Params {
  const int32_t* packed;     // [M, 8] (expert << 16 | bf16 weight)
  const uint8_t* hs;         // [M, 2048] FP4 token rows
  const uint8_t* hs_sf;      // [M, 256] e4m3, linear
  const float* pts;          // [M]
  const uint8_t* sf13;       // [P, 8192, 256]
  const uint8_t* sf2;        // [P, 4096, 256]
  const float* gate_alpha;
  const float* up_alpha;
  const float* out_alpha;
  const int8_t* sm_domain;
  uint8_t* hs_perm;          // [kMaxSlots, 2048] FC1 token rows, expert-major
  float* pts_perm;           // [kMaxSlots]
  uint8_t* img1;             // [kMaxPasses][64][512] FC1 token-scale images
  uint8_t* img2;             // [kMaxPasses][64][512] FC2 token-scale images
  unsigned long long* act;   // [kMaxSlots][kActWords]
  uint8_t* hs2;              // [kHs2Rows, 2048] FC2 token rows at bucket-padded pass bases
  float* ramax;              // [kMaxSlots] FC2-input row maxima
  unsigned* cnt;             // [2][kCntSet] counters, one set per launch parity
  __nv_bfloat16* out;        // [M * 8, 4096]
  int32_t* e2p;              // [M * 8]
  uint16_t* expert_weights;  // [M * 8]
  unsigned* ticket;
  int32_t* err;
  int32_t m, pool, local_offset;
  int32_t n_dom0, n_dom1;
  float gsi;
  int32_t flags;
  int32_t dbg;               // ablation (timing studies only): 1 no MMAs, 128 no epilogue / builder roles, 256 no token
                             // copies, 512 no work items
  unsigned long long* stamps;  // [grid][kStampWords] per-CTA timestamps (timing builds only), or nullptr
#if MUSE_VR_LDW_ROWS_READY
  unsigned long long* rr;      // [kRrWords]: [0] epoch ticket, [1] published epoch, [2, 2 + kRrLast) the consumer's words (never
                               // touched here), [kRrFlagBase, + kRrFlags) flags [pass][tile][warp], then e2pass (int16
                               // [kMaxTokens * kTopK]: pass index of each routed slot's row, or -1)
#endif
#if MUSE_VR_LDW_LANDED
  // Landing buffers of this rank: copy 0 of each site's receive region, laid out [copy][source rank][rows_max][pitch].
  const unsigned long long* ids_land;  // routed ids, 8 tagged words per row (pitch >= 8)
  const unsigned long long* tok_land;  // token rows, kTokWords tagged words per row (pitch >= kTokWords, even)
  unsigned long long ids_copy_words, tok_copy_words;  // 64-bit words per copy
  const unsigned long long* seq64;     // forward sequence word (device memory; never a per-forward argument)
  unsigned long long* last_seq;        // missed-advance guard: one word per CTA [grid]
  uint32_t ids_pitch, tok_pitch, rows_max, world;
  uint32_t ids_parity, tok_parity;     // 1: two copies selected by the sequence word's parity
#endif
#if MUSE_VR_LDW_EPI
  const float* norm_w;                 // [4096] post-expert norm weight (1.0 pre-added), weight-first form
  float norm_eps;
  float* moments;                      // [kMaxSlots][kMomParts] per permuted row: partial sums of (w y)^2
  int32_t* pos_slot;                   // [kMaxSlots] permuted row -> routed slot (copy phase)
  unsigned* tok_cnt;                   // [2][kMaxTokens] per-token completion counters, one set per launch parity
  __nv_bfloat16* tok_out;              // [M, 4096] per-token weighted sum of the local routes
#endif
#if MUSE_VR_LDW_HB0_PUSH
  muse_vr::lb_hop::Site<MUSE_VR_LDW_HB0_PUSH> hb;  // the expert-output hop site the routed rows are stored into
  muse_vr::lb_hop::Ctl hbc;                        // its wait bounds and sticky error word (credit wait)
  int32_t* hb_slot;                                // [kMaxSlots] permuted row -> routed slot t * 8 + k (copy phase)
#endif
#if MUSE_VR_LDW_BALB
  const uint32_t* bal_table;           // [256] placement words: bits 0-3 home rank, 4-7 second holder (0xF none), 8-15 home
                                       // slot, 16-23 second-holder slot of the pool
  const unsigned long long* board;     // [33] this step's board (same GPU): word l = tag << 32 | codes of experts 8l..8l+7
                                       // (4 bits each, expert 8l in bits 3..0; 0xF = not routed); word 32 = tag << 32 | digest
  int32_t bal_rank;                    // this expert-parallel rank
#if MUSE_VR_LDW_TEST_BOARD_DELAY_NS > 0
  uint32_t test_delay_ns;              // tests only: the emulated board arrival after the dependency wait
#endif
#endif
};
constexpr int kFlagEvictFirst = 2, kFlagTailTrigger = 32;
constexpr int kStampWords = 32;
// Timing builds (-DMUSE_VR_LDW_STAMPS) include a stamp header that is not part of this package; production builds compile
// the stamps out.
#ifdef MUSE_VR_LDW_STAMPS
#include "ldw_stamps.cuh"
#else
#define LDW_STAMP(p, slot) ((void)0)
#define LDW_STAMPV(p, slot, v) ((void)0)
#endif
constexpr int kStStart = 0, kStPro = 1, kStCopy = 2, kStRows = 3, kStFirstFull = 4, kStFc1Issued = 5, kStFc2Dep = 6,
              kStAllIssued = 7, kStMma = 8, kStEpi = 9, kStBuild = 10, kStEnd = 11, kStEpiFc1 = 12, kStCopyRows = 13,
              kStCopyImg = 14, kStCopyBar = 15, kStItems = 16, kStCost = 17, kStSm = 18, kStCopyIn = 21, kStCopyLd = 22;
constexpr int kStWait = 20;  // timing builds: the dependency wait returned (thread 0)
// timing builds, prologue phases (thread 0 unless noted): ids accepted (own), counts, prefix, passes + slot, work list
// (warp 0), head list published (warp 3), stage 0's weights issued early (HEAD_GO), the head loaders' batch issued (warp 4)
constexpr int kStIdsOwn = 24, kStCnt = 25, kStPfx = 26, kStPass = 27, kStList = 28, kStHeadGo = 29, kStW0 = 30,
              kStHeadIss = 31;
#if MUSE_VR_LDW_BALB
constexpr int kStBoard = 23, kStPinEnd = 19;  // board timing builds: board part published (warp 3); producer at the pinned end
#endif
// Tests only (board build with MUSE_VR_LDW_TEST_BOARD_DELAY_NS > 0): a header that is not part of this package records the
// dependency wait's return and makes the board phase start that many ns later; otherwise both hooks compile to nothing.
#if MUSE_VR_LDW_BALB && MUSE_VR_LDW_TEST_BOARD_DELAY_NS > 0
#include "ldw_test_delay.cuh"
#else
#define LDW_TEST_T0(s) ((void)0)
#define LDW_TEST_BOARD_WAIT(s, p) ((void)0)
#endif

// ---------------------------------------------------------------- shared memory
struct Smem {
  alignas(1024) uint8_t st[kStages][kStageBytes];  // [weights 32K | weight scales 4K | token rows 32K | token-scale images 4K]
  uint64_t full[kStages], empty[kStages], sf_full[kStages];
  uint64_t tmem_full[2], tmem_empty[2], drain_sf, drain_mma;
  alignas(16) uint16_t ostage[4][8][32];  // FC2 epilogue: 8 rows x 32 output columns per epilogue warp
  int16_t q_expert[kMaxPasses], q_row[kMaxPasses], q_rows[kMaxPasses];
  int16_t q_row2[kMaxPasses];  // bucket-padded first row of each pass in hs2 (see kHs2Rows)
  int16_t base[kMaxPool], qbase[kMaxPool];  // first permuted row / first pass of each local expert
  int16_t my_slot[kMaxMySlots], my_pos[kMaxMySlots], my_exp[kMaxMySlots];
  int16_t ptok[kHeadItems][kHeadRows];  // head copies: token of each row of the pass of FC1 item m < head_n
#if MUSE_VR_LDW_LANDED
  uint64_t tok_full[kStages];            // head stages: the 4 loader warps' token operand in the slot (count 4)
  float hpts[4][kHeadRows];              // per-token scales of the head item's rows, one copy per loader warp (each
                                         // loader warp reads its own copy in its FC1 epilogue)
  unsigned long long land_seq;           // this forward's sequence word, read once per CTA
  uint32_t ids_tag, tok_tag, ids_copy, tok_copy;
  int32_t land_go;                       // 0: missed advance (nothing consumed, error word set)
#if MUSE_VR_LDW_HEAD_GO
  int32_t head_go;                       // 1: warp 3 published the head-row list (release / acquire, CTA scope)
  int32_t head_q;                        // the head item's pass (published with head_go)
#endif
#if MUSE_VR_LDW_HEAD_PRIO == 1
  uint64_t head_iss;                     // the 4 loader warps issued their first stage's token batch (count 4)
#endif
#endif
#if MUSE_VR_LDW_EPI
  int16_t own_t[kMaxOwn];                // tokens this CTA finalizes
  int8_t own_r[kMaxOwn];                 // their local routes (0: a zero row)
  int32_t own_n;
#endif
#if MUSE_VR_LDW_BALB
  uint32_t btab[256];                    // placement words (static; staged before the dependency wait)
  uint8_t bcode[256];                    // the board's holder codes (warp 3, before bal_ready)
  int16_t slot_exp[kMaxPool];            // expert held at each pool slot of this rank (-1: none)
  int16_t cand_n[kMaxPool];              // routed rows of each candidate pool slot (0 otherwise)
  int8_t cand[kMaxPool];                 // 1: the slot holds a replicated expert (local only if the board codes this rank)
  int16_t my_rank[kMaxMySlots];          // per slot of this CTA: its rank among its pool slot's routed slots
  int8_t my_cand[kMaxMySlots];           // 1: a candidate slot (position set by the board phase)
  int32_t bal_nq_p, bal_rows_p, bal_n1_p;      // pinned part: passes, rows, FC1 items of this CTA
  int32_t bal_n_items, bal_n_fc1, bal_n_pass;  // final counts (warp 3, before bal_ready)
  uint32_t bal_ready;                    // 1: the board part is published (release / acquire, CTA scope)
#if MUSE_VR_LDW_TEST_BOARD_DELAY_NS > 0
  unsigned long long t_wait;             // tests only: the time the dependency wait returned (thread 0)
#endif
#endif
  int32_t head_n;                       // leading FC1 items whose token rows are copied by row index (0: none)
  uint32_t items[kMaxItems];
  int32_t n_items, n_fc1, n_pass, n_my, set;
#if MUSE_VR_LDW_TEST_EC_COUNT
  uint32_t ec_dbg[2];  // tests only: [0] hand-off barrier (arrivals << 16 | waiters), [1] copy warps' barrier (waiters)
#endif
  uint32_t fc1_stored[4];  // per FC1 epilogue warp: tiles whose activation words it stored (red.release / ld.acquire, cta)
#if MUSE_VR_LDW_HB0_PUSH
  uint32_t fc2_stored[4];  // per FC2 epilogue warp: FC2 items whose output rows it stored (red.release / ld.acquire, cta)
#endif
  uint32_t tmem_base;
  unsigned start_base;
  unsigned slot;  // this CTA's claimed slot (domain << 7 | rank), set in the prologue
#if MUSE_VR_LDW_ROWS_READY
  unsigned long long rr_epoch;  // this launch's rows-ready epoch (prologue)
#endif
};
static_assert(kStageBytes % 1024 == 0 && kStageA % 1024 == 0, "stage parts stay 1024-byte aligned");
#if MUSE_VR_LDW_LANDED
// the weight and weight-scale copies of a stage end where its token rows begin (the head loaders' generic stores never
// overlap the bytes the producer's weight copies write into the same stage)
static_assert(kStageW + kChunks * 512 == kStageA && kStageA + kStageB + kStageImg == kStageBytes, "stage regions");
#endif
// prologue scratch (the last ring stage, before its first copy): expert of each slot, per-chunk expert counts
struct Scratch {
  int16_t exp[kMaxSlots];
  uint32_t ccnt[kNumChunks][kMaxPool];
  uint32_t total[kMaxPool];
};
static_assert(sizeof(Scratch) <= kStageBytes, "prologue scratch fits one ring stage");
#if MUSE_VR_LDW_EARLY_COPY
// early copy: warps 4-11 arrive at barrier 3 (after which the producer's first copies may overwrite the scratch in the last ring
// stage) before they run the copy phase; the copy reads only these shared fields, all of them outside the ring stages
constexpr size_t kRingEnd = offsetof(Smem, st) + sizeof(Smem::st);
static_assert(offsetof(Smem, base) >= kRingEnd && offsetof(Smem, qbase) >= kRingEnd && offsetof(Smem, my_slot) >= kRingEnd &&
                  offsetof(Smem, my_pos) >= kRingEnd && offsetof(Smem, my_exp) >= kRingEnd && offsetof(Smem, n_my) >= kRingEnd &&
                  offsetof(Smem, set) >= kRingEnd,
              "early copy: the copy phase's shared operands are outside the ring stages (the prologue scratch)");
#endif
#if MUSE_VR_LDW_HEAD_GO
// the head loaders may store into the token region of the last ring stage while other warps still read the scratch in it
// (they no longer wait for the end-of-prologue barrier): the scratch must end before that region
static_assert(sizeof(Scratch) <= kStageA, "prologue scratch below the token region of its ring stage");
#endif

__device__ __forceinline__ uint32_t item_pack(int fc2, int q, int tile) {
  return uint32_t(q) | (uint32_t(tile) << 8) | (uint32_t(fc2) << 14);
}
struct Item {
  int fc2, q, tile;
};
__device__ __forceinline__ Item item_unpack(uint32_t v) {
  Item it;
  it.q = int(v & 0xFFu);
  it.tile = int((v >> 8) & 0x3Fu);
  it.fc2 = int((v >> 14) & 1u);
  return it;
}
__device__ __forceinline__ int steps_of(int units, int rank, int nr) {
  return units > rank ? (units - 1 - rank) / nr + 1 : 0;
}
// Bounded acquire poll of a counter (one lane); false when the bound ran out or the CTA aborted.
// Relaxed polls until the value is reached, then one acquire load (the acquire pattern of PTX ISA: the acquire reads
// the same location after the relaxed observations).
__device__ __forceinline__ bool acquire_at_least(const unsigned* c, unsigned want, int32_t* err) {
  uint32_t polls = 0;
  while (muse_vr_ldq::ld_relaxed_gpu(c) < want) {
    if (aborted()) return false;
    if (++polls > kMaxPolls) {
      fail(err, kErrPoll);
      return false;
    }
    __nanosleep(64);
  }
  while (ld_acquire_gpu_u32(c) < want) {  // counts only grow: the first acquire normally observes it
    if (aborted() || ++polls > kMaxPolls) {
      if (!aborted()) fail(err, kErrPoll);
      return false;
    }
  }
  return true;
}
#if MUSE_VR_LDW_LANDED
// Landed words. A word is accepted on its own 64-bit relaxed system-scope load (the producers' store width and scope)
// whose tag field equals this transfer's tag; the payload of that load is kept (a vector load is one access per 64-bit
// element, PTX ISA memory consistency model). A tag of one of the next three transfers means the slot was already
// rewritten: waiting cannot help, so the error word is set at once, as when the poll bound runs out.
__device__ __forceinline__ unsigned long long ld_relaxed_sys_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ ulonglong2 ld_relaxed_sys_v2u64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ ulonglong2 ld_head_v2u64(const unsigned long long* p) {
#if MUSE_VR_LDW_TEST_HEAD_LDGPU
  ulonglong2 v;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
#else
  return ld_relaxed_sys_v2u64(p);
#endif
}
__device__ __forceinline__ void mbar_arrive_cnt(uint64_t* bar, uint32_t n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(n) : "memory");
}
// tag fields: token words bits 63..48 (tags 1..65535), id words bits 63..32 (tags 1..2^32 - 1); 0 = an empty word
constexpr int kTokTagShift = 48, kIdsTagShift = 32;
__device__ __forceinline__ bool land_ok(unsigned long long v, uint32_t tag, int shift) { return uint32_t(v >> shift) == tag; }
__device__ __forceinline__ bool land_ahead(uint32_t t, uint32_t w, int shift) {
  if (t == 0u) return false;
  const unsigned long long n = shift == kTokTagShift ? 65535ull : 0xFFFFFFFFull;
  const unsigned long long a = t - 1u, b = w - 1u;
  const unsigned long long d = a >= b ? a - b : a + n - b;
  return d >= 1ull && d <= 3ull;
}
__device__ __noinline__ unsigned long long land_poll(const unsigned long long* w, uint32_t tag, int shift, int32_t* err) {
  uint32_t polls = 0;
  for (;;) {
    const unsigned long long v = ld_relaxed_sys_u64(w);
    const uint32_t t = uint32_t(v >> shift);
    if (t == tag) return v;
    if (land_ahead(t, tag, shift) || ++polls > kMaxPolls) {
      fail(err, kErrLand);
      return 0ull;
    }
    if (aborted()) return 0ull;
    __nanosleep(32);
  }
}
__device__ __forceinline__ unsigned long long land_accept(unsigned long long v, const unsigned long long* w, uint32_t tag,
                                                          int shift, int32_t* err) {
  return land_ok(v, tag, shift) ? v : land_poll(w, tag, shift, err);
}
// rows: rank r holds rows [floor(M r / W), floor(M (r + 1) / W)) and sends them from its block of the site (32-bit
// arithmetic: M <= 1024 rows, W <= 16 ranks; one source when W = 1)
__device__ __forceinline__ uint32_t land_own0(uint32_t m, uint32_t world, uint32_t r) { return (m * r) / world; }
__device__ __forceinline__ uint32_t land_owner(uint32_t m, uint32_t world, uint32_t row) {
  if (world == 1u) return 0u;
  uint32_t r = (row * world) / (m ? m : 1u);
  while (r + 1u < world && land_own0(m, world, r + 1u) <= row) ++r;
  while (r > 0u && land_own0(m, world, r) > row) --r;
  return r;
}
__device__ __forceinline__ const unsigned long long* tok_row(const Smem& s, const Params& p, int t) {
  return p.tok_land + s.tok_copy * p.tok_copy_words +
         (uint64_t(land_owner(uint32_t(p.m), p.world, uint32_t(t))) * p.rows_max + uint32_t(t)) * p.tok_pitch;
}
__device__ __forceinline__ const unsigned long long* ids_word(const Smem& s, const Params& p, int slot) {
  const uint32_t t = uint32_t(slot / kTopK);
  return p.ids_land + s.ids_copy * p.ids_copy_words +
         (uint64_t(land_owner(uint32_t(p.m), p.world, t)) * p.rows_max + t) * p.ids_pitch + uint32_t(slot % kTopK);
}
// 32-bit words 3 j .. 3 j + 2 of a token row from the payloads of its words 2 j and 2 j + 1
__device__ __forceinline__ void tok_dwords(unsigned long long w0, unsigned long long w1, uint32_t* d) {
  d[0] = uint32_t(w0);
  d[1] = (uint32_t(w0 >> 32) & 0xFFFFu) | (uint32_t(w1) << 16);
  d[2] = uint32_t(w1 >> 16);
}
// 32-bit word dw of a token row: one aligned word-pair load, accepting only the word(s) that carry it (the per-token
// scale is the first 4 bytes of the row's last word, whose pair word is pitch padding that no producer writes)
__device__ __forceinline__ uint32_t tok_dword(const unsigned long long* row, int dw, uint32_t tag, int32_t* err) {
  const int j = dw / 3, r = dw - 3 * j;
  const ulonglong2 v = ld_relaxed_sys_v2u64(row + 2 * j);
  const unsigned long long w0 = r < 2 ? land_accept(v.x, row + 2 * j, tag, kTokTagShift, err) : 0ull;
  const unsigned long long w1 = r > 0 ? land_accept(v.y, row + 2 * j + 1, tag, kTokTagShift, err) : 0ull;
  uint32_t d[3];
  tok_dwords(w0, w1, d);
  return d[r];
}
#endif
// Two 16-lane x 16-column loads (lane bases t0 and t1), then one wait. Fragment of .16x256b (PTX ISA tcgen05 matrix
// fragments): thread t holds lane t / 4 (registers 0, 1) and lane t / 4 + 8 (registers 2, 3), columns 2 (t % 4) and
// 2 (t % 4) + 1; .x2 repeats it for the next 8 columns (registers 4..7).
__device__ __forceinline__ void tmem_ld_16x256b_x2_pair(uint32_t t0, uint32_t t1, uint32_t (&a)[8], uint32_t (&b)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%16];\n\t"
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%8,%9,%10,%11,%12,%13,%14,%15}, [%17];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]), "=r"(a[4]), "=r"(a[5]), "=r"(a[6]), "=r"(a[7]), "=r"(b[0]),
        "=r"(b[1]), "=r"(b[2]), "=r"(b[3]), "=r"(b[4]), "=r"(b[5]), "=r"(b[6]), "=r"(b[7])
      : "r"(t0), "r"(t1)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x4(uint32_t taddr, uint32_t (&r)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];\n\ttcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(taddr)
               : "memory");
}
__device__ __forceinline__ void tmem_ld_x16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n\ttcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr)
               : "memory");
}

#if MUSE_VR_LDW_BALB
// Readers of the board part (any warp after barrier 3): bounded acquire of bal_ready by one lane, then the warp. Returns false
// (warp-uniform) if the wait ended without it (abort or poll bound; the sticky error word is set).
__device__ __forceinline__ bool bal_acquire(Smem& s, const Params& p) {
  uint32_t v = 0u;
  if ((threadIdx.x & 31u) == 0u) {
    uint32_t polls = 0u;
    while (true) {
      asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(smem_u32(&s.bal_ready)) : "memory");
      if (v != 0u || aborted()) break;
      if (++polls > kMaxPolls) {
        fail(p.err, kErrMbar);
        break;
      }
      __nanosleep(32);
    }
  }
  v = __shfl_sync(0xffffffffu, v, 0);
  __syncwarp();  // memory ordering among the warp's threads: their reads follow lane 0's acquire
  return v != 0u;
}
// The item count to iterate up to at item m: the pinned part's count until m reaches it, then (once) the final count
// (unchanged if the wait failed, so the loop ends).
__device__ __forceinline__ int bal_count(Smem& s, const Params& p, int m, int& n, bool& fin) {
  if (m >= n && !fin) {
    if (bal_acquire(s, p)) n = s.bal_n_items;
    fin = true;
  }
  return n;
}
#endif

// ---------------------------------------------------------------- roles
// warp 0: weight, weight-scale, token-row and token-scale copies, one stage at a time
#if MUSE_VR_LDW_HEAD_GO
// warp 0 right after its work list (before the end-of-prologue barrier): the weights of the CTA's first ring stage (stage 0
// of item 0), so they do not wait for the other roles' prologue work. Ring slot 0 holds no prologue scratch (the last slot
// does); the proxy fence orders this warp's earlier generic shared accesses before the asynchronous copy all the same.
// Returns 1 if stage 0's weights were issued (role_producer then issues stages 1..).
__device__ __noinline__ int producer_first(Smem& s, const Params& p, const CUtensorMap* m13) {
  __syncwarp();  // the work list's stores by every lane of this warp before this read of it
  const bool leader = elect_one();
  const int lane = int(threadIdx.x & 31u), lead = __ffs(int(__ballot_sync(0xffffffffu, leader))) - 1;
  int done = 0;
  if ((p.dbg & 256) == 0 && s.n_items > 0 && leader && !aborted()) {
    const Item it = item_unpack(s.items[0]);
    if (!it.fc2) {  // an FC2 first item keeps the ordinary order
      fence_proxy_async_smem();
      const int e = s.q_expert[it.q];
      const int tile_row = e * kFc1TilesPerExpert + it.tile;
      const uint8_t* sf = p.sf13 + size_t(tile_row) * (kHidden / 64) * 512;
      const uint32_t dst = smem_u32(&s.st[0][0]);
      mbar_expect_tx(&s.full[0], uint32_t(kStageA));
      if ((p.flags & kFlagEvictFirst) != 0) {
        const uint64_t pol = policy_evict_first();
        tma_load_3d_hint(dst, m13, 0, tile_row * kTile, 0, &s.full[0], pol);
        bulk_load_hint(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[0], pol);
      } else {
        tma_load_3d(dst, m13, 0, tile_row * kTile, 0, &s.full[0]);
        bulk_load(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[0]);
      }
      done = 1;
      LDW_STAMP(p, kStW0);
    }
  }
  (void)lane;
  return __shfl_sync(0xffffffffu, done, lead);
}
#endif

// pre0: stages whose weights warp 0 already issued (producer_first; 0 otherwise)
__device__ MUSE_VR_LDW_ROLE void role_producer(Smem& s, const Params& p, const CUtensorMap* m13, const CUtensorMap* m2,
                                               const CUtensorMap* b1, const CUtensorMap* b2, int pre0,
                                               const CUtensorMap* hm = nullptr) {
  const bool leader = elect_one();
  const int lane = int(threadIdx.x & 31u), lead = __ffs(int(__ballot_sync(0xffffffffu, leader))) - 1;
#if MUSE_VR_LDW_BALB
  int n_items = s.n_items;  // the pinned part's count; the final one once the board part is published
  bool bal_fin = false, bal_trig = false, rows_ready_r = false;
#else
  const int n_items = s.n_items;
#endif
  const bool ef = (p.flags & kFlagEvictFirst) != 0;
  const bool tok = (p.dbg & 256) == 0;
  const uint64_t pol = policy_evict_first();
  unsigned* const cset = p.cnt + s.set * kCntSet;
#if !MUSE_VR_LDW_BALB
  if (!MUSE_VR_LDW_EARLY_SLOTTAB && n_items == 0 && (p.flags & kFlagTailTrigger))
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  int k = 0, issued = 0;
  bool rows_ready = false;
  const int head_n = MUSE_VR_LDW_HEADS ? s.head_n : 0;
#if MUSE_VR_LDW_LANDED
  bool head_waited = false;  // the head loaders' stores into the ring are ordered before this warp's later copies
#endif
  // the last ring stage held the prologue scratch (generic accesses, ordered before this point by the CTA barrier): a
  // proxy fence before any asynchronous copy into the ring (PTX ISA memory consistency model, proxies)
  fence_proxy_async_smem();
  // the weights of the first kStages stages go out at once (free slots); their token rows wait for the rows-ready count
  int pre = 0;
#if MUSE_VR_LDW_HEAD_PRIO || MUSE_VR_LDW_HEAD_GO
  if (tok && leader && !aborted()) {
    pre = min(kStages, n_items * kStagesPerTile);
    // HEAD_PRIO: stage 0's weights, then (head CTAs) the loaders' first token batch, then stages 1..
    const int split = (MUSE_VR_LDW_HEAD_PRIO && head_n > 0) ? 1 : pre;
#pragma unroll 1
    for (int x = pre0; x < pre; ++x) {
      if (x == split) {
#if MUSE_VR_LDW_HEAD_PRIO == 1
        mbar_wait(&s.head_iss, 0u, p.err);  // the 4 loader warps issued their batch (bounded)
#elif MUSE_VR_LDW_HEAD_PRIO == 2
        mbar_wait(&s.tok_full[0], 0u, p.err);  // their batch landed in stage 0 (bounded; phase 0 cannot be overtaken: phase 1
                                               // needs stage 0 consumed, which needs this warp's arrival below)
#endif
        if (aborted()) break;
      }
      const Item it = item_unpack(s.items[x / kStagesPerTile]);
      const int e = s.q_expert[it.q], j = x % kStagesPerTile;
      const int tile_row = it.fc2 ? e * kFc2TilesPerExpert + it.tile : e * kFc1TilesPerExpert + it.tile;
      const uint8_t* sf = (it.fc2 ? p.sf2 : p.sf13) + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512;
      const uint32_t dst = smem_u32(&s.st[x][0]);
      mbar_expect_tx(&s.full[x], uint32_t(kStageA));
      if (ef) {
        tma_load_3d_hint(dst, it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4), &s.full[x], pol);
        bulk_load_hint(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[x], pol);
      } else {
        tma_load_3d(dst, it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4), &s.full[x]);
        bulk_load(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[x]);
      }
    }
    LDW_STAMP(p, kStFc1Issued);  // timing builds: the first stages' weight copies are out
  }
  if (false) {
#else
  (void)pre0;
  if (tok && leader && !aborted()) {
    pre = min(kStages, n_items * kStagesPerTile);
#pragma unroll 1
    for (int x = 0; x < pre; ++x) {
      const Item it = item_unpack(s.items[x / kStagesPerTile]);
      const int e = s.q_expert[it.q], j = x % kStagesPerTile;
      const int tile_row = it.fc2 ? e * kFc2TilesPerExpert + it.tile : e * kFc1TilesPerExpert + it.tile;
      const uint8_t* sf = (it.fc2 ? p.sf2 : p.sf13) + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512;
      const uint32_t dst = smem_u32(&s.st[x][0]);
      mbar_expect_tx(&s.full[x], uint32_t(kStageA));
      if (ef) {
        tma_load_3d_hint(dst, it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4), &s.full[x], pol);
        bulk_load_hint(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[x], pol);
      } else {
        tma_load_3d(dst, it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4), &s.full[x]);
        bulk_load(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[x]);
      }
    }
#if MUSE_VR_LDW_LANDED
    LDW_STAMP(p, kStFc1Issued);  // timing builds: the first stages' weight copies are out
#endif
    if (MUSE_VR_LDW_HEAD_PF > 0) {
      const int end = min(pre + MUSE_VR_LDW_HEAD_PF, n_items * kStagesPerTile);
#pragma unroll 1
      for (int x = pre; x < end; ++x) {
        const Item it = item_unpack(s.items[x / kStagesPerTile]);
        const int e = s.q_expert[it.q], j = x % kStagesPerTile;
        const int tile_row = it.fc2 ? e * kFc2TilesPerExpert + it.tile : e * kFc1TilesPerExpert + it.tile;
        l2_prefetch_3d(it.fc2 ? m2 : m13, 0, tile_row * kTile, j * (kChunks / 4));
        l2_prefetch((it.fc2 ? p.sf2 : p.sf13) + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512,
                    uint32_t(kChunks * 512));
      }
    }
#endif
  }
#pragma unroll 1
#if MUSE_VR_LDW_BALB
  for (int m = 0; m < bal_count(s, p, m, n_items, bal_fin); ++m) {
    if (!bal_fin && m + 1 == n_items && leader) LDW_STAMP(p, kStPinEnd);  // timing builds: the last pinned item starts
    if (!bal_fin) {  // the final count as soon as it is out (no wait), so the tail trigger can fire at the last item
      uint32_t v = 0u;
      if (lane == 0) asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(smem_u32(&s.bal_ready)) : "memory");
      v = __shfl_sync(0xffffffffu, v, 0);
      __syncwarp();  // memory ordering among the warp's threads: reads below follow lane 0's acquire
      if (v != 0u) {
        n_items = s.bal_n_items;
        bal_fin = true;
      }
    }
#else
  for (int m = 0; m < n_items; ++m) {
#endif
    const Item it = item_unpack(s.items[m]);
    const int e = s.q_expert[it.q], rows = s.q_rows[it.q], qrow = s.q_row[it.q];
    const int bk = bucket_of(rows), nb = bucket_rows(bk);
#if MUSE_VR_LDW_BALB
    if (bal_fin && m + 1 == n_items && (p.flags & kFlagTailTrigger)) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
      bal_trig = true;
    }
#else
    if (!MUSE_VR_LDW_EARLY_SLOTTAB && m + 1 == n_items && (p.flags & kFlagTailTrigger))
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    bool fc2_ready = !it.fc2 || (p.dbg & 128) != 0;
    const int tile_row = it.fc2 ? e * kFc2TilesPerExpert + it.tile : e * kFc1TilesPerExpert + it.tile;
    const uint8_t* sfw = it.fc2 ? p.sf2 : p.sf13;
    const CUtensorMap* wmap = it.fc2 ? m2 : m13;
    const CUtensorMap* bmap = (it.fc2 ? b2 : b1) + bk;
    const uint8_t* img = (it.fc2 ? p.img2 : p.img1) + size_t(it.q) * kImgBytes;
    const bool gat = m < head_n;  // head FC1 item: rows + scale image by row-index copies (landed: by the head loaders)
    // THEAD: a head stage's token bytes are the gathers' (rows rounded up to four); its image comes from warp 3
    const uint32_t tok_tx = gat ? (MUSE_VR_LDW_THEAD ? uint32_t(((rows + 3) / 4) * 4 * 256) : 0u) : uint32_t(nb * 256 + kStageImg);
#pragma unroll 1
    for (int j = 0; j < kStagesPerTile; ++j, ++k) {
      const int pos = k % kStages;
      const uint32_t use = uint32_t(k / kStages);
      if (k >= kStages) mbar_wait(&s.empty[pos], (use & 1u) ^ 1u, p.err);
#ifdef MUSE_VR_LDW_STAGES
      static_assert(!MUSE_VR_LDW_GATHER || kStages == 3, "the group arithmetic below assumes a 3-slot ring");
#else
      static_assert(kStages == 3, "the group arithmetic below assumes a 3-slot ring");
#endif
#if MUSE_VR_LDW_LANDED
      // the first copy of a later item into the ring: wait (once) until the head loaders finished every stage they
      // write (their generic stores and proxy fences precede their arrival on the head barrier)
      if (head_n > 0 && tok && !gat && !head_waited) {
        if (!MUSE_VR_LDW_TEST_SKIP_HEAD_DONE) named_bar_sync(MUSE_VR_LDW_HEAD_BAR, 128 + 32);
        head_waited = true;
      }
      if (false) {  // head stages: token rows from the head loaders (warps 4-7)
#else
      if (MUSE_VR_LDW_GATHER && k >= kStages && k - kStages < head_n * kStagesPerTile)
        cp_async_wait_newer(min(head_n * kStagesPerTile, k) - (k - kStages) - 1);  // the group of stage k - 3
      if (!MUSE_VR_LDW_THEAD && gat && tok && __shfl_sync(0xffffffffu, int(issued == k && !aborted()), lead)) {
#endif
        // the stage's token rows straight from the token tensor (written by predecessor grids), in the layout the row map
        // gives them (K atom a of row i at a * nb * 128 + i * 128, 16-byte pieces in 128-byte swizzle), then their scale
        // image (the layout the copy phase writes to img1: chunk c at c * 512, row i at 16 (i % 32) + 4 (i / 32)), 4 bytes
        // per row and K chunk; every lane's copies are tracked by the stage's full barrier before the leader's arrival,
        // so the stage completes only once they landed. Rows at and above the pass's row count are not copied: their MMA
        // columns are never read.
        const uint32_t rb = smem_u32(&s.st[pos][kStageA]), im = smem_u32(&s.st[pos][kStageA + kStageB]);
        // write-after-read across proxies: the stage's previous contents were read by the MMA (async proxy), whose
        // completion the empty barrier above acquired; order those reads before this lane's generic-proxy copies
        fence_proxy_async_smem();
        const int16_t* rt = s.ptok[m];
        const uint8_t* hsj = p.hs + size_t(j) * 256;
#pragma unroll 1
        for (int x = lane; x < rows * 16; x += 32) {
          const int i = x >> 4, a = (x >> 3) & 1, c = x & 7;
          cp_async_16(rb + uint32_t(a * nb * 128 + i * 128 + ((c ^ (i & 7)) << 4)),
                      hsj + size_t(rt[i]) * (kHidden / 2) + a * 128 + c * 16);
        }
#pragma unroll 1
        for (int x = lane; x < rows * kChunks; x += 32) {
          const int i = x >> 3, c = x & 7;  // 4 bytes of row i, K chunk c
          cp_async_4(im + uint32_t(c) * 512u + 16u * uint32_t(i & 31) + 4u * uint32_t(i >> 5),
                     p.hs_sf + size_t(rt[i]) * (kHidden / 16) + size_t(j) * 32 + 4 * c);
        }
        cp_async_arrive(&s.full[pos]);
        cp_async_commit();
        __syncwarp();
      }
      if (leader && issued == k && !aborted()) {
        const uint32_t dst = smem_u32(&s.st[pos][0]);
        const uint8_t* sf = sfw + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kChunks) * 512;
        if (k >= pre) {
          mbar_arrive_expect_tx(&s.full[pos], uint32_t(kStageA) + (tok ? tok_tx : 0u));
          // weights first: they depend on nothing
          if (ef) {
            tma_load_3d_hint(dst, wmap, 0, tile_row * kTile, j * (kChunks / 4), &s.full[pos], pol);
            bulk_load_hint(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[pos], pol);
          } else {
            tma_load_3d(dst, wmap, 0, tile_row * kTile, j * (kChunks / 4), &s.full[pos]);
            bulk_load(dst + kStageW, sf, uint32_t(kChunks * 512), &s.full[pos]);
          }
        } else {
          mbar_arrive_expect_tx(&s.full[pos], tok ? tok_tx : 0u);  // weights already out (expect-tx above)
        }
#if MUSE_VR_LDW_THEAD
        if (gat && tok) {
          // the head item's token rows: four-row gathers into the stage (K atom a of row i at a * nb * 128 + i * 128); rows past
          // the pass repeat its last row (their MMA columns are never read)
          const uint32_t rb = smem_u32(&s.st[pos][kStageA]);
          const int16_t* rt = s.ptok[m];
#pragma unroll 1
          for (int g = 0; g < (rows + 3) / 4; ++g) {
            const int r0 = rt[min(4 * g, rows - 1)], r1 = rt[min(4 * g + 1, rows - 1)], r2 = rt[min(4 * g + 2, rows - 1)],
                      r3 = rt[min(4 * g + 3, rows - 1)];
#pragma unroll
            for (int a = 0; a < 2; ++a)
              tma_gather4(rb + uint32_t(a * nb * 128 + g * 512), hm, (2 * j + a) * 128, r0, r1, r2, r3, &s.full[pos]);
          }
        } else {
          mbar_arrive(&s.full[pos]);  // the second arrival of a stage without a head image
        }
#endif
        if (tok && !gat) {
          // token rows / images are stored by other CTAs with generic stores: acquire their counts, then a proxy fence
          // orders the asynchronous-proxy copies after them (FC1: the grid's rows-ready count; FC2: the pass's built rows)
          if (!rows_ready) {
            rows_ready = acquire_at_least(cset + kRowsReady, gridDim.x, p.err) || true;
            LDW_STAMP(p, kStRows);
          }
#if MUSE_VR_LDW_BALB
          // rows of the board's passes: every CTA's second release (after its board-part copies)
          if (!it.fc2 && it.q >= s.bal_nq_p && !rows_ready_r)
            rows_ready_r = acquire_at_least(cset + kRowsReady + 1, gridDim.x, p.err) || true;
#endif
          if (!fc2_ready) fc2_ready = acquire_at_least(cset + kMaxPasses + it.q, unsigned(rows), p.err) || true;
          fence_proxy_async_global();
          tma_load_3d(dst + kStageA, bmap, 0, it.fc2 ? int(s.q_row2[it.q]) : qrow, j * (kChunks / 4), &s.full[pos]);
          bulk_load(dst + kStageA + kStageB, img + size_t(j) * kStageImg, uint32_t(kStageImg), &s.full[pos]);
        }
        ++issued;
      }
      __syncwarp();
    }
  }
  if (leader) LDW_STAMP(p, kStAllIssued);
#if MUSE_VR_LDW_BALB
  if (!bal_trig && (p.flags & kFlagTailTrigger)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
#if MUSE_VR_LDW_LANDED
  // a CTA whose only item is the head item: meet the loaders' arrival here (the barrier completes exactly once per launch)
  if (head_n > 0 && tok && !head_waited && !MUSE_VR_LDW_TEST_SKIP_HEAD_DONE) named_bar_sync(MUSE_VR_LDW_HEAD_BAR, 128 + 32);
#endif
  if (leader && aborted()) {
    // pre-issued weight stages without their arrival: arrive, so the phase completes once the weights land
#pragma unroll 1
    for (int kk = issued; kk < pre; ++kk) {
      mbar_arrive(&s.full[kk]);
#if MUSE_VR_LDW_THEAD
      if (kk >= head_n * kStagesPerTile) mbar_arrive(&s.full[kk]);  // no head image arrives on a later item's stage
#endif
    }
    const int out = issued > pre ? issued : pre;
#pragma unroll 1
    for (int kk = out > kStages ? out - kStages : 0; kk < out; ++kk)
      mbar_drain(&s.full[kk % kStages], uint32_t(kk / kStages) & 1u, p.err);
  }
  __syncwarp();
}

// warp 2: weight scales and token scales, shared memory -> tensor memory (one region of 32 columns each per ring stage)
__device__ MUSE_VR_LDW_ROLE void role_sfcopy(Smem& s, const Params& p) {
  const bool leader = elect_one();
#if MUSE_VR_LDW_BALB
  int n_items = s.n_items;
  bool bal_fin = false;
#else
  const int n_items = s.n_items;
#endif
  const uint32_t tbase = s.tmem_base;
  const int head_n = MUSE_VR_LDW_HEADS ? s.head_n : 0;
  int k = 0;
#pragma unroll 1
#if MUSE_VR_LDW_BALB
  for (int m = 0; m < bal_count(s, p, m, n_items, bal_fin); ++m) {
#else
  for (int m = 0; m < n_items; ++m) {
#endif
#pragma unroll 1
    for (int j = 0; j < kStagesPerTile; ++j, ++k) {
      const int pos = k % kStages;
      mbar_wait(&s.full[pos], uint32_t(k / kStages) & 1u, p.err);
#if MUSE_VR_LDW_LANDED
      if (m < head_n) mbar_wait(&s.tok_full[pos], uint32_t(k / kStages) & 1u, p.err);  // the loaders' token operand
#endif
      if (k == 0 && leader) LDW_STAMP(p, kStFirstFull);
      if (k >= kStages) mbar_wait(&s.empty[pos], (uint32_t(k / kStages) & 1u) ^ 1u, p.err);
      // head stages: the token-scale image was written by cp.async (generic proxy), made visible to this thread by the
      // full barrier; a proxy fence orders it before the tensor-memory copies (asynchronous proxy) that read it
      if (m < head_n) fence_proxy_async_smem();
      tc_fence_after();
      if (leader) {
        if (!aborted()) {
          const uint32_t ca = tbase + kSfaCol + uint32_t(pos) * 32u, cb = tbase + kSfbCol + uint32_t(pos) * 32u;
          const uint32_t sa = smem_u32(&s.st[pos][kStageW]), sb = smem_u32(&s.st[pos][kStageA + kStageB]);
#pragma unroll 1
          for (int c = 0; c < kChunks; ++c) {
            tc_cp_32x128b_x4(ca + 4u * c, sdesc_sf(sa + uint32_t(c) * 512u));
            tc_cp_32x128b_x4(cb + 4u * c, sdesc_sf(sb + uint32_t(c) * 512u));
          }
        }
        tc_commit(&s.sf_full[pos]);
      }
      __syncwarp();
    }
  }
  if (leader) tc_commit(&s.drain_sf);
  __syncwarp();
  mbar_drain(&s.drain_sf, 0u, p.err);
}

// warp 1: MMA issue (N = the pass's token-operand height)
__device__ MUSE_VR_LDW_ROLE void role_mma(Smem& s, const Params& p) {
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
#if MUSE_VR_LDW_BALB
  int n_items = s.n_items;
  bool bal_fin = false;
#else
  const int n_items = s.n_items;
#endif
  int k = 0;
#pragma unroll 1
#if MUSE_VR_LDW_BALB
  for (int m = 0; m < bal_count(s, p, m, n_items, bal_fin); ++m) {
#else
  for (int m = 0; m < n_items; ++m) {
#endif
    const Item it = item_unpack(s.items[m]);
    const int nb = bucket_rows(bucket_of(s.q_rows[it.q]));
    const uint32_t idesc = idesc_nvf4(uint32_t(nb));
    const bool gat = MUSE_VR_LDW_HEADS && m < s.head_n;
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    if ((p.dbg & 128) == 0 && m >= 2) mbar_wait(&s.tmem_empty[acc], aph ^ 1u, p.err);
    tc_fence_after();
    const uint32_t d = tbase + kAccCol + uint32_t(acc) * 128u;
#pragma unroll 1
    for (int j = 0; j < kStagesPerTile; ++j, ++k) {
      const int pos = k % kStages;
      mbar_wait(&s.sf_full[pos], uint32_t(k / kStages) & 1u, p.err);
      if (gat) {
        // token rows written by cp.async (generic proxy): visible to this thread through the full barrier, then a proxy
        // fence before the MMAs (asynchronous proxy) read them
        mbar_wait(&s.full[pos], uint32_t(k / kStages) & 1u, p.err);
#if MUSE_VR_LDW_LANDED
        mbar_wait(&s.tok_full[pos], uint32_t(k / kStages) & 1u, p.err);  // the loaders' token operand
#endif
        fence_proxy_async_smem();
      }
      tc_fence_after();
      if (leader) {
        if ((p.dbg & 1) == 0 && !aborted()) {
          const uint64_t da0 = sdesc_sw128(smem_u32(&s.st[pos][0]));
          const uint64_t db0 = sdesc_sw128(smem_u32(&s.st[pos][kStageA]));
          const uint32_t sfa0 = tbase + kSfaCol + uint32_t(pos) * 32u, sfb0 = tbase + kSfbCol + uint32_t(pos) * 32u;
#if MUSE_VR_LDW_K128
          // K = 128 per MMA (instruction descriptor K field), two 64-wide chunks per instruction
#pragma unroll
          for (int c = 0; c < kChunks; c += 2) {
            const uint64_t da = da0 + uint64_t((uint32_t(c >> 2) * 16384u + uint32_t(c & 3) * 32u) >> 4);
            const uint64_t db = db0 + uint64_t((uint32_t(c >> 2) * uint32_t(nb * 128) + uint32_t(c & 3) * 32u) >> 4);
            tc_mma_nvf4(d, da, db, idesc | (1u << 3), sfa0 + 4u * c, sfb0 + 4u * c, (j | c) != 0 ? 1u : 0u);
          }
#else
#pragma unroll
          for (int c = 0; c < kChunks; ++c) {
            const uint64_t da = da0 + uint64_t((uint32_t(c >> 2) * 16384u + uint32_t(c & 3) * 32u) >> 4);
            const uint64_t db = db0 + uint64_t((uint32_t(c >> 2) * uint32_t(nb * 128) + uint32_t(c & 3) * 32u) >> 4);
            tc_mma_nvf4(d, da, db, idesc, sfa0 + 4u * c, sfb0 + 4u * c, (j | c) != 0 ? 1u : 0u);
          }
#endif
        }
        tc_commit(&s.empty[pos]);
      }
      __syncwarp();
    }
    if (leader) tc_commit(&s.tmem_full[acc]);
    __syncwarp();
  }
  if (leader) tc_commit(&s.drain_mma);
  __syncwarp();
  mbar_drain(&s.drain_mma, 0u, p.err);
  if (leader) LDW_STAMP(p, kStMma);
}

// Warps 4-11 (epilogue + builder warps, idle until the first accumulator / FC1 count): this CTA's share of the permuted token
// rows (slots b, b + G, ...): FP4 rows in 16-byte pieces with several loads in flight per thread, per-token scales,
// token-scale image entries, the expanded-index map and top-k weights of those slots; then one release of the grid's
// rows-ready count on behalf of the eight warps.
__device__ __noinline__ void copy_rows(Smem& s, const Params& p, int wt) {  // one shared copy of this cold code
  constexpr int kW = 256;  // worker threads
  const int n_my = s.n_my;
  // One warp per copied row (rolled loop: this code runs once per launch, so its size is instruction-fetch time):
  // lane l moves row pieces l, l + 32, l + 64, l + 96 and token-scale piece l (< 16) of the row, then writes the 4
  // image words of that piece; the id / top-k weight / per-token scale of slot wt are loaded alongside.
  const int lane = wt & 31, wi = wt >> 5;
  const bool tail = wt < n_my;
  const int sl_t = tail ? s.my_slot[wt] : 0;
  const int32_t pk_t = tail ? p.packed[sl_t] : 0;
  const float pt_t = tail ? p.pts[sl_t / kTopK] : 0.0f;
  if (wt == 0) LDW_STAMP(p, kStCopyIn);
#pragma unroll 1
  for (int j = wi; j < n_my; j += kW / 32) {
    const int pos = s.my_pos[j];
    if (pos < 0) continue;  // warp-uniform
    const int t = s.my_slot[j] / kTopK;
    const uint4* src = reinterpret_cast<const uint4*>(p.hs + size_t(t) * (kHidden / 2));
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = src[lane + 32 * u];
    uint4 f = make_uint4(0u, 0u, 0u, 0u);
    if (lane < 16) f = reinterpret_cast<const uint4*>(p.hs_sf + size_t(t) * (kHidden / 16))[lane];
    uint4* dst = reinterpret_cast<uint4*>(p.hs_perm + size_t(pos) * (kHidden / 2));
#pragma unroll
    for (int u = 0; u < 4; ++u) dst[lane + 32 * u] = v[u];
    if (lane < 16) {
      // image words of K chunks g = 4 lane .. 4 lane + 3 (one per 64-wide chunk)
      const int e = s.my_exp[j];
#if MUSE_VR_LDW_TEST_EC_TWIN == 1  // tests only: the expert's bases may be unwritten here (clamped)
      const int r = ec_twin_clamp(pos - s.base[e], kMaxSlots - 1);
      const int q = ec_twin_clamp(s.qbase[e] + r / kPassRows, kMaxPasses - 1), i = r % kPassRows;
#else
      const int r = pos - s.base[e], q = s.qbase[e] + r / kPassRows, i = r % kPassRows;
#endif
      uint8_t* img = p.img1 + size_t(q) * kImgBytes + size_t(4 * lane) * 512 + size_t(16 * (i & 31) + 4 * (i >> 5));
      *reinterpret_cast<uint32_t*>(img) = f.x;
      *reinterpret_cast<uint32_t*>(img + 512) = f.y;
      *reinterpret_cast<uint32_t*>(img + 1024) = f.z;
      *reinterpret_cast<uint32_t*>(img + 1536) = f.w;
    }
  }
  if (wt == 0) LDW_STAMP(p, kStCopyRows);
#if MUSE_VR_LDW_SPLIT_REL
  named_bar_sync(7, kW);  // every worker's row and image stores, before the rows-ready release on their behalf
#if MUSE_VR_LDW_TEST_EC_TWIN == 1  // tests only: the counter parity may be unwritten here (masked)
  if (wt == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + (s.set & 1) * kCntSet + kRowsReady) : "memory");
#else
  if (wt == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + s.set * kCntSet + kRowsReady) : "memory");
#endif
#endif
  if (tail) {
    const int pos = s.my_pos[wt];
    p.e2p[sl_t] = pos;
    p.expert_weights[sl_t] = uint16_t(pk_t & 0xFFFF);
    if (pos >= 0) p.pts_perm[pos] = pt_t;
#if MUSE_VR_LDW_HB0_PUSH
    if (pos >= 0) p.hb_slot[pos] = sl_t;  // read by the push warps after their rows-ready acquire
#endif
#if MUSE_VR_LDW_ROWS_READY
    // the pass of the slot's permuted row (the pass layout of the token-scale images above), or -1
    int16_t pass = -1;
    if (pos >= 0) {
      const int e = s.my_exp[wt];
      pass = int16_t(s.qbase[e] + (pos - s.base[e]) / kPassRows);
    }
    reinterpret_cast<int16_t*>(p.rr + kRrFlagBase + kRrFlags)[sl_t] = pass;
#endif
  }
#if MUSE_VR_LDW_ROWS_READY
  if (blockIdx.x == 0 && wt == 0) p.rr[1] = s.rr_epoch;  // the published epoch (ordered by the barrier + fence below)
#endif
  if (wt == 0) LDW_STAMP(p, kStCopyImg);
  named_bar_sync(7, kW);  // every worker's stores, before one release on their behalf
  if (wt == 0) LDW_STAMP(p, kStCopyBar);
  if (wt == 0) {
    // SPLIT_REL: the metadata count (the rows-ready count was raised above)
#if MUSE_VR_LDW_TEST_EC_TWIN == 1  // tests only: the counter parity may be unwritten here (masked)
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + (s.set & 1) * kCntSet + kMetaReady) : "memory");
#else
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + s.set * kCntSet + kMetaReady) : "memory");
#endif
    LDW_STAMP(p, kStCopy);
#if MUSE_VR_LDW_EARLY_SLOTTAB
    // the slot table (this CTA's share of e2p / expert_weights, ordered here by the named barrier) before the trigger
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  }
}

#if MUSE_VR_LDW_COPY_DIET
// copy_rows with its operands by value (MUSE_VR_LDW_COPY_DIET): the kernel's pointers come from the caller, the CTA's slot
// tables (my_slot / my_pos / my_exp, base / qbase of the local experts, n_my) are read at shared-window address sb with
// shared loads. Same rows, images and per-slot stores as copy_rows, then the same barrier 7 and release(s) of the grid counts
// in cset. (stamps: the timing builds' stamp rows only.)
// No __restrict__ on the token inputs: they come from predecessor grids that may still run when this grid starts (programmatic
// launch), so their loads must stay ordinary coherent loads, ordered by the dependency wait (never the read-only path).
__device__ __noinline__ void copy_rows_v(uint32_t sb, int wt, const uint8_t* hs, const uint8_t* hs_sf, uint8_t* hs_perm,
                                         uint8_t* img1, const int32_t* packed,
                                         const float* pts, int32_t* e2p, uint16_t* ew, float* pts_perm, unsigned* cset,
                                         int32_t* hb_slot, unsigned long long* stamps) {
  constexpr int kW = 256;  // worker threads
  const struct {
    unsigned long long* stamps;
  } p{stamps};
  const uint32_t a_slot = sb + uint32_t(offsetof(Smem, my_slot)), a_pos = sb + uint32_t(offsetof(Smem, my_pos));
  const uint32_t a_exp = sb + uint32_t(offsetof(Smem, my_exp)), a_base = sb + uint32_t(offsetof(Smem, base));
  const uint32_t a_qbase = sb + uint32_t(offsetof(Smem, qbase));
  const int n_my = lds_s32(sb + uint32_t(offsetof(Smem, n_my)));
  const int lane = wt & 31, wi = wt >> 5;
  const bool tail = wt < n_my;
  const int sl_t = tail ? lds_s16(a_slot + 2u * uint32_t(wt)) : 0;
  const int pos_t = tail ? lds_s16(a_pos + 2u * uint32_t(wt)) : -1;
  const int32_t pk_t = tail ? packed[sl_t] : 0;
  const float pt_t = tail ? pts[sl_t / kTopK] : 0.0f;
  (void)p;
  (void)hb_slot;
  if (wt == 0) LDW_STAMP(p, kStCopyIn);
#pragma unroll 1
  for (int j = wi; j < n_my; j += kW / 32) {
    const int pos = lds_s16(a_pos + 2u * uint32_t(j));
    if (pos < 0) continue;  // warp-uniform
    const int t = lds_s16(a_slot + 2u * uint32_t(j)) / kTopK;
    const uint4* src = reinterpret_cast<const uint4*>(hs + size_t(t) * (kHidden / 2));
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = src[lane + 32 * u];
    uint4 f = make_uint4(0u, 0u, 0u, 0u);
    if (lane < 16) f = reinterpret_cast<const uint4*>(hs_sf + size_t(t) * (kHidden / 16))[lane];
    uint4* dst = reinterpret_cast<uint4*>(hs_perm + size_t(pos) * (kHidden / 2));
#pragma unroll
    for (int u = 0; u < 4; ++u) dst[lane + 32 * u] = v[u];
    if (lane < 16) {
      // image words of K chunks g = 4 lane .. 4 lane + 3 (one per 64-wide chunk)
      const int e = lds_s16(a_exp + 2u * uint32_t(j));
#if MUSE_VR_LDW_TEST_EC_TWIN == 1  // tests only: the expert's bases may be unwritten here (clamped)
      const int r = ec_twin_clamp(pos - lds_s16(a_base + 2u * uint32_t(e)), kMaxSlots - 1);
      const int q = ec_twin_clamp(lds_s16(a_qbase + 2u * uint32_t(e)) + r / kPassRows, kMaxPasses - 1);
#else
      const int r = pos - lds_s16(a_base + 2u * uint32_t(e)), q = lds_s16(a_qbase + 2u * uint32_t(e)) + r / kPassRows;
#endif
      const int i = r % kPassRows;
      uint8_t* img = img1 + size_t(q) * kImgBytes + size_t(4 * lane) * 512 + size_t(16 * (i & 31) + 4 * (i >> 5));
      *reinterpret_cast<uint32_t*>(img) = f.x;
      *reinterpret_cast<uint32_t*>(img + 512) = f.y;
      *reinterpret_cast<uint32_t*>(img + 1024) = f.z;
      *reinterpret_cast<uint32_t*>(img + 1536) = f.w;
    }
  }
  if (wt == 0) LDW_STAMP(p, kStCopyRows);
#if MUSE_VR_LDW_SPLIT_REL
  named_bar_sync(7, kW);  // every worker's row and image stores, before the rows-ready release on their behalf
  if (wt == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cset + kRowsReady) : "memory");
#endif
  if (tail) {
    e2p[sl_t] = pos_t;
    ew[sl_t] = uint16_t(pk_t & 0xFFFF);
    if (pos_t >= 0) pts_perm[pos_t] = pt_t;
#if MUSE_VR_LDW_HB0_PUSH
    if (pos_t >= 0) hb_slot[pos_t] = sl_t;  // read by the push warps after their rows-ready acquire
#endif
  }
  if (wt == 0) LDW_STAMP(p, kStCopyImg);
  named_bar_sync(7, kW);  // every worker's stores, before one release on their behalf
  if (wt == 0) LDW_STAMP(p, kStCopyBar);
  if (wt == 0) {
    // SPLIT_REL: the metadata count (the rows-ready count was raised above)
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cset + kMetaReady) : "memory");
    LDW_STAMP(p, kStCopy);
  }
}
#if MUSE_VR_LDW_HB0_PUSH
#define LDW_HB_SLOT p.hb_slot
#else
#define LDW_HB_SLOT nullptr
#endif
#define LDW_COPY_ROWS(s, p, cset, wt)                                                                                   \
  copy_rows_v(smem_u32(&(s)), (wt), (p).hs, (p).hs_sf, (p).hs_perm, (p).img1, (p).packed, (p).pts, (p).e2p,            \
              (p).expert_weights, (p).pts_perm, (cset), LDW_HB_SLOT, (p).stamps)
#else
#define LDW_COPY_ROWS(s, p, cset, wt) copy_rows((s), (p), (wt))
#endif

#if MUSE_VR_LDW_LANDED
// Warps 8-11 (landed build): the copy phase from the token landing. One warp per copied row: lane l takes the row's word
// pairs l, l + 32, ... (a pair = 12 payload bytes = three 32-bit words), all its loads in flight before any check; FP4
// words go to the permuted row, block-scale words to the token-scale image entries of the row's pass (the layout of the
// plain copy phase); thread wt then writes the expanded-index map entry, the top-k weight (from the landed id word) and
// the per-token scale of slot wt, and one release of the rows-ready count follows, on behalf of the four warps.
__device__ __noinline__ void landed_copy_rows(Smem& s, const Params& p, int wt) {
  constexpr int kW = 128, kPairs = kTokSfDword / 3 + kHidden / 64 / 3 + 1;  // pairs holding words 0 .. kTokPtsDword - 1
  static_assert(3 * kPairs >= kTokPtsDword && 3 * (kPairs - 1) < kTokPtsDword, "pairs of the FP4 and block-scale words");
  constexpr int kPerLane = (kPairs + 31) / 32;
  const int n_my = s.n_my;
  const int lane = wt & 31, wi = wt >> 5;
  const uint32_t tag = s.tok_tag;
  const bool go = s.land_go != 0;
  const bool tail = wt < n_my;
  const int sl_t = tail ? s.my_slot[wt] : 0;
  const unsigned long long* const idw = ids_word(s, p, sl_t);
  const unsigned long long* const trow_t = tok_row(s, p, sl_t / kTopK);
  const unsigned long long pk_raw = (tail && go) ? ld_relaxed_sys_u64(idw) : 0ull;
  if (wt == 0) LDW_STAMP(p, kStCopyIn);
#if MUSE_VR_LDW_BALB
  // two parts: the pinned slots, then (once the board part is published) the candidate slots, each followed by its own
  // release of a rows-ready count; candidate slots are never read in the first part (their positions are written meanwhile)
  const bool cand_t = tail && s.my_cand[wt] != 0;
  bool ok = true;
#pragma unroll 1
  for (int part = 0; part < 2; ++part) {
    if (part == 1) ok = bal_acquire(s, p);
#else
  {
    constexpr int part = 0;
    constexpr bool ok = true;
#endif
#pragma unroll 1
  for (int j = wi; j < n_my && go && ok; j += kW / 32) {
#if MUSE_VR_LDW_BALB
    if ((s.my_cand[j] != 0) != (part == 1)) continue;  // warp-uniform
#endif
    const int pos = s.my_pos[j];
    if (pos < 0) continue;  // warp-uniform
    const unsigned long long* row = tok_row(s, p, s.my_slot[j] / kTopK);
    ulonglong2 v[kPerLane];
#pragma unroll
    for (int u = 0; u < kPerLane; ++u) {
      const int jp = lane + 32 * u;
      if (jp < kPairs) v[u] = ld_relaxed_sys_v2u64(row + 2 * jp);
    }
    const int e = s.my_exp[j], r = pos - s.base[e], q = s.qbase[e] + r / kPassRows, i = r % kPassRows;
    uint8_t* const img = p.img1 + size_t(q) * kImgBytes + size_t(16 * (i & 31) + 4 * (i >> 5));
    uint32_t* const dst = reinterpret_cast<uint32_t*>(p.hs_perm + size_t(pos) * (kHidden / 2));
#pragma unroll
    for (int u = 0; u < kPerLane; ++u) {
      const int jp = lane + 32 * u;
      if (jp < kPairs) {
        uint32_t d[3];
#if MUSE_VR_LDW_TEST_EXPIRE == 2
        if (j == wi && jp == 0 && wt == 0) v[u].x = land_poll(row, tag ^ 0x5555u, kTokTagShift, p.err);  // tests only
#endif
        tok_dwords(land_accept(v[u].x, row + 2 * jp, tag, kTokTagShift, p.err),
                   land_accept(v[u].y, row + 2 * jp + 1, tag, kTokTagShift, p.err), d);
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const int dw = 3 * jp + c;
          if (dw < kTokSfDword) dst[dw] = d[c];
          else if (dw < kTokPtsDword) *reinterpret_cast<uint32_t*>(img + size_t(dw - kTokSfDword) * 512) = d[c];
        }
      }
    }
  }
  if (wt == 0) LDW_STAMP(p, kStCopyRows);
#if MUSE_VR_LDW_BALB
  if (tail && go && ok && cand_t == (part == 1)) {
#else
  if (tail && go) {
#endif
    const int pos = s.my_pos[wt];
    const uint32_t pk = uint32_t(land_accept(pk_raw, idw, s.ids_tag, kIdsTagShift, p.err));
    p.e2p[sl_t] = pos;
    p.expert_weights[sl_t] = uint16_t(pk & 0xFFFFu);
    if (pos >= 0) p.pts_perm[pos] = __uint_as_float(tok_dword(trow_t, kTokPtsDword, tag, p.err));
#if MUSE_VR_LDW_EPI
    if (pos >= 0) p.pos_slot[pos] = sl_t;
#endif
  }
  if (wt == 0) LDW_STAMP(p, kStCopyImg);
  named_bar_sync(7, kW);  // every worker's stores, before one release on their behalf
  if (wt == 0) LDW_STAMP(p, kStCopyBar);
  if (wt == 0) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + s.set * kCntSet + kRowsReady + part) : "memory");
    LDW_STAMP(p, kStCopy);
  }
  }  // part
}

// Warps 4-7 before their epilogue role (landed build): the token operand of this CTA's first FC1 item (when the prologue
// made it a head item), every ring stage of it, from the token landing by token index: per stage the pass's rows x 16
// pieces of 16 payload bytes (two word pairs each, all four words accepted) into the 128-byte-swizzled token region and
// the rows' block-scale words into the stage's scale image, the layouts the TMA path produces. Each warp fences its
// shared-memory stores against the asynchronous proxy (tensor-core and tensor-memory copies read them) and gives one of
// the four arrivals of the stage's token barrier (tok_full; the producer warp's full barrier carries the weights). Then each
// warp keeps the rows' per-token scales for its own FC1 epilogue, and the four warps arrive on the head barrier, which the
// producer warp waits on before it copies into a ring slot these warps wrote.
__device__ __noinline__ void landed_head(Smem& s, const Params& p, int wt) {
  constexpr int kW = 128, kB = 4;  // threads; pieces per thread with loads in flight together
  const int lane = wt & 31, lw = wt >> 5;
#if MUSE_VR_LDW_HEAD_GO
  // called before the end-of-prologue barrier: warp 3's head-row list via its flag (lane 0 acquires, bounded; the warp
  // barrier orders the other lanes' reads after it). An expired wait sets the error word and continues (no hang: every
  // later wait is bounded).
  if ((p.dbg & 256) != 0) return;
  if (lane == 0) {
    uint32_t n = 0;
    while (ld_acquire_cta_shared(&s.head_go) == 0) {
      if (++n > kMaxPolls) {
        fail(p.err, kErrPoll);
        break;
      }
    }
  }
  __syncwarp();
  if (s.head_n == 0) return;  // CTA-uniform: the producer warp gives these arrivals then
  const int rows = s.q_rows[s.head_q];
#else
  if (s.head_n == 0 || (p.dbg & 256) != 0) return;  // CTA-uniform: the producer warp gives these arrivals then
  const Item it = item_unpack(s.items[0]);
  const int rows = s.q_rows[it.q];
#endif
  const int nb = bucket_rows(bucket_of(rows));
  const uint32_t tag = s.tok_tag;
  const int16_t* const rt = s.ptok[0];
#pragma unroll 1
  for (int j = 0; j < kStagesPerTile; ++j) {
    const int pos = j % kStages;  // item 0's stages are the CTA's first ring stages
    bool got = true;  // the slot is this stage's (first use, or its previous use consumed)
    if (j >= kStages) {
      got = __all_sync(0xffffffffu, mbar_wait(&s.empty[pos], (uint32_t(j / kStages) & 1u) ^ 1u, p.err));
      // write-after-read across proxies: the MMA's reads of this slot (acquired through the empty barrier) before
      // these generic stores
      fence_proxy_async_smem();
    }
    const uint32_t rb = smem_u32(&s.st[pos][kStageA]), im = smem_u32(&s.st[pos][kStageA + kStageB]);
    if (j == 0 && wt == 0) LDW_STAMP(p, kStFc2Dep);  // timing builds: the head loaders start stage 0
#if MUSE_VR_LDW_TEST_HEAD_DELAY > 0
    if (j + 1 == kStagesPerTile)
      for (int z = 0; z < MUSE_VR_LDW_TEST_HEAD_DELAY / 1000; ++z) __nanosleep(1000);  // tests only
#endif
#if MUSE_VR_LDW_HEAD_PRIO
    bool iss = false;  // HEAD_PRIO 1: this warp's arrival on head_iss (exactly once)
    constexpr int kP = (kHeadRows * 16 + kW - 1) / kW, kS = (kHeadRows * kChunks + kW - 1) / kW;
    if (kP + kS <= 16 && got && !aborted() && j == 0) {
      // the first stage's token operand as ONE batch per thread (builds whose head rows fit 16 pieces per thread): every
      // payload piece and every scale word in flight together, then their decode and stores
#pragma unroll 1
      for (int pass = 0; pass < 1 + MUSE_VR_LDW_TEST_HEAD_TWICE; ++pass) {
      ulonglong2 vp[kP][2], vs[kS];
      const unsigned long long* rpp[kP];
      const unsigned long long* rps[kS];
#pragma unroll
      for (int b = 0; b < kP; ++b) {
        const int x = wt + b * kW;
        rpp[b] = nullptr;
        if (x < rows * 16) {
          const int i = x >> 4, P = 8 * ((x >> 3) & 1) + (x & 7);
          rpp[b] = tok_row(s, p, rt[i]) + 2 * ((4 * P) / 3);
          vp[b][0] = ld_head_v2u64(rpp[b]);
          vp[b][1] = ld_head_v2u64(rpp[b] + 2);
        }
      }
#pragma unroll
      for (int b = 0; b < kS; ++b) {
        const int x = wt + b * kW;
        rps[b] = nullptr;
        if (x < rows * kChunks) {
          const int dw = kTokSfDword + (x & 7);
          rps[b] = tok_row(s, p, rt[x >> 3]) + 2 * (dw / 3);
          vs[b] = ld_head_v2u64(rps[b]);
        }
      }
      if (wt == 0 && pass == 0) LDW_STAMP(p, kStHeadIss);  // timing builds: warp 4's batch is issued
#if MUSE_VR_LDW_HEAD_PRIO == 1
      if (pass == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.head_iss);
        iss = true;
      }
#endif
#pragma unroll
      for (int b = 0; b < kP; ++b) {
        const int x = wt + b * kW;
        if (x < rows * 16) {
          const int i = x >> 4, a = (x >> 3) & 1, c = x & 7, P = 8 * a + c, r0 = 4 * P - 3 * ((4 * P) / 3);
          uint32_t d0[3], d1[3];
#if MUSE_VR_LDW_TEST_EXPIRE == 1
          if (x == 0) vp[b][0].x = land_poll(rpp[b], tag ^ 0x5555u, kTokTagShift, p.err);  // tests only
#endif
          tok_dwords(land_accept(vp[b][0].x, rpp[b], tag, kTokTagShift, p.err),
                     land_accept(vp[b][0].y, rpp[b] + 1, tag, kTokTagShift, p.err), d0);
          tok_dwords(land_accept(vp[b][1].x, rpp[b] + 2, tag, kTokTagShift, p.err),
                     land_accept(vp[b][1].y, rpp[b] + 3, tag, kTokTagShift, p.err), d1);
          const uint4 w4 = r0 == 0 ? make_uint4(d0[0], d0[1], d0[2], d1[0])
                         : r0 == 1 ? make_uint4(d0[1], d0[2], d1[0], d1[1])
                                   : make_uint4(d0[2], d1[0], d1[1], d1[2]);
          sts_v4(rb + uint32_t(a * nb * 128 + i * 128 + ((c ^ (i & 7)) << 4)), w4);
        }
      }
#pragma unroll
      for (int b = 0; b < kS; ++b) {
        const int x = wt + b * kW;
        if (x < rows * kChunks) {
          const int i = x >> 3, c = x & 7, dw = kTokSfDword + c;
          uint32_t d[3];
          tok_dwords(land_accept(vs[b].x, rps[b], tag, kTokTagShift, p.err),
                     land_accept(vs[b].y, rps[b] + 1, tag, kTokTagShift, p.err), d);
          const int r0 = dw - 3 * (dw / 3);
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(im + uint32_t(c) * 512u + 16u * uint32_t(i & 31) + 4u * uint32_t(i >> 5)),
                       "r"(r0 == 0 ? d[0] : r0 == 1 ? d[1] : d[2])
                       : "memory");
        }
      }
#if MUSE_VR_LDW_TEST_HEAD_TWICE && !MUSE_VR_LDW_HEAD_GO
      if (wt == 0 && pass == 0) LDW_STAMP(p, kStW0);  // tests only: the first pass's stores are done (warp 4)
#endif
      }  // pass
    } else
#endif
    if (got && !aborted()) {
#pragma unroll 1
      for (int x0 = wt; x0 < rows * 16; x0 += kW * kB) {
        ulonglong2 v[kB][2];
        const unsigned long long* rp[kB];
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const int x = x0 + b * kW;
          rp[b] = nullptr;
          if (x < rows * 16) {
            const int i = x >> 4, P = 16 * j + 8 * ((x >> 3) & 1) + (x & 7);
            rp[b] = tok_row(s, p, rt[i]) + 2 * ((4 * P) / 3);
            v[b][0] = ld_relaxed_sys_v2u64(rp[b]);
            v[b][1] = ld_relaxed_sys_v2u64(rp[b] + 2);
          }
        }
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const int x = x0 + b * kW;
          if (x < rows * 16) {
            const int i = x >> 4, a = (x >> 3) & 1, c = x & 7, P = 16 * j + 8 * a + c, r0 = 4 * P - 3 * ((4 * P) / 3);
            uint32_t d0[3], d1[3];
#if MUSE_VR_LDW_TEST_EXPIRE == 1
            if (j == 0 && x == 0) v[b][0].x = land_poll(rp[b], tag ^ 0x5555u, kTokTagShift, p.err);  // tests only
#endif
            tok_dwords(land_accept(v[b][0].x, rp[b], tag, kTokTagShift, p.err),
                       land_accept(v[b][0].y, rp[b] + 1, tag, kTokTagShift, p.err), d0);
            tok_dwords(land_accept(v[b][1].x, rp[b] + 2, tag, kTokTagShift, p.err),
                       land_accept(v[b][1].y, rp[b] + 3, tag, kTokTagShift, p.err), d1);
            const uint4 w4 = r0 == 0 ? make_uint4(d0[0], d0[1], d0[2], d1[0])
                           : r0 == 1 ? make_uint4(d0[1], d0[2], d1[0], d1[1])
                                     : make_uint4(d0[2], d1[0], d1[1], d1[2]);
            sts_v4(rb + uint32_t(a * nb * 128 + i * 128 + ((c ^ (i & 7)) << 4)), w4);
          }
        }
      }
#pragma unroll 1
      for (int x0 = wt; x0 < rows * kChunks; x0 += kW * kB) {
        ulonglong2 v[kB];
        const unsigned long long* rp[kB];
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const int x = x0 + b * kW;
          rp[b] = nullptr;
          if (x < rows * kChunks) {
            const int dw = kTokSfDword + kChunks * j + (x & 7);
            rp[b] = tok_row(s, p, rt[x >> 3]) + 2 * (dw / 3);
            v[b] = ld_relaxed_sys_v2u64(rp[b]);
          }
        }
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          const int x = x0 + b * kW;
          if (x < rows * kChunks) {
            const int i = x >> 3, c = x & 7, dw = kTokSfDword + kChunks * j + c;
            uint32_t d[3];
            tok_dwords(land_accept(v[b].x, rp[b], tag, kTokTagShift, p.err),
                       land_accept(v[b].y, rp[b] + 1, tag, kTokTagShift, p.err), d);
            const int r0 = dw - 3 * (dw / 3);
            asm volatile("st.shared.b32 [%0], %1;" ::"r"(im + uint32_t(c) * 512u + 16u * uint32_t(i & 31) + 4u * uint32_t(i >> 5)),
                         "r"(r0 == 0 ? d[0] : r0 == 1 ? d[1] : d[2])
                         : "memory");
          }
        }
      }
    }
#if MUSE_VR_LDW_HEAD_PRIO == 1
    if (j == 0 && !iss) {  // abort path: the producer's wait still completes
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.head_iss);
    }
#endif
    fence_proxy_async_smem();
    __syncwarp();
    // one arrival per warp on the stage's token barrier, only when the slot was this stage's: on an abort path a reused
    // slot whose previous use was not consumed gets no arrival (its readers' waits are bounded and end on the abort)
    if (got && lane == 0) mbar_arrive(&s.tok_full[pos]);
    if (j == 0 && wt == 0) LDW_STAMP(p, kStCopyLd);  // timing builds: warp 4's stage-0 token operand is in
  }
  // the item's per-token scales, one copy per warp (each epilogue warp reads only its own)
  for (int i = lane; i < rows; i += 32)
    s.hpts[MUSE_VR_LDW_TEST_PLANT_RACE ? 0 : lw][i] =
        aborted() ? 0.0f : __uint_as_float(tok_dword(tok_row(s, p, rt[i]), kTokPtsDword, tag, p.err));
  // every loader store and proxy fence is done: the producer warp (head barrier, 128 + 32 threads) waits for this before it
  // copies into a ring slot these warps wrote
  named_bar_arrive(MUSE_VR_LDW_HEAD_BAR, kW + 32);
}

__device__ __forceinline__ void landed_guard_exit(const Smem& s, const Params& p) {
  if (threadIdx.x == 0 && s.land_go != 0)
    *reinterpret_cast<volatile unsigned long long*>(p.last_seq + blockIdx.x) = s.land_seq;
}
#endif

// warps 4-7: epilogue (TMEM lane quadrant = warp % 4): 16 (FC1) / 8 (FC2) token columns per tensor-memory load
__device__ MUSE_VR_LDW_ROLE void role_epilogue(Smem& s, const Params& p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int q = warp & 3;
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
#if MUSE_VR_LDW_BALB
  int n_items = s.n_items;
  bool bal_fin = false, need_rows_r = true;
#else
  const int n_items = s.n_items;
#endif
  const float gsi = p.gsi;
  unsigned* const cset = p.cnt + s.set * kCntSet;
  const int head_n = MUSE_VR_LDW_HEADS ? s.head_n : 0;
#if MUSE_VR_LDW_TEST_RR_DELAY_NS || MUSE_VR_LDW_TEST_RR_PLANT
  bool rr_first = blockIdx.x == 0;  // tests only: this warp's first FC2 item in CTA 0
#endif
#if MUSE_VR_LDW_LANDED
#if !MUSE_VR_LDW_HEAD_GO  // HEAD_GO: the kernel body ran it before the list barrier
  landed_head(s, p, threadIdx.x - 4 * 32);
#endif
#elif !MUSE_VR_LDW_EARLY_COPY  // EARLY_COPY: the kernel body ran the copy phase before the list barrier
  LDW_COPY_ROWS(s, p, cset, threadIdx.x - 4 * 32);
#endif
  if (p.dbg & 128) return;
  // per-token scales of FC1 rows come from other CTAs' row copies (their release of the rows-ready count); head items
  // read them by token (pts, from predecessor grids), so with head copies the count is acquired before the first other
  // item
  bool need_rows = n_items > 0;
  if (head_n == 0 && need_rows) {
    if (lane == 0) acquire_at_least(cset + kMetaReady, gridDim.x, p.err);
    need_rows = false;
  }
  __syncwarp();
#pragma unroll 1
#if MUSE_VR_LDW_BALB
  for (int m = 0; m < bal_count(s, p, m, n_items, bal_fin); ++m) {
#else
  for (int m = 0; m < n_items; ++m) {
#endif
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    const Item it = item_unpack(s.items[m]);
#if MUSE_VR_SLOT_TEST == 3
    if (!it.fc2 && q == 3)  // tests only: warp skew on FC1 tiles
      for (int z = 0; z < 64; ++z) __nanosleep(1000);
#endif
    const int e = s.q_expert[it.q], rows = s.q_rows[it.q], qrow = s.q_row[it.q];
    if (need_rows && m >= head_n) {
      if (lane == 0) acquire_at_least(cset + kMetaReady, gridDim.x, p.err);
      need_rows = false;
    }
#if MUSE_VR_LDW_BALB
    if (need_rows_r && !it.fc2 && it.q >= s.bal_nq_p) {  // per-token scales of the board's passes' rows
      if (lane == 0) acquire_at_least(cset + kRowsReady + 1, gridDim.x, p.err);
      need_rows_r = false;
    }
#endif
    if (it.fc2 && lane == 0) acquire_at_least(cset + kMaxPasses + it.q, unsigned(rows), p.err);  // the builders' maxima
    __syncwarp();
    mbar_wait_idle(&s.tmem_full[acc], aph, p.err);
    tc_fence_after();
    const uint32_t tacc = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(acc) * 128u;
    if (!it.fc2) {
      // FC1: SwiGLU (stock arithmetic) -> activation words of 4 bf16, one routed row per column
      const float a_gate = p.gate_alpha[e], a_up = p.up_alpha[e];
      // TMEM lane a + 8 g + 16 b of this quadrant (a < 8): g = 0 the up row, g = 1 the gate row of intermediate column
      // 16 q + 2 a + b of this tile. The 16x256b fragment gives thread t both rows of a = t / 4 for tokens 2 (t % 4) and
      // 2 (t % 4) + 1 (+ 8 for the .x2 half); the loads at b = 0 and b = 1 give columns 2 a and 2 a + 1, stored as one
      // 32-bit word per token.
      const int ta = lane >> 2, tx = 2 * (lane & 3);
      uint32_t* dst = reinterpret_cast<uint32_t*>(p.act + size_t(qrow) * kActWords + size_t(it.tile) * 16) + (8 * q + ta);
      const uint32_t lb = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(acc) * 128u;
#pragma unroll 1
      for (int x0 = 0; x0 < rows; x0 += 16) {
        uint32_t v0[8], v1[8];
        tmem_ld_16x256b_x2_pair(lb + uint32_t(x0), lb + (16u << 16) + uint32_t(x0), v0, v1);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int x = x0 + 8 * h + tx + c;
            if (x < rows) {
#if MUSE_VR_LDW_LANDED
              const float s_tok = m < head_n ? s.hpts[q][x] : p.pts_perm[qrow + x];
#else
              const float s_tok = m < head_n ? p.pts[s.ptok[m][x]] : p.pts_perm[qrow + x];
#endif
              uint32_t y0 = swiglu_out(__uint_as_float(v0[4 * h + 2 + c]), __uint_as_float(v0[4 * h + c]), s_tok, a_gate, a_up);
              uint32_t y1 = swiglu_out(__uint_as_float(v1[4 * h + 2 + c]), __uint_as_float(v1[4 * h + c]), s_tok, a_gate, a_up);
              y0 = (y0 & 0x7FFFu) > 0x7F80u ? 0x7FFFu : y0;  // canonical NaN
              y1 = (y1 & 0x7FFFu) > 0x7F80u ? 0x7FFFu : y1;
              asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(dst + size_t(x) * (kActWords * 2)), "r"(y0 | (y1 << 16))
                           : "memory");
            }
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      if (leader) {
        mbar_arrive(&s.tmem_empty[acc]);
        asm volatile("red.release.cta.shared::cta.add.u32 [%0], 1;" ::"r"(smem_u32(&s.fc1_stored[MUSE_VR_SLOT_UNSAFE ? 0 : q]))
                     : "memory");
        if (q == 0 && m + 1 == s.n_fc1) LDW_STAMP(p, kStEpiFc1);
      }
    } else {
      // FC2: output rows (stock epilogue arithmetic); 8 rows staged per warp, then one 16-byte store per lane
#if MUSE_VR_LDW_TEST_HB0_DELAY_NS > 0
      if (q == 1)
        for (int z = 0; z < MUSE_VR_LDW_TEST_HB0_DELAY_NS / 1000; ++z) __nanosleep(1000);  // tests only
#endif
      const float a_out = p.out_alpha[e];
      const uint32_t col = uint32_t((lane & 7) * 4 + (lane >> 3));
      const uint32_t st0 = smem_u32(&s.ostage[q][0][0]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(p.out) + (size_t(it.tile) * kTile + size_t(32 * q)) * 2;
#if MUSE_VR_LDW_TEST_RR_DELAY_NS > 0
      if (rr_first)  // tests only: rows and flag of this item late
        for (int z = 0; z < (MUSE_VR_LDW_TEST_RR_DELAY_NS + 999) / 1000; ++z) __nanosleep(1000);
#endif
#pragma unroll 1
      for (int x0 = 0; x0 < rows; x0 += 8) {
        uint32_t v[8];
        tmem_ld_x8(tacc + uint32_t(x0), v);
        const float ra = lane < 8 && x0 + lane < rows ? p.ramax[qrow + x0 + lane] : 0.0f;
#if MUSE_VR_LDW_EPI
        const float nw = p.norm_w[size_t(it.tile) * kTile + size_t(32 * q) + col];
        float sq[8];
#endif
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float pts2 = fmul_ftz(__shfl_sync(0xffffffffu, ra, i), gsi);
#if MUSE_VR_LDW_EPI
          const uint16_t yb = uint16_t(bf16_satfinite(fmul_ftz(fmul_ftz(__uint_as_float(v[i]), pts2), a_out)));
          sts_u16(st0 + uint32_t(i * 64) + col * 2u, yb);
          const float sy = __fmul_rn(nw, __uint_as_float(uint32_t(yb) << 16));
          sq[i] = __fmul_rn(sy, sy);
#else
          sts_u16(st0 + uint32_t(i * 64) + col * 2u,
                  uint16_t(bf16_satfinite(fmul_ftz(fmul_ftz(__uint_as_float(v[i]), pts2), a_out))));
#endif
        }
#if MUSE_VR_LDW_EPI
        // per row: the warp's 32 columns, butterfly in a fixed order; lane i keeps row i's partial
        float mine = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float a = sq[i];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
          if (lane == i) mine = a;
        }
        if (lane < 8 && x0 + lane < rows)
          p.moments[size_t(qrow + x0 + lane) * kMomParts + size_t(it.tile) * 4 + size_t(q)] = mine;
#endif
        __syncwarp();
        const int i = lane >> 2;  // row of this lane's 16 bytes
        if (x0 + i < rows)
          *reinterpret_cast<uint4*>(dst + size_t(qrow + x0 + i) * kHidden * 2 + size_t(lane & 3) * 16) =
              lds_v4(st0 + uint32_t(i * 64) + uint32_t(lane & 3) * 16u);
        __syncwarp();
#if MUSE_VR_LDW_EPI
        // the warp's row stores and partials are done (__syncwarp): one release per row on its token's counter
        if (lane < 8 && x0 + lane < rows) {
          const int t = p.pos_slot[qrow + x0 + lane] / kTopK;
          asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.tok_cnt + size_t(s.set) * kMaxTokens + size_t(t))
                       : "memory");
        }
        __syncwarp();
#endif
      }
#if MUSE_VR_LDW_HB0_PUSH
      // the warp's output row stores of this FC2 item are done: one release for the push warps (CTA scope)
      __syncwarp();
      if (lane == 0)
        asm volatile("red.release.cta.shared::cta.add.u32 [%0], 1;" ::"r"(smem_u32(&s.fc2_stored[q])) : "memory");
#endif
#if MUSE_VR_LDW_ROWS_READY
      // this warp's 32 columns of every row of the item are stored (__syncwarp orders them before lane 0's release): the item's
      // flag for this warp = the launch epoch
      __syncwarp();
      {
        unsigned long long ep = s.rr_epoch;
#if MUSE_VR_LDW_TEST_RR_PLANT
        if (rr_first) ep = MUSE_VR_LDW_TEST_RR_PLANT == 1 ? ep - 1ull : ep + 1ull;  // tests only: a flag never to be accepted
#endif
#if MUSE_VR_LDW_TEST_RR_DELAY_NS || MUSE_VR_LDW_TEST_RR_PLANT
        rr_first = false;
#endif
        if (lane == 0)
          asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p.rr + kRrFlagBase +
                                                                     (size_t(it.q) * (kHidden / kTile) + size_t(it.tile)) * 4 + size_t(q)),
                       "l"(ep)
                       : "memory");
      }
#endif
      tc_fence_before();
      __syncwarp();
      if (leader) mbar_arrive(&s.tmem_empty[acc]);
    }
  }
  if (q == 0 && leader) LDW_STAMP(p, kStEpi);
}

// |max| over one 16-element block (two 16-byte halves of bf16)
__device__ __forceinline__ float block_absmax(ulonglong2 a, ulonglong2 b) {
  const uint32_t w[8] = {uint32_t(a.x), uint32_t(a.x >> 32), uint32_t(a.y), uint32_t(a.y >> 32),
                         uint32_t(b.x), uint32_t(b.x >> 32), uint32_t(b.y), uint32_t(b.y >> 32)};
  float t[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) t[i] = fmaxf(fabsf(__uint_as_float(w[i] << 16)), fabsf(__uint_as_float(w[i] & 0xFFFF0000u)));
#pragma unroll
  for (int st = 4; st > 0; st >>= 1)
#pragma unroll
    for (int i = 0; i < st; ++i) t[i] = fmaxf(t[i], t[i + st]);
  return t[0];
}
// FlashInfer's per-token NVFP4 quantization of one 16-element bf16 block: 16 e2m1 values + the block's e4m3 scale byte
__device__ __forceinline__ uint64_t quant16(ulonglong2 a, ulonglong2 b, float enc, float amax, uint8_t* sf) {
  using PV = tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 16>;
  const uint32_t w8[8] = {uint32_t(a.x), uint32_t(a.x >> 32), uint32_t(a.y), uint32_t(a.y >> 32),
                          uint32_t(b.x), uint32_t(b.x >> 32), uint32_t(b.y), uint32_t(b.y >> 32)};
  PV vec;
#pragma unroll
  for (int x = 0; x < 8; ++x)
    vec.elts[x] = __halves2bfloat162(__ushort_as_bfloat16(uint16_t(w8[x] & 0xFFFFu)), __ushort_as_bfloat16(uint16_t(w8[x] >> 16)));
  return tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false, false, std::false_type>(vec, enc, sf, amax);
}

#if MUSE_VR_LDW_EPI
// Builder warps after their FC2-operand rows: token j of this CTA's list is finalized by warp j mod 4. Lane 0 acquires the
// token's counter (kMomParts per local route; bounded); the route's position and top-k weight come from the copy phase's map,
// which every FC2 epilogue whose release this acquire reads had acquired (rows-ready) before its stores. Per route, lane l sums
// partials 4 l .. 4 l + 3 in order, then a butterfly: ms, inv = rsqrt(ms / H + eps). Lane l then forms 8 columns at a time:
// sum over the local routes in slot order of w_k x ((nw y) x inv_k), fp32, one rounding to bf16.
__device__ __noinline__ void epi_tokens(Smem& s, const Params& p, int w, int lane) {
  const unsigned* const cnt = p.tok_cnt + size_t(s.set) * kMaxTokens;
#pragma unroll 1
  for (int j = w; j < s.own_n; j += 4) {
    const int t = s.own_t[j], r = s.own_r[j];
    uint4* const dst = reinterpret_cast<uint4*>(p.tok_out + size_t(t) * kHidden);
    if (r == 0 || aborted()) {
#pragma unroll 1
      for (int c = lane; c < kHidden / 8; c += 32) dst[c] = make_uint4(0u, 0u, 0u, 0u);
      continue;
    }
    if (lane == 0) acquire_at_least(cnt + t, unsigned(kMomParts * r), p.err);
    __syncwarp();  // memory ordering among the warp's threads: their reads follow lane 0's acquire
    int pos[kTopK];
    float wk[kTopK], inv[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      pos[k] = p.e2p[t * kTopK + k];
      wk[k] = __uint_as_float(uint32_t(p.expert_weights[t * kTopK + k]) << 16);
      inv[k] = 0.0f;
      if (pos[k] >= 0) {
        const float4 m4 = reinterpret_cast<const float4*>(p.moments + size_t(pos[k]) * kMomParts)[lane];
        float a = __fadd_rn(__fadd_rn(__fadd_rn(m4.x, m4.y), m4.z), m4.w);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
        inv[k] = rsqrtf(__fadd_rn(__fdiv_rn(a, float(kHidden)), p.norm_eps));
      }
    }
#pragma unroll 1
    for (int c = lane; c < kHidden / 8; c += 32) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const float4 n0 = reinterpret_cast<const float4*>(p.norm_w)[2 * c], n1 = reinterpret_cast<const float4*>(p.norm_w)[2 * c + 1];
      const float nw8[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        if (pos[k] < 0) continue;
        const uint4 y = reinterpret_cast<const uint4*>(p.out + size_t(pos[k]) * kHidden)[c];
        const uint32_t yw[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float yv = __uint_as_float(e & 1 ? (yw[e >> 1] & 0xFFFF0000u) : (yw[e >> 1] << 16));
          acc[e] = __fadd_rn(acc[e], __fmul_rn(wk[k], __fmul_rn(__fmul_rn(nw8[e], yv), inv[k])));
        }
      }
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[e] = uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * e]))) |
               (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * e + 1]))) << 16);
      dst[c] = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}
#endif

// warps 8-11: FC2 token rows. The rows this CTA builds (for every FC2 tile c of pass q it holds: rows c, c + 32, c + 64,
// c + 96 that exist) are dealt round-robin over the four warps in item order; each waits for the pass's FC1 tiles to be
// published (FC1 never waits on FC2, so the waits are acyclic), loads the row once, quantizes it with FlashInfer's
// per-token recipe, stores the FP4 row, its token-scale image entries and its row maximum, and releases one count.
#if MUSE_VR_LDW_HB0_PUSH
// Builder warps after their FC2-operand rows: the expert-output hop's push (see MUSE_VR_LDW_HB0_PUSH). Lane 0 of each warp
// acquires the grid's rows-ready count (every CTA's slot map entries), warp 0 the ring slot's credits, then a barrier of
// the four warps. Per FC2 item in list order: lane 0 of each warp waits until every epilogue warp released the item (CTA
// scope, bounded), then the 128 threads store the tile's 22 groups of each row of the pass, one word pair per (group,
// destination). Word pairs are stored with st.relaxed.sys.v2.b64 (two tagged 64-bit words, each single-copy atomic).
__device__ __noinline__ void hb0_push(Smem& s, const Params& p, int w, int lane) {
  namespace lh = muse_vr::lb_hop;
  namespace hb = muse_vr::hb0;
  constexpr int kHW = MUSE_VR_LDW_HB0_PUSH;
  const int tid = 32 * w + lane;
  const lh::Xfer x = lh::xfer<kHW>(p.hb);
  if (lane == 0) acquire_at_least(p.cnt + s.set * kCntSet + kMetaReady, gridDim.x, p.err);
  __syncwarp();  // memory ordering among the warp's threads: their slot-map reads follow lane 0's acquire
  if (w == 0 && !lh::credit_acquire<kHW>(p.hb, p.hbc, x) && lane == 0) fail(p.err, kErrPoll);
  named_bar_sync(MUSE_VR_LDW_HB0_BAR, 128);  // the credits before any store into the slot
  const uint32_t me = p.hb.rank;
  uint32_t done = 0;
#pragma unroll 1
  for (int m = s.n_fc1; m < s.n_items; ++m) {
    ++done;
#if MUSE_VR_LDW_TEST_HB0_PARTIAL == 1
    if (2 * (m - s.n_fc1) >= s.n_items - s.n_fc1) {  // tests only: the second half of the items is never stored
      if (tid == 0) fail(p.err, kErrPoll);
      return;
    }
#endif
    if (lane == 0 && !MUSE_VR_LDW_TEST_HB0_SKIP_ACQ) {
      uint32_t polls = 0;
      bool ok = true;
      for (;;) {
        uint32_t lo = 0xFFFFFFFFu;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          uint32_t d;
          asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(d) : "r"(smem_u32(&s.fc2_stored[q])) : "memory");
          lo = d < lo ? d : lo;
        }
        if (lo >= done) break;
        if (aborted() || ++polls > kMaxPolls) {
          ok = false;
          break;
        }
        __nanosleep(64);
      }
      if (!ok && !aborted()) fail(p.err, kErrPoll);
    }
    __syncwarp();  // memory ordering among the warp's threads: their row reads follow lane 0's acquire
    if (aborted()) return;
    const Item it = item_unpack(s.items[m]);
    const int rows = s.q_rows[it.q], qrow = s.q_row[it.q];
    const uint32_t g0 = uint32_t(it.tile) * hb::kTileGroups;
#pragma unroll 1
    for (int u = tid; u < rows * int(hb::kTileGroups); u += 128) {
      const int r = u / int(hb::kTileGroups);
      const uint32_t gi = g0 + uint32_t(u - r * int(hb::kTileGroups));
#if MUSE_VR_LDW_TEST_HB0_PARTIAL == 2
      if (gi - g0 >= hb::kTileGroups / 2u) continue;  // tests only: the second half of each tile's groups is never stored
#endif
      const int pos = qrow + r;
      const uint32_t lin = uint32_t(p.hb_slot[pos]);
      uint32_t e[6];
      hb::load6(reinterpret_cast<const uint16_t*>(p.out) + size_t(pos) * kHidden, gi, e);
      unsigned long long w0, w1;
      lh::words6(x, e, w0, w1);
#pragma unroll
      for (int d = 0; d < kHW; ++d)
        if (uint32_t(d) != me) lh::st_relaxed_sys_v2_b64(hb::lin_word(p.hb, uint32_t(d), x.copy, lin, hb::group_word(gi)), w0, w1);
    }
  }
}
#endif

__device__ MUSE_VR_LDW_ROLE void role_builder(Smem& s, const Params& p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int w = warp & 3;
  const float gsi = p.gsi;
  unsigned* const cset = p.cnt + s.set * kCntSet;
#if MUSE_VR_LDW_LANDED
  landed_copy_rows(s, p, threadIdx.x - 8 * 32);
#elif !MUSE_VR_LDW_EARLY_COPY  // EARLY_COPY: the kernel body ran the copy phase before the list barrier
  LDW_COPY_ROWS(s, p, cset, threadIdx.x - 4 * 32);
#endif
  if (p.dbg & 128) return;
#if MUSE_VR_LDW_BALB
  const bool bal_ok = bal_acquire(s, p);  // already published (the copy phase's second part waited for it)
  const int n_items = bal_ok ? s.bal_n_items : s.n_items, n_fc1 = bal_ok ? s.bal_n_fc1 : s.n_fc1;
#else
  const int n_items = s.n_items, n_fc1 = s.n_fc1;
#endif
  int duty = 0;
#pragma unroll 1
  for (int m = n_fc1; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int rows = s.q_rows[it.q], qrow = s.q_row[it.q];
#pragma unroll 1
    for (int i = it.tile % kFc2TilesPerExpert; i < rows; i += 32, ++duty) {
      if ((duty & 3) != w) continue;  // warp-uniform
      if (lane == 0) {
        const unsigned* c1 = cset + it.q;
        uint32_t polls = 0;
        bool ok = true;
        while (muse_vr_ldq::ld_relaxed_gpu(c1) < unsigned(kFc1Arrivals - 16)) {
          if (aborted() || ++polls > kMaxPolls) {
            ok = false;
            break;
          }
          __nanosleep(64);
        }
        while (ok && ld_acquire_gpu_u32(c1) < unsigned(kFc1Arrivals)) {
          if (aborted() || ++polls > kMaxPolls) {
            ok = false;
            break;
          }
          __nanosleep(32);
        }
        if (!ok && !aborted()) fail(p.err, kErrPoll);
      }
      __syncwarp();  // memory ordering among the warp's threads: the row loads follow lane 0's acquire
      const unsigned long long* row = p.act + size_t(qrow + i) * kActWords;
      // row maximum with MoE-LD's reduction structure (its builder warp w' covers blocks 64 w' .. 64 w' + 63 = this
      // lane's blocks lane + 64 w' and lane + 64 w' + 32): per quarter an integer warp max of fmaxf over the lane's two
      // blocks (|x| >= 0: the bit pattern orders like the value), then fmaxf over the quarters. All loads first.
      float bm[8];
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const unsigned long long* r = row + size_t(lane + 32 * b) * 4;
        bm[b] = block_absmax(ld_ca_v2u64(r), ld_ca_v2u64(r + 2));
      }
      float qmax[4];
#pragma unroll
      for (int qq = 0; qq < 4; ++qq)
        qmax[qq] = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(fmaxf(bm[2 * qq], bm[2 * qq + 1]))));
      const float amax = fmaxf(fmaxf(qmax[0], qmax[1]), fmaxf(qmax[2], qmax[3]));
      const float pts2 = fmul_ftz(amax, gsi);
      const float enc = pts2 != 0.0f ? tensorrt_llm::kernels::reciprocal_approximate_ftz(pts2) : 0.0f;
      uint8_t* frow = p.hs2 + size_t(int(s.q_row2[it.q]) + i) * (kHidden / 2);
      uint8_t* img = p.img2 + size_t(it.q) * kImgBytes + size_t(16 * (i & 31) + 4 * (i >> 5));
      // quantization pass kept rolled (hot-code size): the row's blocks are reloaded (first-level cache hits)
#pragma unroll 1
      for (int b = 0; b < 8; ++b) {
        const int blk = lane + 32 * b;
        const unsigned long long* r = row + size_t(blk) * 4;
        uint8_t sf8;
        const uint64_t q8 = quant16(ld_ca_v2u64(r), ld_ca_v2u64(r + 2), enc, amax, &sf8);
        *reinterpret_cast<uint64_t*>(frow + size_t(blk) * 8) = q8;
        // the four scale bytes of K chunk blk / 4 sit on lanes 4 k .. 4 k + 3: lane 4 k assembles the image word
        const uint32_t v = sf8;
        const uint32_t v1 = __shfl_down_sync(0xffffffffu, v, 1), v2 = __shfl_down_sync(0xffffffffu, v, 2),
                       v3 = __shfl_down_sync(0xffffffffu, v, 3);
        if ((lane & 3) == 0)
          *reinterpret_cast<uint32_t*>(img + size_t(blk >> 2) * 512) = v | (v1 << 8) | (v2 << 16) | (v3 << 24);
      }
      if (lane == 0) p.ramax[qrow + i] = amax;
      __syncwarp();  // the warp's stores before lane 0's release
      if (lane == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cset + kMaxPasses + it.q) : "memory");
    }
  }
  if (w == 0 && lane == 0) LDW_STAMP(p, kStBuild);
#if MUSE_VR_LDW_EPI
  epi_tokens(s, p, w, lane);
#endif
#if MUSE_VR_LDW_HB0_PUSH
  hb0_push(s, p, w, lane);
#endif
}

// warp 3 after the prologue: FC1 publication per tile, residency exit wait
#if MUSE_VR_LDW_BALB
// Warp 3 right after barrier 3 (idle until the first FC1 tile is stored): reads the board (each word accepted on its own
// matching 64-bit load of this step's ids tag; bounded; same-GPU producer), checks every code (0xF or one of the expert's two
// holders) and the digest (the producer's form), then appends this rank's assigned candidate slots as further passes (bases
// after the pinned rows, passes after the pinned passes; home-replicated experts in ascending id, then replicas in ascending
// id), this CTA's FC1 items of those passes and its FC2 items of all passes (the prologue's deal with the final pass count; the
// pinned FC1 prefix is unchanged), and the positions of this CTA's candidate slots. All of it is written before one release of
// bal_ready (CTA scope); readers acquire it before they read any of it. A failed read or check sets the error word and
// publishes the pinned part only (the CTA is aborted, so the launch fails loudly).
__device__ __noinline__ void bal_board_phase(Smem& s, const Params& p) {
  const int lane = int(threadIdx.x & 31u);
  const uint32_t rk = uint32_t(p.bal_rank), tag = s.ids_tag;
  bool good = s.land_go != 0;
  LDW_TEST_BOARD_WAIT(s, p);  // tests only (a no-op unless the test header is included)
  // codes: lane l takes word l (experts 8l .. 8l + 7), lane 0 also the digest word
  unsigned long long wv = 0ull, dv = 0ull;
  if (good) {
    uint32_t polls = 0;
    bool mine = false, dmine = lane != 0;
    for (;;) {
      if (!mine) {
        wv = ld_relaxed_u64(p.board + lane);
        mine = uint32_t(wv >> 32) == tag;
      }
      if (!dmine) {
        dv = ld_relaxed_u64(p.board + 32);
        dmine = uint32_t(dv >> 32) == tag;
      }
      if (__all_sync(0xffffffffu, mine && dmine)) break;
      if (aborted() || ++polls > kMaxPolls) {
        good = false;
        break;
      }
      __nanosleep(32);
    }
    good = __all_sync(0xffffffffu, good);
  }
  uint32_t h = 0u;
  bool codes_ok = true;
  if (good) {
    const uint32_t payload = uint32_t(wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t e = uint32_t(lane) * 8u + uint32_t(j), code = (payload >> (4 * j)) & 0xFu;
      const uint32_t t = s.btab[e], hh = t & 0xFu, sc = (t >> 4) & 0xFu;
      codes_ok = codes_ok && (code == kNoHolder || code == hh || (sc != kNoHolder && code == sc));
      s.bcode[e] = uint8_t(code);
      const uint32_t y = ((e << 8) | code) * 0x9e3779b1u;
      h += y ^ (y >> 16);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) h += __shfl_xor_sync(0xffffffffu, h, o);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    const uint32_t want = __shfl_sync(0xffffffffu, uint32_t(dv), 0);
    good = __all_sync(0xffffffffu, codes_ok) && h == want;
  }
  if (!good && lane == 0) fail(p.err, kErrBoard);
  __syncwarp();  // every lane's code stores before any lane reads another lane's codes
  int rows = s.bal_rows_p, q = s.bal_nq_p;
  if (good) {
#pragma unroll 1
    for (int ph = 0; ph < 2; ++ph) {
#pragma unroll 1
      for (int c = 0; c < 256; c += 32) {
        const uint32_t e = uint32_t(c + lane);
        const uint32_t t = s.btab[e], hh = t & 0xFu, sc = (t >> 4) & 0xFu;
        const bool sel = sc != kNoHolder && uint32_t(s.bcode[e]) == rk && (ph == 0 ? hh == rk : sc == rk);
        const int ls = ph == 0 ? int((t >> 8) & 0xFFu) : int((t >> 16) & 0xFFu);
        const int n = (sel && ls < p.pool) ? int(s.cand_n[ls]) : 0;
        const int np = (n + kPassRows - 1) / kPassRows;
        int incl = n, inclq = np;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int y = __shfl_up_sync(0xffffffffu, incl, o), yq = __shfl_up_sync(0xffffffffu, inclq, o);
          if (lane >= o) {
            incl += y;
            inclq += yq;
          }
        }
        if (sel && ls < p.pool) {
          const int b = rows + incl - n, qb = q + inclq - np;
          s.base[ls] = int16_t(b);
          s.qbase[ls] = int16_t(qb);
          for (int pq = 0; pq < np; ++pq) {
            if (qb + pq < kMaxPasses) {
              s.q_expert[qb + pq] = int16_t(ls);
              s.q_row[qb + pq] = int16_t(b + pq * kPassRows);
              s.q_rows[qb + pq] = int16_t(min(kPassRows, n - pq * kPassRows));
            }
          }
        }
        rows += __shfl_sync(0xffffffffu, incl, 31);
        q += __shfl_sync(0xffffffffu, inclq, 31);
      }
    }
    __syncwarp();  // every lane's q_rows stores before lane 0 reads them
    if (lane == 0) {  // bucket-padded FC2 row bases of the new passes (the pinned ones are unchanged)
      int acc = s.bal_nq_p > 0 ? int(s.q_row2[s.bal_nq_p - 1]) + bucket_rows(bucket_of(int(s.q_rows[s.bal_nq_p - 1]))) : 0;
#pragma unroll 1
      for (int qq = s.bal_nq_p; qq < q && qq < kMaxPasses; ++qq) {
        s.q_row2[qq] = int16_t(acc);
        acc += bucket_rows(bucket_of(int(s.q_rows[qq])));
      }
      if (acc > kHs2Rows) fail(p.err, kErrShape);
    }
  }
  // the deal of the prologue's work list with the final pass count (the pinned FC1 prefix is unchanged)
  const unsigned slot = s.slot;
  int nq = q;
  if (nq > kMaxPasses) {
    if (lane == 0) fail(p.err, kErrShape);
    nq = 0;
  }
  const unsigned dom = (slot >> 7) & 1u;
  const int R = dom ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
  const int n1 = R > 0 ? steps_of(kFc1TilesPerDomain * nq, r, R) : 0;
  const int n2 = R > 0 ? steps_of((kFc1TilesPerDomain + kFc2TilesPerDomain) * nq, r, R) - n1 : 0;
  int nt = (p.dbg & 512) ? 0 : n1 + n2;
  if (nt > kMaxItems) {
    if (lane == 0) fail(p.err, kErrSlot);
    nt = 0;
  }
  const int first = s.bal_n1_p;
#pragma unroll 1
  for (int i = first + lane; i < nt; i += 32) {
    uint32_t v;
    if (i < n1) {
      const int g = r + i * R;
      v = item_pack(0, g / kFc1TilesPerDomain, kFc1TilesPerDomain * int(dom) + (g % kFc1TilesPerDomain));
    } else {
      const int pc = r + i * R - kFc1TilesPerDomain * nq;
      v = item_pack(1, pc / kFc2TilesPerDomain, kFc2TilesPerDomain * int(dom) + (pc % kFc2TilesPerDomain));
    }
    s.items[i] = v;
  }
  // positions of this CTA's candidate slots: assigned here -> base of the slot + the slot's rank; else none
#pragma unroll 1
  for (int j = lane; j < s.n_my; j += 32) {
    if (s.my_cand[j]) {
      const int ls = s.my_exp[j];
      const int ge = ls >= 0 ? int(s.slot_exp[ls]) : -1;
      const bool here = good && ge >= 0 && uint32_t(s.bcode[ge]) == rk;
      s.my_pos[j] = int16_t(here ? int(s.base[ls]) + int(s.my_rank[j]) : -1);
      s.my_exp[j] = int16_t(here ? ls : -1);
    }
  }
  __syncwarp();
  if (lane == 0) {
    s.bal_n_pass = nq;
    s.bal_n_fc1 = nt > 0 ? n1 : 0;
    s.bal_n_items = nt;
  }
  __syncwarp();  // every lane's stores above, before lane 0's release
  if (lane == 0) asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" ::"r"(smem_u32(&s.bal_ready)), "r"(1u) : "memory");
  if (lane == 0) LDW_STAMP(p, kStBoard);
  __syncwarp();
}
#endif

#if MUSE_VR_LDW_THEAD
// Warp 3 before its FC1 publication (THEAD): the head item's token-scale images, stage by stage (the copy phase's image layout:
// chunk c at c * 512, row i at 16 (i % 32) + 4 (i / 32), 4 bytes per row and K chunk), each stage after its slot's previous
// use was consumed (empty barrier, then the write-after-read proxy fence), then a proxy fence and one arrival on its full
// barrier. The head item is item 0: its stages are the launch's stages 0 .. kStagesPerTile - 1.
__device__ __noinline__ void thead_images(Smem& s, const Params& p, int lane) {
  const Item it = item_unpack(s.items[0]);
  const int rows = s.q_rows[it.q];
  const int16_t* rt = s.ptok[0];
#pragma unroll 1
  for (int j = 0; j < kStagesPerTile; ++j) {
    const int pos = j % kStages;
    if (j >= kStages) {
      __syncwarp();
      mbar_wait(&s.empty[pos], (uint32_t(j / kStages) & 1u) ^ 1u, p.err);
      fence_proxy_async_smem();
    }
    if (aborted()) {
      // abort: this warp's arrival on the current stage (whose phase is this stage's: first use, or its empty wait above), so
      // its drain completes; later head stages' drains end on their poll bounds
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.full[pos]);
      return;
    }
    const uint32_t im = smem_u32(&s.st[pos][kStageA + kStageB]);
#pragma unroll 1
    for (int x = lane; x < rows * kChunks; x += 32) {
      const int i = x >> 3, c = x & 7;
      const uint32_t v = *reinterpret_cast<const uint32_t*>(p.hs_sf + size_t(rt[i]) * (kHidden / 16) + size_t(j) * 32 + 4 * c);
      asm volatile("st.shared.b32 [%0], %1;" ::"r"(im + uint32_t(c) * 512u + 16u * uint32_t(i & 31) + 4u * uint32_t(i >> 5)), "r"(v)
                   : "memory");
    }
    fence_proxy_async_smem();  // this lane's generic stores before the asynchronous-proxy reads behind the barrier
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.full[pos]);
  }
}
#endif

__device__ MUSE_VR_LDW_ROLE void role_ctrl(Smem& s, const Params& p) {
  const int lane = threadIdx.x & 31;
  unsigned* const cset = p.cnt + s.set * kCntSet;
#if MUSE_VR_LDW_THEAD
  if (s.head_n > 0 && (p.dbg & 256) == 0) thead_images(s, p, lane);
#endif
#if MUSE_VR_LDW_BALB
  bal_board_phase(s, p);
#endif
  if ((p.dbg & 128) == 0 && lane == 0) {
#pragma unroll 1
#if MUSE_VR_LDW_BALB
    for (int mm = 0; mm < s.bal_n_fc1; ++mm) {  // this warp wrote the final count (bal_board_phase)
#else
    for (int mm = 0; mm < s.n_fc1; ++mm) {
#endif
      uint32_t polls = 0, done = 0;
      bool ok = true;
      while (true) {
        #if MUSE_VR_SLOT_UNSAFE
        // previous form: one CTA-wide count (assumes the four epilogue warps finish FC1 tiles in step)
        asm volatile("ld.relaxed.cta.shared::cta.u32 %0, [%1];" : "=r"(done) : "r"(smem_u32(&s.fc1_stored[0])) : "memory");
        if (done >= uint32_t(4 * (mm + 1))) break;
        #else
        // tile mm is stored once each of the four epilogue warps counted it (exact whatever their skew; the fence below
        // then acquires all four warps' releases)
        done = 0xFFFFFFFFu;
        #pragma unroll
        for (int w4 = 0; w4 < 4; ++w4) {
          uint32_t d;
          asm volatile("ld.relaxed.cta.shared::cta.u32 %0, [%1];" : "=r"(d) : "r"(smem_u32(&s.fc1_stored[w4])) : "memory");
          done = d < done ? d : done;
        }
        if (done >= uint32_t(mm + 1)) break;
        #endif
        if (aborted()) {
          ok = false;
          break;
        }
        if (++polls > kMaxPolls) {
          fail(p.err, kErrMbar);
          ok = false;
          break;
        }
        __nanosleep(64);
      }
      if (!ok) break;
      asm volatile("fence.acq_rel.cta;" ::: "memory");  // acquire pattern (cta)
      const Item it = item_unpack(s.items[mm]);
      asm volatile("red.release.gpu.global.add.u32 [%0], 4;" ::"l"(cset + it.q) : "memory");
    }
  }
  __syncwarp();
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  // this CTA's claim (prologue, before named barrier 3) is complete: the last CTA checks and clears the claim bitmap
  if (lane == 0) muse_vr_slot::finish(p.ticket + kClaimWord, p.err);
#endif
  if (lane == 0 && !aborted()) {
    const unsigned cycle = gridDim.x * kStartCycle;
    uint32_t polls = 0;
    for (;;) {
      const unsigned v = muse_vr_ldq::ld_relaxed_gpu(p.ticket);
      const unsigned d = v >= s.start_base ? v - s.start_base : v + cycle - s.start_base;
      if (d >= gridDim.x) break;
      if (aborted()) break;
      if (++polls > kMaxPolls) {
        fail(p.err, kErrStart);
        break;
      }
      __nanosleep(256);
    }
  }
  __syncwarp();
}

// Prologue, every thread of the CTA: counting the local routed slots per expert (per 256-slot chunk), expert bases and
// passes, this CTA's work list (warp 0) and the permuted positions of the slots whose rows this CTA copies (warps 8-11).
// The scratch lives in the last ring stage, which no copy targets before the work list is published (named barrier 3).
__device__ MUSE_VR_LDW_ROLE void prologue(Smem& s, const Params& p) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  Scratch& x = *reinterpret_cast<Scratch*>(&s.st[kStages - 1][0]);
  const int nslot = p.m * kTopK;
  // Start ticket (residency, counter parity) and slot claim: both are atomic RMWs on control words that the previous
  // launch of this call site updates (its last CTA clears the claim bitmap in finish()). They are issued right AFTER
  // griddepcontrol.wait: under programmatic dependent launch the wait is the documented edge that orders them after the
  // completion of every prerequisite grid (PTX ISA, griddepcontrol), so the previous launch's clear is ordered before
  // this launch's claim. Issued before the routed-ids loads below, so their latency still overlaps those loads and the
  // ccnt zeroing. -DMUSE_VR_LDW_TEST_PREWAIT_CLAIM=1 (tests only: the negative control of tests/test_moe_ldw_order.py)
  // keeps the earlier pre-wait placement.
  unsigned ticket = 0;
#if MUSE_VR_LDW_ROWS_READY
  unsigned long long rr_old = 0;
#endif
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
  // launch-counter claim (slot_claim.cuh): issued by thread 0 after its routed-ids loads below (an RMW issued at
  // launch delays those loads), settled when the work list needs it
  unsigned ep_smid = 0, ep_old = 0, ep_hint = 0;
#endif
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  unsigned claim_hint = 0, claim_old = 0;
#endif
#if MUSE_VR_LDW_LANDED
  // this forward's sequence word (raised by a normally launched operation before the forward's first kernel, so it may
  // be read before the dependency wait, as the hop API's consumers do), the copies and tags it selects, and the
  // missed-advance guard: a CTA that sees the word of its previous consume consumes nothing and sets the error word
  // (shared-typed stores only before the dependency wait; the error word is written after it)
  if (tid == 0) {
    const unsigned long long seq = *reinterpret_cast<const volatile unsigned long long*>(p.seq64);
    sts_u64(smem_u32(&s.land_seq), seq);
    sts_u32(smem_u32(&s.ids_tag), uint32_t(seq % 0xFFFFFFFFull) + 1u);
    sts_u32(smem_u32(&s.tok_tag), uint32_t(seq % 65535ull) + 1u);
    sts_u32(smem_u32(&s.ids_copy), p.ids_parity ? uint32_t(seq & 1ull) : 0u);
    sts_u32(smem_u32(&s.tok_copy), p.tok_parity ? uint32_t(seq & 1ull) : 0u);
    const bool go = *reinterpret_cast<const volatile unsigned long long*>(p.last_seq + blockIdx.x) != seq;
    sts_u32(smem_u32(&s.land_go), go ? 1u : 0u);
  }
#if MUSE_VR_LDW_BALB
  // the static placement table (written once by the host, not by a predecessor): staged before the dependency wait
  if (warp == 1)
    for (int i = lane; i < 256; i += 32) sts_u32(smem_u32(&s.btab[i]), p.bal_table[i]);
  if (tid == 0) sts_u32(smem_u32(&s.bal_ready), 0u);
#endif
#endif
#if MUSE_VR_LDW_TEST_PREWAIT_CLAIM
  if (tid == 0) ticket = atomicInc(p.ticket, gridDim.x * kStartCycle - 1u);
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  if (tid == 0) {
    claim_hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
    claim_old = muse_vr_slot::try_hint(p.ticket + kClaimWord, claim_hint, p.n_dom0, p.n_dom1);
  }
#endif
#endif
  grid_dep_wait();  // ids and token rows come from predecessors
#if !MUSE_VR_LDW_LANDED
  if (tid == 0) LDW_STAMP(p, kStWait);
#endif
#if MUSE_VR_LDW_LANDED
  if (tid == 0) {
    LDW_STAMP(p, kStWait);
#if MUSE_VR_LDW_BALB
    LDW_TEST_T0(s);
#endif
    if (s.land_go == 0) fail(p.err, kErrLand);  // missed advance (guard read above): nothing is consumed
  }
#endif
#if !MUSE_VR_LDW_TEST_PREWAIT_CLAIM
  // start ticket: its value is first needed at the parity step below
  if (tid == 0) ticket = atomicInc(p.ticket, gridDim.x * kStartCycle - 1u);
#if MUSE_VR_LDW_ROWS_READY
  if (tid == 0) rr_old = atomicAdd(p.rr, 1ull);  // the epoch ticket: its value is first needed with the parity below
#endif
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  // slot claim (slot_claim.cuh): the SM's slot is the hint; settled when the work list needs it
  if (tid == 0) {
    claim_hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
    claim_old = muse_vr_slot::try_hint(p.ticket + kClaimWord, claim_hint, p.n_dom0, p.n_dom1);
  }
#endif
#endif
#if MUSE_VR_LDW_PRO_COMPACT
  const int nch_used = (nslot + kChunkSlots - 1) / kChunkSlots;  // chunks beyond it are never read
  for (int i = tid; i < nch_used * kMaxPool; i += kThreads) (&x.ccnt[0][0])[i] = 0u;
#else
  for (int i = tid; i < kNumChunks * kMaxPool; i += kThreads) (&x.ccnt[0][0])[i] = 0u;
#endif
#if MUSE_VR_LDW_BALB
  for (int i = tid; i < kMaxPool; i += kThreads) {
    s.cand[i] = 0;
    s.slot_exp[i] = -1;
    s.cand_n[i] = 0;
  }
#endif
  named_bar_sync(6, kThreads);
#if MUSE_VR_LDW_PRO_COMPACT
  {
    // this thread's ids in groups of kIdsGroup words in flight (a rolled loop: one group up to 384 rows), then their counts;
    // a CTA whose guard failed reads none and keeps no local slot
    constexpr int kIdsGroup = 8;
#if MUSE_VR_LDW_LANDED
    const uint32_t tag = s.ids_tag;
    const bool go = s.land_go != 0;
#else
    const bool go = true;
#endif
    const int nper = (go && nslot > tid) ? (nslot - 1 - tid) / kThreads + 1 : 0;
#pragma unroll 1
    for (int u0 = 0; u0 < nper; u0 += kIdsGroup) {
#if MUSE_VR_LDW_LANDED
      unsigned long long w[kIdsGroup];
#pragma unroll
      for (int b = 0; b < kIdsGroup; ++b)
        w[b] = u0 + b < nper ? ld_relaxed_sys_u64(ids_word(s, p, tid + (u0 + b) * kThreads)) : 0ull;
#else
      int32_t w[kIdsGroup];
#pragma unroll
      for (int b = 0; b < kIdsGroup; ++b) w[b] = u0 + b < nper ? p.packed[tid + (u0 + b) * kThreads] : -1;
#endif
#pragma unroll
      for (int b = 0; b < kIdsGroup; ++b) {
        if (u0 + b < nper) {
          const int sl = tid + (u0 + b) * kThreads;
#if MUSE_VR_LDW_LANDED
          const int32_t v = int32_t(uint32_t(land_accept(w[b], ids_word(s, p, sl), tag, kIdsTagShift, p.err)));
#else
          const int32_t v = w[b];
#endif
          const int e = int((v >> 16) & 0xFFFF) - p.local_offset;
          const bool loc = e >= 0 && e < p.pool;
          x.exp[sl] = int16_t(loc ? e : -1);
          if (loc) atomicAdd(&x.ccnt[sl / kChunkSlots][e], 1u);
        }
      }
    }
    if (!go)
      for (int sl = tid; sl < nslot; sl += kThreads) x.exp[sl] = -1;
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    if (tid == 0) {
      const uint8_t* const table = reinterpret_cast<const uint8_t*>(p.sm_domain);
      ep_smid = muse_vr_slot::epoch_test_smid(table, muse_vr_slot::read_smid());
      ep_old = muse_vr_slot::epoch_try_sm(p.ticket + 32, ep_smid, kStartCycle);  // counters: words 32.. of ctl
      ep_hint = ep_smid < gridDim.x ? unsigned(table[ep_smid]) : muse_vr_slot::kInvalid;
    }
#endif
  }
#else
  {
    // all of this thread's ids in flight at once, then the counts
    constexpr int kPer = (kMaxSlots + kThreads - 1) / kThreads;
    int32_t v[kPer];
#if MUSE_VR_LDW_LANDED
    // the routed ids from the landing (s.ids_tag / s.ids_copy were set before barrier 6); a CTA whose guard failed reads
    // none and keeps no local slot
    {
      const uint32_t tag = s.ids_tag;
      const bool go = s.land_go != 0;
      unsigned long long w[kPer];
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        const int sl = tid + u * kThreads;
        w[u] = (go && sl < nslot) ? ld_relaxed_sys_u64(ids_word(s, p, sl)) : 0ull;
      }
#pragma unroll
      for (int u = 0; u < kPer; ++u) {
        const int sl = tid + u * kThreads;
        v[u] = (go && sl < nslot) ? int32_t(uint32_t(land_accept(w[u], ids_word(s, p, sl), tag, kIdsTagShift, p.err))) : -1;
      }
    }
#else
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int sl = tid + u * kThreads;
      v[u] = sl < nslot ? p.packed[sl] : -1;
    }
#endif
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    if (tid == 0) {
      const uint8_t* const table = reinterpret_cast<const uint8_t*>(p.sm_domain);
      ep_smid = muse_vr_slot::epoch_test_smid(table, muse_vr_slot::read_smid());
      ep_old = muse_vr_slot::epoch_try_sm(p.ticket + 32, ep_smid, kStartCycle);  // counters: words 32.. of ctl
      ep_hint = ep_smid < gridDim.x ? unsigned(table[ep_smid]) : muse_vr_slot::kInvalid;
    }
#endif
#if MUSE_VR_LDW_BALB
    // the pool slot of each routed slot held here (placement table): pinned (one holder, this rank) or candidate (replicated,
    // this rank one of the two holders); both counted per pool slot. The pool slots of this rank's replicated experts are
    // marked candidates (their locality comes from the board), and each pool slot learns its expert.
    if (tid < 256) {
      const uint32_t t = s.btab[tid], h = t & 0xFu, sc = (t >> 4) & 0xFu, rk = uint32_t(p.bal_rank);
      const int hs = int((t >> 8) & 0xFFu), ss = int((t >> 16) & 0xFFu);
      if (h == rk && hs < p.pool) {
        s.slot_exp[hs] = int16_t(tid);
        if (sc != kNoHolder) s.cand[hs] = 1;
      }
      if (sc != kNoHolder && sc == rk && ss < p.pool) {
        s.slot_exp[ss] = int16_t(tid);
        s.cand[ss] = 1;
      }
    }
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int sl = tid + u * kThreads;
      if (sl < nslot) {
        const uint32_t e = (uint32_t(v[u]) >> 16) & 0xFFFFu;
        int le = -1;
        if (e < 256u) {
          const uint32_t t = s.btab[e], h = t & 0xFu, sc = (t >> 4) & 0xFu, rk = uint32_t(p.bal_rank);
          if (h == rk) le = int((t >> 8) & 0xFFu);
          else if (sc != kNoHolder && sc == rk) le = int((t >> 16) & 0xFFu);
        }
        const bool loc = le >= 0 && le < p.pool;
        x.exp[sl] = int16_t(loc ? le : -1);
        if (loc) atomicAdd(&x.ccnt[sl / kChunkSlots][le], 1u);
      }
    }
#else
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int sl = tid + u * kThreads;
      if (sl < nslot) {
        const int e = int((v[u] >> 16) & 0xFFFF) - p.local_offset;
        const bool loc = e >= 0 && e < p.pool;
        x.exp[sl] = int16_t(loc ? e : -1);
        if (loc) atomicAdd(&x.ccnt[sl / kChunkSlots][e], 1u);
      }
    }
#endif
  }
#endif
  if (tid == 0) LDW_STAMP(p, kStIdsOwn);  // timing builds: thread 0's ids are read and counted
  named_bar_sync(6, kThreads);
  if (tid == 0) LDW_STAMP(p, kStCnt);
  if (tid < kMaxPool) {  // per expert: exclusive prefix over chunks (in place) and the total
    uint32_t run = 0;
#if MUSE_VR_LDW_PRO_COMPACT
    for (int c = 0; c < nch_used; ++c) {
#else
    for (int c = 0; c < kNumChunks; ++c) {
#endif
      const uint32_t n = x.ccnt[c][tid];
      x.ccnt[c][tid] = run;
      run += n;
    }
    x.total[tid] = tid < p.pool ? run : 0u;
  }
  named_bar_sync(6, kThreads);
  if (tid == 0) LDW_STAMP(p, kStPfx);
#if MUSE_VR_LDW_EARLY_COPY
  // early copy, warps 4-11: their share of the pass barrier below (only warps 0-3 wait there), then the hand-off barrier, after
  // which warp 0's expert bases, pass bases and counter parity are visible (the slots' experts and the prefix counts read below
  // are ordered by the barrier just passed). No exit before either operation: every thread of every CTA issues both once.
  if (warp >= 4) {
    named_bar_arrive(6, kThreads);
#if MUSE_VR_LDW_TEST_EC_TWIN != 1 && MUSE_VR_LDW_TEST_EC_PLANT != 2
#if MUSE_VR_LDW_TEST_EC_COUNT
    atomicAdd(&s.ec_dbg[0], 1u);
#endif
    named_bar_sync(kEcgBar, kEcgThreads);
#elif MUSE_VR_LDW_TEST_EC_TWIN == 1
    ec_test_wait();  // tests only: the hand-off dropped (racecheck control)
#endif  // PLANT == 2 (tests only): the hand-off dropped and no wait (the hand-off control)
#if MUSE_VR_LDW_TEST_EC_PLANT == 1
    if (warp >= 8) {  // tests only: the ordering control (barrier 3 before the scratch reads of the positions below)
      named_bar_arrive(3, kThreads);
      ec_test_wait();
    }
#endif
  }
#endif
  if (warp == 0) {  // expert bases and passes (ascending expert order), pass tables
#if MUSE_VR_LDW_TEST_EC_PLANT == 2
    ec_test_wait();  // tests only: the publication below is delayed (the hand-off control; nothing waits for it)
#endif
    int rows = 0, q = 0;
#pragma unroll 1
    for (int c = 0; c < kMaxPool; c += 32) {
      const int e = c + lane;
#if MUSE_VR_LDW_BALB
      // pinned slots only; a candidate slot's rows are kept for the board phase
      const int n = s.cand[e] ? 0 : int(x.total[e]);
      s.cand_n[e] = int16_t(s.cand[e] ? int(x.total[e]) : 0);
#else
      const int n = int(x.total[e]);
#endif
      const int np = (n + kPassRows - 1) / kPassRows;
      int incl = n, inclq = np;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o), yq = __shfl_up_sync(0xffffffffu, inclq, o);
        if (lane >= o) {
          incl += y;
          inclq += yq;
        }
      }
      const int b = rows + incl - n, qb = q + inclq - np;
      s.base[e] = int16_t(b);
      s.qbase[e] = int16_t(qb);
      for (int pq = 0; pq < np; ++pq) {
        if (qb + pq < kMaxPasses) {
          s.q_expert[qb + pq] = int16_t(e);
          s.q_row[qb + pq] = int16_t(b + pq * kPassRows);
          s.q_rows[qb + pq] = int16_t(min(kPassRows, n - pq * kPassRows));
        }
      }
      rows += __shfl_sync(0xffffffffu, incl, 31);
      q += __shfl_sync(0xffffffffu, inclq, 31);
    }
    __syncwarp();  // every lane's q_rows stores before lane 0 reads them
#if MUSE_VR_LDW_EARLY_COPY
    // early copy: the counter parity (moved here from the lane-0 block below), then the hand-off. Each lane's arrival orders its
    // own base / qbase stores (lane 0: also start_base / set) before the waiters' reads: when a barrier completes, the prior memory
    // accesses of its participants are performed relative to all of them (PTX ISA barrier{.cta}). Unconditional: the expert loop
    // above has a fixed trip count and no exit, so warp 0 of every CTA (with or without rows or items) arrives exactly once.
    if (lane == 0) {
      s.start_base = ticket - ticket % gridDim.x;
      s.set = int((s.start_base / gridDim.x) & 1u);
    }
    __syncwarp();
#if MUSE_VR_LDW_TEST_EC_TWIN != 1 && MUSE_VR_LDW_TEST_EC_PLANT != 2
#if MUSE_VR_LDW_TEST_EC_COUNT
    atomicAdd(&s.ec_dbg[0], 1u << 16);
#endif
    named_bar_arrive(kEcgBar, kEcgThreads);
#endif
#endif
    if (lane == 0) {
#if !MUSE_VR_LDW_PRO_COMPACT  // compact form: warp 2 builds q_row2 in the next phase (no reader before barrier 3)
      int acc = 0;
#pragma unroll 1
      for (int qq = 0; qq < q && qq < kMaxPasses; ++qq) {
        s.q_row2[qq] = int16_t(acc);
        acc += bucket_rows(bucket_of(int(s.q_rows[qq])));
      }
      if (acc > kHs2Rows) fail(p.err, kErrShape);  // unreachable by the bound above
#endif
      s.n_pass = q;
#if MUSE_VR_LDW_BALB
      s.bal_nq_p = q;
      s.bal_rows_p = rows;
#endif
      s.head_n = 0;
#if !MUSE_VR_LDW_EARLY_COPY  // early copy: set above, before the hand-off
      s.start_base = ticket - ticket % gridDim.x;
      s.set = int((s.start_base / gridDim.x) & 1u);
#endif
#if MUSE_VR_LDW_ROWS_READY
      s.rr_epoch = rr_old / gridDim.x + 1ull;
#endif
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
      s.slot = muse_vr_slot::epoch_settle_sm(p.ticket + 32, p.ticket + kClaimWord + muse_vr_slot::kMissWord,
                                             reinterpret_cast<const uint8_t*>(p.sm_domain), ep_smid, ep_hint, ep_old,
                                             ticket / gridDim.x, kStartCycle, p.n_dom0, p.n_dom1, p.err);
#elif !MUSE_VR_SLOT_UNSAFE
      s.slot = muse_vr_slot::settle(p.ticket + kClaimWord, claim_hint, claim_old, p.n_dom0, p.n_dom1, p.err);
#endif
    }
  }
#if MUSE_VR_LDW_EARLY_COPY
  if (warp < 4) named_bar_sync(6, kThreads);  // the pass barrier (warps 4-11 arrived at it before the hand-off)
#else
  named_bar_sync(6, kThreads);
#endif
  if (tid == 0) LDW_STAMP(p, kStPass);
  if (warp == 0) {  // this CTA's work list: one round-robin over the domain list FC1(all passes) then FC2(all passes)
    unsigned slot = 0;
#if MUSE_VR_SLOT_UNSAFE
    if (lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      slot = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[smid]);
    }
    slot = shfl_idx(slot, 0);
#else
    slot = s.slot;  // claimed in the prologue (named barrier 6 above)
#endif
    int nq = s.n_pass;
    if (nq > kMaxPasses) {
      if (lane == 0) fail(p.err, kErrShape);
      nq = 0;
    }
    const unsigned dom = (slot >> 7) & 1u;
    const int R = dom ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
    const int n1 = R > 0 ? steps_of(kFc1TilesPerDomain * nq, r, R) : 0;
    const int n2 = R > 0 ? steps_of((kFc1TilesPerDomain + kFc2TilesPerDomain) * nq, r, R) - n1 : 0;
#if MUSE_VR_LDW_BALB
    // pinned part only: its FC1 tiles are the prefix of the deal; FC1 tiles of the board's passes and every FC2 tile follow
    // once the board is read (FC2 offsets depend on the final pass count)
    int nt = (p.dbg & 512) ? 0 : n1;
    (void)n2;
#else
    int nt = (p.dbg & 512) ? 0 : n1 + n2;
#endif
    if (nt > kMaxItems) {
      if (lane == 0) fail(p.err, kErrSlot);
      nt = 0;
    }
#pragma unroll 1
    for (int i = lane; i < nt; i += 32) {
      uint32_t v;
      if (i < n1) {
        const int g = r + i * R;
        v = item_pack(0, g / kFc1TilesPerDomain, kFc1TilesPerDomain * int(dom) + (g % kFc1TilesPerDomain));
      } else {
        const int pc = r + i * R - kFc1TilesPerDomain * nq;
        v = item_pack(1, pc / kFc2TilesPerDomain, kFc2TilesPerDomain * int(dom) + (pc % kFc2TilesPerDomain));
      }
      s.items[i] = v;
    }
    if (lane == 0) {
      s.n_fc1 = nt > 0 ? n1 : 0;
      s.n_items = nt;
#if MUSE_VR_LDW_BALB
      s.bal_n1_p = nt;
#endif
      LDW_STAMP(p, kStList);
    }
  } else if (MUSE_VR_LDW_HEADS && warp == 3 && (p.dbg & 256) == 0 && s.n_pass <= kMaxPasses) {
    // head copies: the token of each row of the passes of this CTA's first FC1 items (item h computes tile g = r + h R of
    // pass g / 32), while a pass has at most kHeadRows rows. Row i of pass qq of expert e is e's slot of rank
    // (qq - qbase[e]) * 128 + i in slot order - the copy phase's stable expert-major order: the per-chunk prefix counts
    // give the chunk holding the pass's first row, then an ordered scan from there (8 independent ballots per chunk).
    unsigned slot = 0;
#if MUSE_VR_SLOT_UNSAFE
    if (lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      slot = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[smid]);
    }
    slot = shfl_idx(slot, 0);
#else
    slot = s.slot;  // the slot warp 0's work list uses (claimed in the prologue)
#endif
    const int R = ((slot >> 7) & 1u) ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
    const int nq = s.n_pass;
    int hn = 0;
#if MUSE_VR_LDW_HEAD_GO
    int hq = -1;  // the head item's pass (the loaders do not read the work list)
#endif
#pragma unroll 1
    for (int h = 0; h < kHeadItems; ++h) {
      const int g = r + h * R;
      if (R <= 0 || g >= kFc1TilesPerDomain * nq) break;  // warp-uniform
      const int qq = g / kFc1TilesPerDomain, e = s.q_expert[qq], rows = s.q_rows[qq];
      if (rows > kHeadRows) break;
#if MUSE_VR_LDW_HEAD_GO
      if (h == 0) hq = qq;
#endif
      const int r0 = (qq - int(s.qbase[e])) * kPassRows;
      const int nch = (nslot + kChunkSlots - 1) / kChunkSlots;
      int c0 = 0;  // last chunk whose prefix count of e is <= r0: it holds e's slot of rank r0
#pragma unroll
      for (int st = kNumChunks / 2; st > 0; st >>= 1)
        if (c0 + st < nch && int(x.ccnt[c0 + st][e]) <= r0) c0 += st;
      int seen = int(x.ccnt[c0][e]);
#pragma unroll 1
      for (int c = c0; c < nch && seen < r0 + rows; ++c) {
        // one chunk at a time: its 8 groups of 32 slots loaded and balloted independently, then ranked in order
        int ex[kChunkSlots / 32];
        unsigned bal[kChunkSlots / 32];
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) {
          const int sl = c * kChunkSlots + 32 * u + lane;
          ex[u] = sl < nslot ? int(x.exp[sl]) : -2;
        }
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) bal[u] = __ballot_sync(0xffffffffu, ex[u] == e);
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) {
          const int rk = seen + __popc(bal[u] & ((1u << lane) - 1u));
          if (ex[u] == e && rk >= r0 && rk < r0 + rows) s.ptok[h][rk - r0] = int16_t((c * kChunkSlots + 32 * u + lane) / kTopK);
          seen += __popc(bal[u]);
        }
      }
      hn = h + 1;
    }
    if (lane == 0) {
      s.head_n = hn;
#if MUSE_VR_LDW_HEAD_GO
      s.head_q = hq;
#endif
    }
#if MUSE_VR_LDW_PRO_COMPACT
  } else if (warp == 2) {  // the FC2 operand's bucket-padded first row of each pass (a warp scan; read after barrier 3)
    const int nq = s.n_pass < kMaxPasses ? s.n_pass : kMaxPasses;
    int acc = 0;
#pragma unroll 1
    for (int q0 = 0; q0 < nq; q0 += 32) {
      const int qq = q0 + lane;
      const int v = qq < nq ? bucket_rows(bucket_of(int(s.q_rows[qq]))) : 0;
      int incl = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      if (qq < nq) s.q_row2[qq] = int16_t(acc + incl - v);
      acc += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (lane == 0 && acc > kHs2Rows) fail(p.err, kErrShape);  // unreachable by the bound above
#endif
  } else if (warp >= 8) {  // permuted positions of this CTA's slots (one warp per slot)
    const int w = warp & 3;
    const int G = int(gridDim.x);
    int n_my = (nslot > int(blockIdx.x)) ? (nslot - 1 - int(blockIdx.x)) / G + 1 : 0;
    if (n_my > kMaxMySlots) {
      if (tid == 8 * 32) fail(p.err, kErrShape);
      n_my = 0;
    }
#pragma unroll 1
    for (int j = w; j < n_my; j += 4) {
      const int sl = int(blockIdx.x) + G * j;
      const int e = x.exp[sl];
      int pos = -1;
      if (e >= 0) {
        // stable rank: earlier chunks (prefix) + earlier slots of the same expert in this chunk
        const int c0 = (sl / kChunkSlots) * kChunkSlots;
        int before = 0;
#pragma unroll 1
        for (int c = c0; c < sl; c += 32) {
          const int idx = c + lane;
          before += __popc(__ballot_sync(0xffffffffu, idx < sl && x.exp[idx] == e));
        }
#if MUSE_VR_LDW_BALB
        const int rank = int(x.ccnt[sl / kChunkSlots][e]) + before;
        if (lane == 0) {
          s.my_rank[j] = int16_t(rank);
          s.my_cand[j] = int8_t(s.cand[e]);
        }
        pos = s.cand[e] ? -1 : int(s.base[e]) + rank;  // a candidate's position comes with the board
#elif MUSE_VR_LDW_TEST_EC_TWIN == 1  // tests only: the expert's base may be unwritten here (clamped to a routed slot)
        pos = ec_twin_clamp(int(s.base[e]) + int(x.ccnt[sl / kChunkSlots][e]) + before, nslot - 1);
#else
        pos = int(s.base[e]) + int(x.ccnt[sl / kChunkSlots][e]) + before;
#endif
      }
#if MUSE_VR_LDW_BALB
      else if (lane == 0) {
        s.my_rank[j] = 0;
        s.my_cand[j] = 0;
      }
#endif
      if (lane == 0) {
        s.my_slot[j] = int16_t(sl);
        s.my_pos[j] = int16_t(pos);
        s.my_exp[j] = int16_t(e);
      }
    }
    if (tid == 8 * 32) s.n_my = n_my;
#if MUSE_VR_LDW_EPI
    // the tokens this CTA finalizes and their local route counts (from the prologue's slot table)
    if (w == 0) {
      int no = 0;
#pragma unroll 1
      for (int t0 = int(blockIdx.x); t0 < p.m; t0 += G * 32) {
        const int t = t0 + G * lane;
        int r = 0;
        if (t < p.m)
#pragma unroll
          for (int kk = 0; kk < kTopK; ++kk) r += x.exp[t * kTopK + kk] >= 0 ? 1 : 0;
        const unsigned bt = __ballot_sync(0xffffffffu, t < p.m);
        const int j = no + __popc(bt & ((1u << lane) - 1u));
        if (t < p.m && j < kMaxOwn) {
          s.own_t[j] = int16_t(t);
          s.own_r[j] = int8_t(r);
        }
        no += __popc(bt);
      }
      if (no > kMaxOwn) {
        if (lane == 0) fail(p.err, kErrShape);
        no = 0;
      }
      if (lane == 0) s.own_n = no;
    }
#endif
  } else if (warp >= 4) {  // the other launch parity's counters: zero this CTA's share for the next launch
#if MUSE_VR_LDW_TEST_EC_TWIN == 1  // tests only: the counter parity may be unwritten here (masked)
    unsigned* const other = p.cnt + (1 - (s.set & 1)) * kCntSet;
#else
    unsigned* const other = p.cnt + (1 - s.set) * kCntSet;
#endif
    for (int i = int(blockIdx.x) + int(gridDim.x) * (tid - 4 * 32); i < kCntSet; i += int(gridDim.x) * 128)
      muse_vr_ldq::st_relaxed_gpu(other + i, 0u);
#if MUSE_VR_LDW_EPI
    unsigned* const other_t = p.tok_cnt + (1 - s.set) * kMaxTokens;
    for (int i = int(blockIdx.x) + int(gridDim.x) * (tid - 4 * 32); i < kMaxTokens; i += int(gridDim.x) * 128)
      muse_vr_ldq::st_relaxed_gpu(other_t + i, 0u);
#endif
  }
#if MUSE_VR_LDW_HEAD_GO
  // warp 3: the head-row list (s.ptok, s.head_n, s.head_q; head_n = 0 when none was built) is complete: one release that the
  // head loaders acquire instead of the end-of-prologue barrier (every lane's stores are ordered before lane 0's release by
  // the warp barrier)
  if (warp == 3) {
    __syncwarp();
    if (lane == 0) {
      LDW_STAMP(p, kStHeadGo);
      st_release_cta_shared(&s.head_go, 1);
    }
  }
#endif
}

struct alignas(64) Maps {
  CUtensorMap m13, m2;                           // FC1 / FC2 weight stages
  CUtensorMap b1[kNumBuckets], b2[kNumBuckets];  // FC1 / FC2 token rows, one box height per bucket
#if MUSE_VR_LDW_THEAD
  CUtensorMap hm;                                // the token tensor's rows (2D, 128-byte boxes of one row) for the head gathers
#endif
};

__global__ void __launch_bounds__(kThreads, 1) ldw_kernel(const __grid_constant__ Maps maps, const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    g_abort = 0;
    LDW_STAMP(p, kStStart);
  }
  if (warp == 3) {
    if (elect_one()) {
      for (int i = 0; i < kStages; ++i) {
        mbar_init(&s.full[i], MUSE_VR_LDW_FULL_ARRIVALS);
        mbar_init(&s.empty[i], 1);
        mbar_init(&s.sf_full[i], 1);
      }
      for (int i = 0; i < 2; ++i) {
        mbar_init(&s.tmem_full[i], 1);
        mbar_init(&s.tmem_empty[i], 4);
      }
      mbar_init(&s.drain_sf, 1);
      mbar_init(&s.drain_mma, 1);
#if MUSE_VR_LDW_LANDED
      for (int i = 0; i < kStages; ++i) mbar_init(&s.tok_full[i], 4);
      for (int i = 0; i < 4; ++i) sts_u32(smem_u32(&s.fc1_stored[i]), 0u);  // shared-typed (a pre-wait store)
#if MUSE_VR_LDW_HEAD_GO
      sts_u32(smem_u32(&s.head_go), 0u);  // shared-typed (a pre-wait store); ordered before every reader by barrier 6
#endif
#if MUSE_VR_LDW_HEAD_PRIO == 1
      mbar_init(&s.head_iss, 4);
#endif
#else
      for (int i = 0; i < 4; ++i) s.fc1_stored[i] = 0u;
#if MUSE_VR_LDW_HB0_PUSH
      for (int i = 0; i < 4; ++i) s.fc2_stored[i] = 0u;
#endif
#if MUSE_VR_LDW_TEST_EC_COUNT
      s.ec_dbg[0] = 0u;  // tests only; ordered before every count by barrier 6 below
      s.ec_dbg[1] = 0u;
#endif
#endif
      fence_barrier_init();
      prefetch_tensormap(&maps.m13);
      prefetch_tensormap(&maps.m2);
    }
    __syncwarp();
  } else if (warp == 1) {
    tmem_alloc(&s.tmem_base, kTmemCols);
    tmem_relinquish();
    tc_fence_before();
  }
  named_bar_sync(6, kThreads);  // barrier initialization, tensor memory
  prologue(s, p);
#if MUSE_VR_LDW_HEAD_GO
  // the producer's first stage weights right after its own work list (warp 0 wrote it); the head loaders (warps 4-7) only
  // arrive at barrier 3 and start on warp 3's flag; every other warp arrives at the list barrier, where the loaders wait
  // before the rest of their role
  int pre0 = 0;
  if (warp == 0) pre0 = producer_first(s, p, &maps.m13);
  if (warp >= 4 && warp < 8) {
    named_bar_arrive(3, kThreads);
    landed_head(s, p, threadIdx.x - 4 * 32);
    named_bar_sync(MUSE_VR_LDW_LIST_BAR, kThreads);
  } else {
    named_bar_sync(3, kThreads);  // work list, pass tables, positions, counter parity for these roles
    named_bar_arrive(MUSE_VR_LDW_LIST_BAR, kThreads);
  }
  if (tid == 0) LDW_STAMP(p, kStPro);
#elif MUSE_VR_LDW_EARLY_COPY
  // early copy (HEAD_GO's split of barrier 3): warps 0-3 wait at barrier 3 for the work list (warp 0), the FC2 pass rows (warp 2
  // with the compact prologue) and warps 4-11's arrivals, then arrive at the list barrier. Warps 4-11 arrive at barrier 3 only
  // here, after their last read of the prologue scratch (the slot positions); the producer's first copies into the last ring
  // stage, which holds the scratch, follow barrier 3. They then meet at the copy warps' barrier (positions of warps 8-11 for all
  // eight copy warps), run the copy phase (no scratch read: its operands are outside the ring stages, asserted at Smem), and
  // wait at the list barrier before their roles read the work list. The other parity's counters were zeroed by warps 4-7 above,
  // so those stores precede, in program order, the copy's barrier 7 and its release of this parity's count.
  if (warp < 4) {
    named_bar_sync(3, kThreads);  // work list, pass tables, positions, counter parity for these roles
    named_bar_arrive(MUSE_VR_LDW_LIST_BAR, kThreads);
  } else {
#if MUSE_VR_LDW_TEST_EC_PLANT == 1
    if (warp < 8) named_bar_arrive(3, kThreads);  // tests only: warps 8-11 arrived before their positions (prologue)
#else
    named_bar_arrive(3, kThreads);
#endif
#if MUSE_VR_LDW_TEST_EC_TWIN != 2
#if MUSE_VR_LDW_TEST_EC_COUNT
    atomicAdd(&s.ec_dbg[1], 1u);
#endif
    named_bar_sync(kEcBar, kEcThreads);
#else
    ec_test_wait();  // tests only: the copy warps' barrier dropped (racecheck control)
#endif
#if MUSE_VR_LDW_TEST_EC_TWIN == 1  // tests only: the counter parity may be unwritten here (masked)
    LDW_COPY_ROWS(s, p, p.cnt + (s.set & 1) * kCntSet, tid - 4 * 32);
#else
    LDW_COPY_ROWS(s, p, p.cnt + s.set * kCntSet, tid - 4 * 32);
#endif
    named_bar_sync(MUSE_VR_LDW_LIST_BAR, kThreads);  // work list, pass tables for the epilogue and builder roles
  }
  if (tid == 0) LDW_STAMP(p, kStPro);
#else
  named_bar_sync(3, kThreads);  // work list, pass tables, positions, counter parity for every role
  if (tid == 0) LDW_STAMP(p, kStPro);
#endif
  tc_fence_after();
  if (warp == 0) {
#if MUSE_VR_LDW_HEAD_GO
    role_producer(s, p, &maps.m13, &maps.m2, maps.b1, maps.b2, pre0);
#else
#if MUSE_VR_LDW_THEAD
    role_producer(s, p, &maps.m13, &maps.m2, maps.b1, maps.b2, 0, &maps.hm);
#else
    role_producer(s, p, &maps.m13, &maps.m2, maps.b1, maps.b2, 0);
#endif
#endif
  } else if (warp == 1) {
    role_mma(s, p);
  } else if (warp == 2) {
    role_sfcopy(s, p);
  } else if (warp == 3) {
    role_ctrl(s, p);
  } else if (warp < 8) {
    role_epilogue(s, p);
  } else {
    role_builder(s, p);
  }
  named_bar_sync(5, kThreads);
#if MUSE_VR_LDW_TEST_EC_COUNT
  // tests only: every participant counted itself before its operation on the two early-copy barriers (ordered by barrier 5)
  if (tid == 0 && (s.ec_dbg[0] != ((32u << 16) | 256u) || s.ec_dbg[1] != uint32_t(kEcThreads))) atomicOr(p.err, kErrEcCount);
#endif
  if (tid == 0) {
    LDW_STAMP(p, kStEnd);
#ifdef MUSE_VR_LDW_STAMPS
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    int cost = 0;
    for (int i = 0; i < s.n_items; ++i) cost += bucket_rows(bucket_of(s.q_rows[item_unpack(s.items[i]).q]));
    LDW_STAMPV(p, kStItems, (uint64_t(s.n_items) << 16) | uint64_t(s.n_fc1));
    LDW_STAMPV(p, kStCost, uint64_t(cost));
    LDW_STAMPV(p, kStSm, uint64_t(smid));
#endif
  }
#if MUSE_VR_LDW_LANDED
  landed_guard_exit(s, p);  // after every read of the landing by this CTA (barrier 5)
#endif
  if (warp == 1) tmem_dealloc(s.tmem_base, kTmemCols);
#if MUSE_VR_LDW_TRAP
  if (tid == 0 && aborted()) __trap();  // after the drains and the tensor-memory release: the error word is already set
#endif
}

}  // inline namespace MUSE_VR_LDW_ABI
}  // namespace muse_vr_ldw

// ---------------------------------------------------------------- host
#ifndef MUSE_VR_LDW_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <mutex>
#if MUSE_VR_LDW_HB0_PUSH
#include "lb_hop_ops.cuh"  // make_site / make_ctl: the same Site / Ctl structs the hop library's launchers build
#endif
#include <unordered_map>

namespace muse_vr_ldw {
inline namespace MUSE_VR_LDW_ABI {
namespace {

using PFN_encodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                      const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                      CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

PFN_encodeTiled get_encode_fn() {
  static PFN_encodeTiled fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    sglang::host::RuntimeDeviceCheck(
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
    sglang::host::RuntimeCheck(q == cudaDriverEntryPointSuccess && ptr != nullptr, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_encodeTiled>(ptr);
  }();
  return fn;
}

// [rows, 2048] uint8 rows viewed as (128-byte K atom, row, atom index); a box of box_rows rows x box_atoms atoms lands as
// box_atoms consecutive 128-byte-swizzled atoms of box_rows rows.
CUtensorMap make_rows_map(const void* base, uint64_t rows, uint32_t box_rows, uint32_t box_atoms) {
  CUtensorMap map{};
  const cuuint64_t dim[3] = {128, rows, kHidden / 2 / 128};
  const cuuint64_t stride[2] = {kHidden / 2, 128};
  const cuuint32_t box[3] = {128, box_rows, box_atoms};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult rc = get_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dim, stride, box,
                                      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return map;
}

#if MUSE_VR_LDW_THEAD
// [rows, 2048] uint8 token rows as a 2D map with boxes of one row x 128 bytes (128-byte swizzle): the head's four-row gathers
CUtensorMap make_tokens_map(const void* base, uint64_t rows) {
  CUtensorMap map{};
  const cuuint64_t dim[2] = {kHidden / 2, rows};
  const cuuint64_t stride[1] = {kHidden / 2};
  const cuuint32_t box[2] = {128, 1};
  const cuuint32_t es[2] = {1, 1};
  const CUresult rc = get_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dim, stride, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled (token rows, gather) failed");
  return map;
}
#endif

const CUtensorMap& cached_map(const void* base, uint64_t rows, uint32_t box_rows, uint32_t box_atoms) {
  static std::mutex mu;
  static std::unordered_map<uint64_t, CUtensorMap> cache;
  uint64_t key = reinterpret_cast<uint64_t>(base);
  key = key * 1000003ull ^ rows;
  key = key * 1000003ull ^ (uint64_t(box_rows) << 8 | box_atoms);
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, make_rows_map(base, rows, box_rows, box_atoms)).first;
  return it->second;
}

}  // namespace

struct MuseVrLdw {
  static constexpr std::size_t kSmem = sizeof(Smem) + 1024;

  static void preload() {
    using namespace sglang::host;
    int dev = 0, optin = 0;
    RuntimeDeviceCheck(cudaGetDevice(&dev));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    RuntimeCheck(kSmem <= std::size_t(optin), "shared memory per block above the device's opt-in limit");
    RuntimeDeviceCheck(cudaFuncSetAttribute(ldw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
    cudaFuncAttributes attr;
    RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, ldw_kernel));
  }

  static void run(const tvm::ffi::TensorView packed, const tvm::ffi::TensorView hs, const tvm::ffi::TensorView hs_sf,
                  const tvm::ffi::TensorView pts, const tvm::ffi::TensorView w13, const tvm::ffi::TensorView sf13,
                  const tvm::ffi::TensorView w2, const tvm::ffi::TensorView sf2, const tvm::ffi::TensorView gate_alpha,
                  const tvm::ffi::TensorView up_alpha, const tvm::ffi::TensorView out_alpha,
                  const tvm::ffi::TensorView sm_domain, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView expert_weights, const tvm::ffi::TensorView e2p,
                  const tvm::ffi::TensorView hs_perm, const tvm::ffi::TensorView pts_perm, const tvm::ffi::TensorView img1,
                  const tvm::ffi::TensorView img2, const tvm::ffi::TensorView act, const tvm::ffi::TensorView hs2,
                  const tvm::ffi::TensorView ramax, const tvm::ffi::TensorView cnt, const tvm::ffi::TensorView ctl,
                  const tvm::ffi::TensorView err, const tvm::ffi::Optional<tvm::ffi::TensorView> stamps,
                  int64_t local_offset, int64_t n_dom0, int64_t n_dom1, double gsi, int64_t flags, int64_t dbg,
                  int64_t pdl
#if MUSE_VR_LDW_HB0_PUSH
                  , const tvm::ffi::TensorView hb_table, const tvm::ffi::TensorView hb_ctl, int64_t hb_rank,
                  int64_t hb_rows_max, int64_t hb_width, int64_t hb_pitch, int64_t hb_policy, int64_t hb_parity,
                  int64_t hb_index, int64_t hb_site, int64_t hb_units, int64_t hb_max_rounds, int64_t hb_spin_rounds,
                  int64_t hb_flags, int64_t hb_backoff_ns, const tvm::ffi::TensorView hb_slot
#endif
#if MUSE_VR_LDW_ROWS_READY
                  , const tvm::ffi::TensorView rr
#endif
                  ) {
    using namespace sglang;
    using namespace sglang::host;
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"pool"};
    auto NSM = SymbolicSize{"sm_count"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kTopK}).with_dtype<int32_t>().with_device(device_).verify(packed);
    TensorMatcher({M, kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs);
    TensorMatcher({M, kHidden / 16}).with_dtype<uint8_t>().with_device(device_).verify(hs_sf);
    TensorMatcher({M}).with_dtype<float>().with_device(device_).verify(pts);
    TensorMatcher({P, 2 * kInter, kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(w13);
    TensorMatcher({P, 2 * kInter, kHidden / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf13);
    TensorMatcher({P, kHidden, kInter / 2}).with_dtype<uint8_t>().with_device(device_).verify(w2);
    TensorMatcher({P, kHidden, kInter / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf2);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(gate_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(up_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(out_alpha);
    TensorMatcher({NSM}).with_dtype<int8_t>().with_device(device_).verify(sm_domain);
    auto MR = SymbolicSize{"rows"};
    TensorMatcher({MR, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(out);
    TensorMatcher({M, kTopK}).with_dtype<int16_t>().with_device(device_).verify(expert_weights);
    TensorMatcher({MR}).with_dtype<int32_t>().with_device(device_).verify(e2p);
    TensorMatcher({int64_t(kMaxSlots), kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs_perm);
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<float>().with_device(device_).verify(pts_perm);
    TensorMatcher({int64_t(kMaxPasses) * kImgBytes}).with_dtype<uint8_t>().with_device(device_).verify(img1);
    TensorMatcher({int64_t(kMaxPasses) * kImgBytes}).with_dtype<uint8_t>().with_device(device_).verify(img2);
    TensorMatcher({int64_t(kMaxSlots) * kActWords}).with_dtype<int64_t>().with_device(device_).verify(act);
    TensorMatcher({int64_t(kHs2Rows), kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs2);
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<float>().with_device(device_).verify(ramax);
    TensorMatcher({int64_t(2 * kCntSet)}).with_dtype<int32_t>().with_device(device_).verify(cnt);
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    TensorMatcher({32 + muse_vr_slot::kEpochWords}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#else
    TensorMatcher({32}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#endif
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(err);
    if (stamps.has_value())
      TensorMatcher({NSM.unwrap() * int64_t(kStampWords)}).with_dtype<int64_t>().with_device(device_).verify(stamps.value());
    const int64_t m = M.unwrap(), pool = P.unwrap();
    RuntimeCheck(m >= 1 && m <= kMaxTokens, "token count outside 1..512");
    RuntimeCheck(MR.unwrap() == m * kTopK, "output rows must be num_tokens * top_k");
    RuntimeCheck(pool >= 1 && pool <= kMaxPool, "pool outside 1..128");
    RuntimeCheck(local_offset >= 0 && local_offset + pool <= 32768, "bad local expert offset");
#if MUSE_VR_LDW_TEST_GRID > 0
    n_dom0 = MUSE_VR_LDW_TEST_GRID / 2;
    n_dom1 = MUSE_VR_LDW_TEST_GRID - MUSE_VR_LDW_TEST_GRID / 2;
    RuntimeCheck(MUSE_VR_LDW_TEST_GRID <= NSM.unwrap(), "test grid above the SM count");
#define MUSE_VR_LDW_GRID_ int64_t(MUSE_VR_LDW_TEST_GRID)
#else
    RuntimeCheck(n_dom0 >= 0 && n_dom1 >= 0 && n_dom0 + n_dom1 == NSM.unwrap(), "domain slots must cover the grid");
#define MUSE_VR_LDW_GRID_ NSM.unwrap()
#endif
    RuntimeCheck(NSM.unwrap() <= 256 && n_dom0 <= 128 && n_dom1 <= 128, "slot table holds 128 ranks per domain");
    RuntimeCheck(m * kTopK <= int64_t(kMaxMySlots) * MUSE_VR_LDW_GRID_, "routed slots per CTA above the copy table");

    Params p{};
    p.packed = static_cast<const int32_t*>(packed.data_ptr());
    p.hs = static_cast<const uint8_t*>(hs.data_ptr());
    p.hs_sf = static_cast<const uint8_t*>(hs_sf.data_ptr());
    p.pts = static_cast<const float*>(pts.data_ptr());
    p.sf13 = static_cast<const uint8_t*>(sf13.data_ptr());
    p.sf2 = static_cast<const uint8_t*>(sf2.data_ptr());
    p.gate_alpha = static_cast<const float*>(gate_alpha.data_ptr());
    p.up_alpha = static_cast<const float*>(up_alpha.data_ptr());
    p.out_alpha = static_cast<const float*>(out_alpha.data_ptr());
    p.sm_domain = static_cast<const int8_t*>(sm_domain.data_ptr());
    p.hs_perm = static_cast<uint8_t*>(hs_perm.data_ptr());
    p.pts_perm = static_cast<float*>(pts_perm.data_ptr());
    p.img1 = static_cast<uint8_t*>(img1.data_ptr());
    p.img2 = static_cast<uint8_t*>(img2.data_ptr());
    p.act = static_cast<unsigned long long*>(act.data_ptr());
    p.hs2 = static_cast<uint8_t*>(hs2.data_ptr());
    p.ramax = static_cast<float*>(ramax.data_ptr());
    p.cnt = static_cast<unsigned*>(cnt.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.e2p = static_cast<int32_t*>(e2p.data_ptr());
    p.expert_weights = static_cast<uint16_t*>(expert_weights.data_ptr());
    p.ticket = static_cast<unsigned*>(ctl.data_ptr());
    p.err = static_cast<int32_t*>(err.data_ptr());
    p.m = static_cast<int32_t>(m);
    p.pool = static_cast<int32_t>(pool);
    p.local_offset = static_cast<int32_t>(local_offset);
    p.n_dom0 = static_cast<int32_t>(n_dom0);
    p.n_dom1 = static_cast<int32_t>(n_dom1);
    p.gsi = static_cast<float>(gsi);
    p.flags = static_cast<int32_t>(flags);
    p.dbg = static_cast<int32_t>(dbg);
    p.stamps = stamps.has_value() ? static_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
#if MUSE_VR_LDW_HB0_PUSH
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<int32_t>().with_device(device_).verify(hb_slot);
    RuntimeCheck(hb_width == int64_t(muse_vr::hb0::kWidth), "HB0 push site: width must be the tile-aligned row (c3_hb0.cuh kWidth)");
    RuntimeCheck(m * kTopK <= hb_rows_max * MUSE_VR_LDW_HB0_PUSH, "HB0 push site: M * 8 linear rows above the region");
    p.hb = sglang::muse_vr_lb_hop_ops::make_site<MUSE_VR_LDW_HB0_PUSH>(hb_table, hb_rank, hb_rows_max, hb_width, hb_pitch,
                                                                       hb_policy, hb_parity, hb_index, hb_site, hb_units);
    p.hbc = sglang::muse_vr_lb_hop_ops::make_ctl(hb_ctl, hb_max_rounds, hb_spin_rounds, hb_flags, hb_backoff_ns);
    p.hb_slot = static_cast<int32_t*>(hb_slot.data_ptr());
#endif
#if MUSE_VR_LDW_ROWS_READY
    TensorMatcher({int64_t(kRrWords)}).with_dtype<int64_t>().with_device(device_).verify(rr);
    p.rr = static_cast<unsigned long long*>(rr.data_ptr());
#endif

    Maps maps;
    const uint64_t r13 = uint64_t(pool) * 2 * kInter, r2 = uint64_t(pool) * kHidden;
    maps.m13 = cached_map(w13.data_ptr(), r13, kTile, kChunks / 4);
    maps.m2 = cached_map(w2.data_ptr(), r2, kTile, kChunks / 4);
    for (int b = 0; b < kNumBuckets; ++b) {
      maps.b1[b] = cached_map(hs_perm.data_ptr(), uint64_t(kMaxSlots), uint32_t(bucket_rows(b)), kChunks / 4);
      maps.b2[b] = cached_map(hs2.data_ptr(), uint64_t(kHs2Rows), uint32_t(bucket_rows(b)), kChunks / 4);
    }
#if MUSE_VR_LDW_THEAD
    maps.hm = make_tokens_map(hs.data_ptr(), uint64_t(m));  // this call's token tensor (a graph capture fixes its address)
#endif
    [[maybe_unused]] static const bool configured = [] {
      preload();
      return true;
    }();
    LaunchKernel(dim3(static_cast<uint32_t>(MUSE_VR_LDW_GRID_)), dim3(kThreads), device_.unwrap(), kSmem)
        .enable_pdl(pdl != 0)(ldw_kernel, maps, p);
  }
#if MUSE_VR_LDW_LANDED
  // Landed build: ids and token rows from this rank's landing buffers (int64 words; copy 0 at the tensor's start,
  // [copy][source rank][rows_max][pitch]); seq64 = the forward sequence word, last_seq = one guard word per CTA.
  static void run_landed(int64_t m, const tvm::ffi::TensorView ids_land, const tvm::ffi::TensorView tok_land,
                         const tvm::ffi::TensorView seq64, const tvm::ffi::TensorView last_seq, int64_t ids_pitch,
                         int64_t tok_pitch, int64_t rows_max, int64_t world, int64_t ids_parity, int64_t tok_parity,
                         const tvm::ffi::TensorView w13, const tvm::ffi::TensorView sf13, const tvm::ffi::TensorView w2,
                         const tvm::ffi::TensorView sf2, const tvm::ffi::TensorView gate_alpha,
                         const tvm::ffi::TensorView up_alpha, const tvm::ffi::TensorView out_alpha,
                         const tvm::ffi::TensorView sm_domain, const tvm::ffi::TensorView out,
                         const tvm::ffi::TensorView expert_weights, const tvm::ffi::TensorView e2p,
                         const tvm::ffi::TensorView hs_perm, const tvm::ffi::TensorView pts_perm,
                         const tvm::ffi::TensorView img1, const tvm::ffi::TensorView img2, const tvm::ffi::TensorView act,
                         const tvm::ffi::TensorView hs2, const tvm::ffi::TensorView ramax, const tvm::ffi::TensorView cnt,
                         const tvm::ffi::TensorView ctl, const tvm::ffi::TensorView err,
                         const tvm::ffi::Optional<tvm::ffi::TensorView> stamps, int64_t local_offset, int64_t n_dom0,
                         int64_t n_dom1, double gsi, int64_t flags, int64_t dbg, int64_t pdl
#if MUSE_VR_LDW_EPI
                         , const tvm::ffi::TensorView norm_w, double norm_eps, const tvm::ffi::TensorView moments,
                         const tvm::ffi::TensorView pos_slot, const tvm::ffi::TensorView tok_cnt,
                         const tvm::ffi::TensorView tok_out
#endif
#if MUSE_VR_LDW_BALB
                         , const tvm::ffi::TensorView bal_table, const tvm::ffi::TensorView board, int64_t bal_rank
#if MUSE_VR_LDW_TEST_BOARD_DELAY_NS > 0
                         , int64_t test_delay_ns
#endif
#endif
                         ) {
    using namespace sglang;
    using namespace sglang::host;
    auto P = SymbolicSize{"pool"};
    auto NSM = SymbolicSize{"sm_count"};
    auto IDW = SymbolicSize{"ids_words"};
    auto TKW = SymbolicSize{"tok_words"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({IDW}).with_dtype<int64_t>().with_device(device_).verify(ids_land);
    TensorMatcher({TKW}).with_dtype<int64_t>().with_device(device_).verify(tok_land);
    TensorMatcher({1}).with_dtype<int64_t>().with_device(device_).verify(seq64);
    TensorMatcher({P, 2 * kInter, kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(w13);
    TensorMatcher({P, 2 * kInter, kHidden / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf13);
    TensorMatcher({P, kHidden, kInter / 2}).with_dtype<uint8_t>().with_device(device_).verify(w2);
    TensorMatcher({P, kHidden, kInter / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf2);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(gate_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(up_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(out_alpha);
    TensorMatcher({NSM}).with_dtype<int8_t>().with_device(device_).verify(sm_domain);
    TensorMatcher({NSM}).with_dtype<int64_t>().with_device(device_).verify(last_seq);
    TensorMatcher({m * kTopK, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(out);
    TensorMatcher({m, kTopK}).with_dtype<int16_t>().with_device(device_).verify(expert_weights);
    TensorMatcher({m * kTopK}).with_dtype<int32_t>().with_device(device_).verify(e2p);
    TensorMatcher({int64_t(kMaxSlots), kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs_perm);
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<float>().with_device(device_).verify(pts_perm);
    TensorMatcher({int64_t(kMaxPasses) * kImgBytes}).with_dtype<uint8_t>().with_device(device_).verify(img1);
    TensorMatcher({int64_t(kMaxPasses) * kImgBytes}).with_dtype<uint8_t>().with_device(device_).verify(img2);
    TensorMatcher({int64_t(kMaxSlots) * kActWords}).with_dtype<int64_t>().with_device(device_).verify(act);
    TensorMatcher({int64_t(kHs2Rows), kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs2);
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<float>().with_device(device_).verify(ramax);
    TensorMatcher({int64_t(2 * kCntSet)}).with_dtype<int32_t>().with_device(device_).verify(cnt);
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    TensorMatcher({32 + muse_vr_slot::kEpochWords}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#else
    TensorMatcher({32}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#endif
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(err);
    if (stamps.has_value())
      TensorMatcher({NSM.unwrap() * int64_t(kStampWords)}).with_dtype<int64_t>().with_device(device_).verify(stamps.value());
    const int64_t pool = P.unwrap();
    RuntimeCheck(m >= 1 && m <= kMaxTokens, "token count outside the build's rows per call");
    RuntimeCheck(pool >= 1 && pool <= kMaxPool, "pool outside 1..128");
    RuntimeCheck(local_offset >= 0 && local_offset + pool <= 32768, "bad local expert offset");
    RuntimeCheck(world == 1 || world == 2 || world == 4 || world == 8 || world == 16, "world size 1, 2, 4, 8 or 16");
    RuntimeCheck(rows_max >= m && ids_pitch >= kTopK && tok_pitch >= kTokWords + 1 && tok_pitch % 2 == 0,
                 "landing rows_max / pitch (token pitch even, above the row's words)");
    const int64_t ids_copy = world * rows_max * ids_pitch, tok_copy = world * rows_max * tok_pitch;
    RuntimeCheck(IDW.unwrap() >= (ids_parity ? 2 : 1) * ids_copy && TKW.unwrap() >= (tok_parity ? 2 : 1) * tok_copy,
                 "landing buffers smaller than their copies");
    RuntimeCheck(reinterpret_cast<uintptr_t>(tok_land.data_ptr()) % 16 == 0, "token landing 16-byte aligned");
#if MUSE_VR_LDW_TEST_GRID > 0
    n_dom0 = MUSE_VR_LDW_TEST_GRID / 2;
    n_dom1 = MUSE_VR_LDW_TEST_GRID - MUSE_VR_LDW_TEST_GRID / 2;
    RuntimeCheck(MUSE_VR_LDW_TEST_GRID <= NSM.unwrap(), "test grid above the SM count");
#else
    RuntimeCheck(n_dom0 >= 0 && n_dom1 >= 0 && n_dom0 + n_dom1 == NSM.unwrap(), "domain slots must cover the grid");
#endif
    RuntimeCheck(NSM.unwrap() <= 256 && n_dom0 <= 128 && n_dom1 <= 128, "slot table holds 128 ranks per domain");
    RuntimeCheck(m * kTopK <= int64_t(kMaxMySlots) * MUSE_VR_LDW_GRID_, "routed slots per CTA above the copy table");

    Params p{};
    p.sf13 = static_cast<const uint8_t*>(sf13.data_ptr());
    p.sf2 = static_cast<const uint8_t*>(sf2.data_ptr());
    p.gate_alpha = static_cast<const float*>(gate_alpha.data_ptr());
    p.up_alpha = static_cast<const float*>(up_alpha.data_ptr());
    p.out_alpha = static_cast<const float*>(out_alpha.data_ptr());
    p.sm_domain = static_cast<const int8_t*>(sm_domain.data_ptr());
    p.hs_perm = static_cast<uint8_t*>(hs_perm.data_ptr());
    p.pts_perm = static_cast<float*>(pts_perm.data_ptr());
    p.img1 = static_cast<uint8_t*>(img1.data_ptr());
    p.img2 = static_cast<uint8_t*>(img2.data_ptr());
    p.act = static_cast<unsigned long long*>(act.data_ptr());
    p.hs2 = static_cast<uint8_t*>(hs2.data_ptr());
    p.ramax = static_cast<float*>(ramax.data_ptr());
    p.cnt = static_cast<unsigned*>(cnt.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.e2p = static_cast<int32_t*>(e2p.data_ptr());
    p.expert_weights = static_cast<uint16_t*>(expert_weights.data_ptr());
    p.ticket = static_cast<unsigned*>(ctl.data_ptr());
    p.err = static_cast<int32_t*>(err.data_ptr());
    p.m = static_cast<int32_t>(m);
    p.pool = static_cast<int32_t>(pool);
    p.local_offset = static_cast<int32_t>(local_offset);
    p.n_dom0 = static_cast<int32_t>(n_dom0);
    p.n_dom1 = static_cast<int32_t>(n_dom1);
    p.gsi = static_cast<float>(gsi);
    p.flags = static_cast<int32_t>(flags);
    p.dbg = static_cast<int32_t>(dbg);
    p.stamps = stamps.has_value() ? static_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
    p.ids_land = static_cast<const unsigned long long*>(ids_land.data_ptr());
    p.tok_land = static_cast<const unsigned long long*>(tok_land.data_ptr());
    p.ids_copy_words = static_cast<unsigned long long>(ids_copy);
    p.tok_copy_words = static_cast<unsigned long long>(tok_copy);
    p.seq64 = static_cast<const unsigned long long*>(seq64.data_ptr());
    p.last_seq = static_cast<unsigned long long*>(last_seq.data_ptr());
    p.ids_pitch = static_cast<uint32_t>(ids_pitch);
    p.tok_pitch = static_cast<uint32_t>(tok_pitch);
    p.rows_max = static_cast<uint32_t>(rows_max);
    p.world = static_cast<uint32_t>(world);
    p.ids_parity = ids_parity ? 1u : 0u;
    p.tok_parity = tok_parity ? 1u : 0u;
#if MUSE_VR_LDW_EPI
    TensorMatcher({kHidden}).with_dtype<float>().with_device(device_).verify(norm_w);
    TensorMatcher({int64_t(kMaxSlots) * kMomParts}).with_dtype<float>().with_device(device_).verify(moments);
    TensorMatcher({int64_t(kMaxSlots)}).with_dtype<int32_t>().with_device(device_).verify(pos_slot);
    TensorMatcher({int64_t(2 * kMaxTokens)}).with_dtype<int32_t>().with_device(device_).verify(tok_cnt);
    TensorMatcher({m, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(tok_out);
    p.norm_w = static_cast<const float*>(norm_w.data_ptr());
    p.norm_eps = static_cast<float>(norm_eps);
    p.moments = static_cast<float*>(moments.data_ptr());
    p.pos_slot = static_cast<int32_t*>(pos_slot.data_ptr());
    p.tok_cnt = static_cast<unsigned*>(tok_cnt.data_ptr());
    p.tok_out = static_cast<__nv_bfloat16*>(tok_out.data_ptr());
#endif
#if MUSE_VR_LDW_BALB
    TensorMatcher({256}).with_dtype<int32_t>().with_device(device_).verify(bal_table);
    TensorMatcher({33}).with_dtype<int64_t>().with_device(device_).verify(board);
    RuntimeCheck(bal_rank >= 0 && bal_rank < MUSE_VR_LDW_BALB, "bal rank outside the expert-parallel world");
    p.bal_table = static_cast<const uint32_t*>(bal_table.data_ptr());
    p.board = static_cast<const unsigned long long*>(board.data_ptr());
    p.bal_rank = static_cast<int32_t>(bal_rank);
#if MUSE_VR_LDW_TEST_BOARD_DELAY_NS > 0
    p.test_delay_ns = static_cast<uint32_t>(test_delay_ns);
#endif
#endif

    Maps maps;
    const uint64_t r13 = uint64_t(pool) * 2 * kInter, r2 = uint64_t(pool) * kHidden;
    maps.m13 = cached_map(w13.data_ptr(), r13, kTile, kChunks / 4);
    maps.m2 = cached_map(w2.data_ptr(), r2, kTile, kChunks / 4);
    for (int b = 0; b < kNumBuckets; ++b) {
      maps.b1[b] = cached_map(hs_perm.data_ptr(), uint64_t(kMaxSlots), uint32_t(bucket_rows(b)), kChunks / 4);
      maps.b2[b] = cached_map(hs2.data_ptr(), uint64_t(kHs2Rows), uint32_t(bucket_rows(b)), kChunks / 4);
    }
    [[maybe_unused]] static const bool configured = [] {
      preload();
      return true;
    }();
    LaunchKernel(dim3(static_cast<uint32_t>(MUSE_VR_LDW_GRID_)), dim3(kThreads), device_.unwrap(), kSmem)
        .enable_pdl(pdl != 0)(ldw_kernel, maps, p);
  }
#endif
};

}  // inline namespace MUSE_VR_LDW_ABI
}  // namespace muse_vr_ldw
#endif  // MUSE_VR_LDW_DEVICE_ONLY
