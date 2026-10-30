# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Finite SM103 configured-q1 component checks, not model/TP8 qualification.

Run this file on one SM103 GPU with the pinned native dependencies. Imports and
compilation are real; no native entry point is mocked. Host execution only tests
the CPU oracle and skips the three CUDA cases. A GPU run must execute all three.

The q1 oracle applies the FP32 KDA reference equations to persistent q1 state.
The BF16 F_B projection boundary and existing tolerances are retained. Projection
checks follow the API2 component probe's PreparedProjection eager/graph pattern.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

from sgl_nvidia_mega.kimi_k3.shape_policy import DECODE_WIDTHS

NEW_ROWS = (9, 15, 17, 31, 33, 63)
CONTROL_ROWS = (1, 8, 16, 32, 64)
ROWS = tuple(sorted((*NEW_ROWS, *CONTROL_ROWS)))
REPLAY_ORDER = (0, 1, 0)
HEADS, DIM, PITCH, HIDDEN = 12, 1536, 6288, 7168
SEED, GUARD, SENTINEL = 20261029, 256, -13.0
KDA_ATOL, KDA_RTOL, COSINE = 0.03, 0.03, 0.999
PROJECTION_ATOL, PROJECTION_RTOL = 0.2, 0.02


def _random(shape, seed, *, dtype=torch.bfloat16, scale=0.2):
    return (
        torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * scale
    ).to(dtype)


def _weights(heads=HEADS):
    dim = heads * 128
    return {
        "conv": _random((3 * dim, 1, 4), SEED, dtype=torch.float32),
        "f_b": _random((dim, 128), SEED + 1, scale=128**-0.5),
        "a_log": _random((heads,), SEED + 2, dtype=torch.float32),
        "dt_bias": _random((dim,), SEED + 3, dtype=torch.float32),
        "o_norm": _random((128,), SEED + 4, scale=0.1) + 1,
    }


def _q1_reference(projected, caches, slots, weights):
    """CPU FP32 mathematical oracle; inactive KDA slots <=0 are read/write-free."""
    if projected.device.type != "cpu" or any(
        t.device.type != "cpu" for t in (*caches, *weights.values())
    ):
        raise ValueError("reference requires CPU tensors")
    active = [index for index, slot in enumerate(slots) if slot > 0]
    targets = [slots[index] for index in active]
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate live state slots race in native q1")
    heads = weights["a_log"].numel()
    dim = heads * 128
    out = torch.zeros((len(slots), dim), dtype=torch.bfloat16)
    conv, state = (value.clone() for value in caches)
    if not active:
        return out, (conv, state)
    row = projected[active]
    history = caches[0][targets]
    full = torch.cat((history, row[:, : 3 * dim].unsqueeze(1)), dim=1)
    cw = weights["conv"].view(3, dim, 4).permute(0, 2, 1)
    query, key, value = [
        F.silu(
            (full[:, :, part * dim : (part + 1) * dim].float() * cw[part]).sum(1)
        ).view(-1, heads, 128)
        for part in range(3)
    ]
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    gate = (
        F.linear(row[:, 4 * dim : 4 * dim + 128].float(), weights["f_b"].float())
        .to(torch.bfloat16)
        .float()
        .view(-1, heads, 128)
    )
    decay = torch.exp(
        -5.0
        * torch.sigmoid(
            weights["a_log"].exp()[None, :, None]
            * (gate + weights["dt_bias"].view(heads, 128))
        )
    )
    beta = torch.sigmoid(row[:, 4 * dim + 128 : 4 * dim + 128 + heads].float())
    updated = caches[1][targets] * decay[:, :, None, :]
    residual = value - (updated * key[:, :, None, :]).sum(-1)
    updated = updated + (residual * beta[:, :, None]).unsqueeze(-1) * key[:, :, None, :]
    raw = (updated * query[:, :, None, :]).sum(-1)
    normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-5)
    output_gate = torch.sigmoid(row[:, 3 * dim : 4 * dim].float().view(-1, heads, 128))
    out[active] = (
        (normalized * weights["o_norm"].float() * output_gate)
        .flatten(1)
        .to(torch.bfloat16)
    )
    conv[targets] = full[:, 1:]
    state[targets] = updated
    return out, (conv, state)


class _Guarded:
    def __init__(self, shape, dtype, device):
        self.storage = torch.full(
            (math.prod(shape) + 2 * GUARD,), SENTINEL, dtype=dtype, device=device
        )
        self.value = self.storage[GUARD:-GUARD].view(shape)
        self.pointer = self.value.data_ptr()

    def check(self, case):
        case.assertEqual(self.value.data_ptr(), self.pointer)
        case.assertTrue(bool((self.storage[:GUARD] == SENTINEL).all()))
        case.assertTrue(bool((self.storage[-GUARD:] == SENTINEL).all()))


def _capture(run, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
            # KdaSplit's PDL trigger is not a shared-state dependency boundary.
            # Warmups mutate the same cache; finish each launch before the next.
            stream.synchronize()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream(device).wait_stream(stream)
    return graph


def _numerical(case, actual, expected, *, atol, rtol, cosine=False):
    actual, expected = actual.detach().cpu().float(), expected.float()
    case.assertTrue(bool(torch.isfinite(actual).all()))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if cosine:
        a, b = actual.flatten(1), expected.flatten(1)
        active = b.norm(dim=-1) > 0
        if bool(active.any()):
            minimum = float(F.cosine_similarity(a[active], b[active], dim=-1).min())
            case.assertGreaterEqual(minimum, COSINE)


class TestConfiguredWidthOracle(unittest.TestCase):
    def test_cpu_reference_has_exact_conv_and_inactive_semantics(self):
        heads, dim, capacity = 2, 256, 4
        weights = _weights(heads)
        weights["conv"].zero_()
        weights["conv"][:, :, -1] = 1
        for name in ("f_b", "a_log", "dt_bias"):
            weights[name].zero_()
        weights["o_norm"].fill_(1)
        projected = torch.zeros(3, 4 * dim + 128 + heads, dtype=torch.bfloat16)
        projected[:, : 2 * dim] = 1
        projected[:, 2 * dim : 3 * dim] = 0.5
        caches = (
            torch.zeros(capacity, 3, 3 * dim, dtype=torch.bfloat16),
            torch.zeros(capacity, heads, 128, 128),
        )
        out, (conv, state) = _q1_reference(projected, caches, [2, 0, -1], weights)
        self.assertTrue(torch.equal(out[1:], torch.zeros_like(out[1:])))
        self.assertTrue(torch.equal(conv[2, 2], projected[0, : 3 * dim]))
        self.assertTrue(torch.equal(conv[[0, 1, 3]], caches[0][[0, 1, 3]]))
        self.assertTrue(torch.equal(state[[0, 1, 3]], caches[1][[0, 1, 3]]))
        scalar = F.silu(torch.tensor(1.0))
        key = scalar * torch.rsqrt(128 * scalar.square() + 1e-6)
        expected_state = 0.5 * F.silu(torch.tensor(0.5)) * key
        torch.testing.assert_close(state[2], torch.full_like(state[2], expected_state))
        raw = expected_state * key * math.sqrt(128)
        expected_out = (0.5 * raw / torch.sqrt(raw.square() + 1e-5)).bfloat16()
        self.assertTrue(torch.equal(out[0], torch.full_like(out[0], expected_out)))
        self.assertEqual(set(NEW_ROWS) & set(DECODE_WIDTHS), set())
        self.assertTrue(set(CONTROL_ROWS) <= set(DECODE_WIDTHS))


@unittest.skipUnless(
    torch.cuda.is_available(), "native CUDA component qualification required"
)
class TestConfiguredWidthsCuda(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.assertEqual(
            torch.cuda.get_device_capability(self.device),
            (10, 3),
            "qualification requires SM103",
        )

    def test_kda_mode_zero_new_rows_and_default_controls_eager_graph_aba(self):
        from sgl_nvidia_mega.kimi_k3.cutedsl.kda.prepared import PreparedKda
        from sgl_nvidia_mega.kimi_k3.providers import _kda_policy

        weights_cpu = _weights()
        weights = {name: value.to(self.device) for name, value in weights_cpu.items()}
        attention = NS(
            conv1d=NS(weight=weights["conv"], bias=None),
            dt_bias=weights["dt_bias"],
            A_log=weights["a_log"],
            f_b_proj=NS(weight=weights["f_b"]),
            o_norm=NS(weight=weights["o_norm"], eps=1e-5),
            gate_lower_bound=-5.0,
        )
        capacity = max(ROWS) + 3
        seeds = (
            _random((capacity, 3, 3 * DIM), SEED + 10),
            _random(
                (capacity, HEADS, 128, 128), SEED + 11, dtype=torch.float32, scale=0.125
            ),
        )
        conv = _Guarded(seeds[0].shape, torch.bfloat16, self.device)
        state = _Guarded(seeds[1].shape, torch.float32, self.device)
        caches = (conv.value, state.value)
        policies = _kda_policy(1, ROWS)
        self.assertTrue(all(policies[row] == (PITCH, 0) for row in NEW_ROWS))
        self.assertEqual(
            {row: policies[row] for row in CONTROL_ROWS},
            {row: _kda_policy(1)[row] for row in CONTROL_ROWS},
        )
        prepared = PreparedKda(attention, policies, False).prepare_caches(caches)
        self.assertEqual(tuple(prepared.kernels), ROWS)
        kernel_ids = {row: id(kernel) for row, kernel in prepared.kernels.items()}
        prepared_weights = [
            (value.data_ptr(), value.cpu()) for value in prepared.weights
        ]
        for rows in ROWS:
            with self.subTest(rows=rows):
                projected = _Guarded((rows, PITCH), torch.bfloat16, self.device)
                output = _Guarded((rows, DIM), torch.bfloat16, self.device)
                slots = _Guarded((rows,), torch.int32, self.device)
                phases = [
                    _random((rows, PITCH), SEED + rows * 2 + phase) for phase in (0, 1)
                ]
                maps = [list(range(rows, 0, -1)), list(range(1, rows + 1))]
                if rows == 1:
                    maps[1][0] = 0
                else:
                    maps[0][-2:] = [0, -1]
                    maps[1][:2] = [-1, 0]
                expected = [
                    _q1_reference(phases[p], seeds, maps[p], weights_cpu)
                    for p in (0, 1)
                ]

                def reset(
                    phase,
                    projected=projected,
                    slots=slots,
                    output=output,
                    phases=phases,
                    maps=maps,
                ):
                    projected.value.copy_(phases[phase])
                    slots.value.copy_(torch.tensor(maps[phase], dtype=torch.int32))
                    conv.value.copy_(seeds[0])
                    state.value.copy_(seeds[1])
                    output.value.fill_(float("nan"))

                def run(projected=projected, output=output, slots=slots):
                    self.assertIs(
                        prepared.run(
                            projected.value,
                            output.value,
                            caches=caches,
                            slots=slots.value,
                        ),
                        output.value,
                    )

                reset(0)
                graph = _capture(run, self.device)
                first = None
                for turn, phase in enumerate(REPLAY_ORDER):
                    eager = None
                    for route in ("eager", "graph"):
                        reset(phase)
                        run() if route == "eager" else graph.replay()
                        torch.cuda.synchronize(self.device)
                        oracle_out, (oracle_conv, oracle_state) = expected[phase]
                        _numerical(
                            self,
                            output.value,
                            oracle_out,
                            atol=KDA_ATOL,
                            rtol=KDA_RTOL,
                            cosine=True,
                        )
                        _numerical(
                            self,
                            state.value,
                            oracle_state,
                            atol=KDA_ATOL,
                            rtol=KDA_RTOL,
                        )
                        self.assertTrue(torch.equal(conv.value.cpu(), oracle_conv))
                        padding = [
                            index for index, slot in enumerate(maps[phase]) if slot <= 0
                        ]
                        self.assertTrue(
                            torch.equal(
                                output.value[padding].cpu(),
                                torch.zeros((len(padding), DIM), dtype=torch.bfloat16),
                            )
                        )
                        inactive = [
                            slot
                            for slot in range(capacity)
                            if slot not in maps[phase] or slot <= 0
                        ]
                        self.assertTrue(
                            torch.equal(state.value[inactive].cpu(), seeds[1][inactive])
                        )
                        self.assertTrue(
                            torch.equal(projected.value.cpu(), phases[phase])
                        )
                        self.assertEqual(slots.value.cpu().tolist(), maps[phase])
                        active = [slot for slot in maps[phase] if slot > 0]
                        if active:
                            self.assertFalse(
                                torch.equal(state.value[active].cpu(), seeds[1][active])
                            )
                        current = (output.value.cpu(), state.value.cpu())
                        if route == "eager":
                            eager = current
                        else:
                            self.assertTrue(
                                all(
                                    torch.equal(a, b)
                                    for a, b in zip(current, eager, strict=True)
                                )
                            )
                        if turn == 0 and route == "eager":
                            first = current
                        if phase == 0:
                            self.assertTrue(
                                all(
                                    torch.equal(a, b)
                                    for a, b in zip(current, first, strict=True)
                                )
                            )
                        else:
                            self.assertFalse(torch.equal(current[0], first[0]))
                        for guarded in (projected, output, slots, conv, state):
                            guarded.check(self)
                self.assertEqual(
                    kernel_ids,
                    {row: id(kernel) for row, kernel in prepared.kernels.items()},
                )
        for name, weight in weights.items():
            self.assertTrue(torch.equal(weight.cpu(), weights_cpu[name]))
        for value, (pointer, expected) in zip(
            prepared.weights, prepared_weights, strict=True
        ):
            self.assertEqual(value.data_ptr(), pointer)
            self.assertTrue(torch.equal(value.cpu(), expected))

    def _projection(self, pitch):
        from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import PreparedProjection
        from sgl_nvidia_mega.kimi_k3.providers import _projection_policies

        weight_cpu = _random((pitch, HIDDEN), SEED + pitch, scale=HIDDEN**-0.5)
        if pitch == PITCH:
            weight_cpu[-4:].zero_()
        weight = weight_cpu.to(self.device)
        policies = _projection_policies(pitch, ROWS)
        self.assertEqual(
            {row: policies[row] for row in CONTROL_ROWS},
            _projection_policies(pitch, CONTROL_ROWS),
        )
        prepared = PreparedProjection(weight, policies).prepare()
        self.assertIs(prepared.weight, weight)
        self.assertEqual(tuple(prepared.kernels), ROWS)
        owners = {
            key: (id(value), value.data_ptr()) for key, value in prepared.maps.items()
        }
        for rows in ROWS:
            with self.subTest(pitch=pitch, rows=rows):
                x = _Guarded((rows, HIDDEN), torch.bfloat16, self.device)
                out = _Guarded((rows, pitch), torch.bfloat16, self.device)
                phases = [
                    _random((rows, HIDDEN), SEED + rows * 2 + p, scale=1.0)
                    for p in (0, 1)
                ]
                expected = [
                    F.linear(value.float(), weight_cpu.float()) for value in phases
                ]
                x.value.copy_(phases[0])

                def run(x=x, out=out):
                    self.assertIs(prepared.run(x.value, out.value), out.value)

                graph = _capture(run, self.device)
                first = None
                for turn, phase in enumerate(REPLAY_ORDER):
                    eager = None
                    for route in ("eager", "graph"):
                        x.value.copy_(phases[phase])
                        out.value.fill_(float("nan"))
                        run() if route == "eager" else graph.replay()
                        torch.cuda.synchronize(self.device)
                        _numerical(
                            self,
                            out.value,
                            expected[phase],
                            atol=PROJECTION_ATOL,
                            rtol=PROJECTION_RTOL,
                            cosine=True,
                        )
                        self.assertTrue(torch.equal(x.value.cpu(), phases[phase]))
                        current = out.value.cpu()
                        if route == "eager":
                            eager = current
                        else:
                            self.assertTrue(torch.equal(current, eager))
                        if turn == 0 and route == "eager":
                            first = current
                        if phase == 0:
                            self.assertTrue(torch.equal(current, first))
                        else:
                            self.assertFalse(torch.equal(current, first))
                        x.check(self)
                        out.check(self)
                self.assertEqual(
                    owners,
                    {
                        key: (id(value), value.data_ptr())
                        for key, value in prepared.maps.items()
                    },
                )
        self.assertTrue(torch.equal(weight.cpu(), weight_cpu))

    def test_projection_mla_new_rows_and_default_controls_eager_graph_aba(self):
        self._projection(3648)

    def test_projection_kda_new_rows_and_default_controls_eager_graph_aba(self):
        self._projection(PITCH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
