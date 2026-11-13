# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for Muse Spark's verify_mk component (TensorRT-LLM's Rubin decode megakernel on SGLang).

The vendored TensorRT-LLM host code needs a Rubin GPU, so it is replaced, for this module only, by small stand-ins with
the names verify_mk reads; the SGLang modules verify_mk imports are stand-ins too. The component's modules are imported
fresh on top of them and dropped again afterwards, so other tests see the real modules.
"""

from __future__ import annotations

import importlib
import sys
import types
from enum import Enum
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import sgl_nvidia_mega
from sgl_nvidia_mega.muse_spark.components import sgl_compat

COMPONENTS = "sgl_nvidia_mega.muse_spark.components"
VERIFY_MK = f"{COMPONENTS}.verify_mk"
MEGAKERNEL = f"{COMPONENTS}.vendor.trtllm_ab93b78.megakernel"
VENDOR = (
    Path(sgl_nvidia_mega.__file__).resolve().parent
    / "muse_spark/components/vendor/trtllm_ab93b78"
)
ROWS, HIDDEN = 8, 8192


# ---- stand-ins for the vendored host code: the names verify_mk reads ----
class _ServedMegakernel:
    def __init__(self, layers, mapping, comm, **kwargs):
        self.layers, self.mapping, self.comm, self.kwargs = (
            list(layers),
            mapping,
            comm,
            kwargs,
        )
        self.rank = int(mapping.tp_rank)
        self.kernels = None
        self.current = None
        self.refused_bindings = 0
        self.forwards = 0
        self.indices = list(range(1, len(self.layers) - 1))
        self.sliding = [True] * len(self.layers)


class _Base:  # the vendored workspace classes verify_mk's subclasses extend
    pass


def _vendor_modules() -> dict:
    return {
        "serving": dict(
            ServedMegakernel=_ServedMegakernel,
            DEFAULT_MODE="lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_lm2_s4",
            DEFAULT_I2B_FLAGS=0,
            DEFAULT_CAP_MOE="0x0",
            DEFAULT_W1_PLAN=(),
            FA_FLAGS=0,
            FA_LATE_FIRST=0,
            FH_PLAN=(),
            ROWS=ROWS,
            HIDDEN=HIDDEN,
            MAX_BINDINGS=4,
            _log=lambda *a, **k: None,
            refusals=lambda capability, mapping, layers: [],
        ),
        "layer_mk": dict(
            HopWorkspace=_Base,
            HEADS_FIRST_BLOCKS=[2 * h for h in range(16)]
            + [32, 33, 34, 35]
            + [2 * h + 1 for h in range(16)],
        ),
        "library": dict(KernelIo=lambda **kw: types.SimpleNamespace(**kw)),
        "rl": dict(RlWorkspace=_Base, RANKS=8, COPY_BYTES=128 * 1024, apply=None),
        "moe_hops": dict(
            RANKS=8,
            OUTPUTS={  # moe_hops.OUTPUTS
                "MhOutRes1": ((ROWS, HIDDEN), torch.float32),
                "MhOutRes2": ((ROWS, HIDDEN), torch.float32),
                "MhOutXffn": ((ROWS, HIDDEN), torch.bfloat16),
                "MhOutPool": ((ROWS, 4096), torch.bfloat16),
                "MhOutLogits": ((256, ROWS), torch.float32),
                "MhOutM": ((ROWS, 4096), torch.bfloat16),
                "MhOutB": ((ROWS, HIDDEN), torch.bfloat16),
                "MhOutXattn": ((ROWS, HIDDEN), torch.bfloat16),
            },
        ),
        "experts": dict(EXPERTS=256),
        "qkv": {},
    }


# ---- stand-ins for the SGLang modules verify_mk imports ----
class HookType(Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    REPLACE = "replace"


class HookRegistry:
    hooks: list = []

    @classmethod
    def register(cls, target, hook, hook_type=HookType.AFTER, **kwargs):
        cls.hooks.append((target, hook, hook_type))


class TRTLLMHAAttnBackend:
    def __init__(self):
        self.page_size = 128
        self.target_verify_metadata = {}
        self.forward_metadata = None
        self.token_to_kv_pool = None


def _sglang_modules() -> dict:
    forward = types.SimpleNamespace(backend=None)
    return {
        "sglang.srt.plugins.hook_registry": dict(
            HookRegistry=HookRegistry, HookType=HookType
        ),
        "sglang.srt.layers.attention.trtllm_mha_backend": dict(
            TRTLLMHAAttnBackend=TRTLLMHAAttnBackend
        ),
        "sglang.srt.model_executor.forward_context": dict(
            get_attn_backend=lambda: forward.backend,
            get_req_to_token_pool=lambda: (
                sys.modules["sglang.srt.model_executor.forward_context"]
                .get_attn_backend()
                .req_to_token_pool
            ),
        ),
        "sglang.srt.runtime_context": dict(get_parallel=None, get_server_args=None),
        "sglang.srt.distributed.parallel_state": dict(get_tp_group=None),
    }


def _stub(mp: pytest.MonkeyPatch, name: str, **attrs) -> types.ModuleType:
    """A stand-in module in ``sys.modules``, with stand-in packages for the parents not imported yet."""
    parts = name.split(".")
    for k in range(1, len(parts)):
        parent = ".".join(parts[:k])
        if parent not in sys.modules:
            package = types.ModuleType(parent)
            package.__path__ = []
            mp.setitem(sys.modules, parent, package)
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    mp.setitem(sys.modules, name, mod)
    return mod


def _drop_verify_mk() -> None:
    for name in [
        n for n in sys.modules if n == VERIFY_MK or n.startswith(VERIFY_MK + ".")
    ]:
        del sys.modules[name]


@pytest.fixture(scope="module")
def mods():
    """verify_mk's modules imported on the stand-ins; dropped (and the real modules back) after this module."""
    _drop_verify_mk()
    with pytest.MonkeyPatch.context() as mp:
        vendor = {
            name: _stub(mp, f"{MEGAKERNEL}.{name}", **attrs)
            for name, attrs in _vendor_modules().items()
        }
        package = _stub(mp, MEGAKERNEL, **vendor)
        package.__path__ = []
        for name, attrs in _sglang_modules().items():
            _stub(mp, name, **attrs)
        ns = types.SimpleNamespace(
            plugin=importlib.import_module(VERIFY_MK),
            runtime=importlib.import_module(f"{VERIFY_MK}.runtime"),
            kv=importlib.import_module(f"{VERIFY_MK}.kv"),
            comm=importlib.import_module(f"{VERIFY_MK}.comm"),
            stages=importlib.import_module(f"{VERIFY_MK}.stages"),
        )
        yield ns
        _drop_verify_mk()


class Mode:
    def __init__(self, name):
        self.name = name

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"


# ---- the entry point: refusals, registration, install hook, layer wraps, capture check ----
@pytest.fixture
def server(mods, monkeypatch):
    """A TP8 = EP8 Rubin server with FP8 KV pages of 128 and 8 draft tokens, and no conflicting component."""
    rc = sys.modules["sglang.srt.runtime_context"]
    state = types.SimpleNamespace(
        par=types.SimpleNamespace(
            tp_size=8,
            moe_ep_size=8,
            attn_tp_size=8,
            pp_size=1,
            tp_rank=3,
            moe_ep_rank=3,
        ),
        args=types.SimpleNamespace(
            page_size=128, kv_cache_dtype="fp8_e4m3", speculative_num_draft_tokens=8
        ),
    )
    monkeypatch.setattr(rc, "get_parallel", lambda: state.par)
    monkeypatch.setattr(rc, "get_server_args", lambda: state.args)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 7))
    for name in mods.plugin.CONFLICTS:
        monkeypatch.delenv(f"MUSE_VR_{name.upper()}", raising=False)
    monkeypatch.delenv("MUSE_VR_PLUGIN_DISABLE", raising=False)
    return state


def lm(mods, n=None):
    return types.SimpleNamespace(
        layers=[object()] * (mods.plugin.LAYERS if n is None else n)
    )


def test_refusals_none(mods, server):
    assert mods.plugin.refusals(lm(mods)) == []


@pytest.mark.parametrize("name", ["full_attn_wide", "mid_rs", "moe_ld", "moe_tail"])
def test_refusals_conflicting_component(mods, server, monkeypatch, name):
    assert name in mods.plugin.CONFLICTS
    monkeypatch.setenv(f"MUSE_VR_{name.upper()}", "1")
    (reason,) = mods.plugin.refusals(lm(mods))
    assert reason.endswith(f"MUSE_VR_{name.upper()}"), reason
    monkeypatch.setenv("MUSE_VR_PLUGIN_DISABLE", name)  # forced off: nothing to replace
    assert mods.plugin.refusals(lm(mods)) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("par.tp_size", 4),
        ("par.moe_ep_size", 1),
        ("par.attn_tp_size", 1),
        ("par.moe_ep_rank", 5),
        ("args.page_size", 64),
        ("args.kv_cache_dtype", "bfloat16"),
        ("args.speculative_num_draft_tokens", 16),
    ],
)
def test_refusals_server(mods, server, field, value):
    obj, name = field.split(".")
    setattr(getattr(server, obj), name, value)
    assert len(mods.plugin.refusals(lm(mods))) == 1


def test_refusals_device_and_model(mods, server, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 3))
    assert len(mods.plugin.refusals(lm(mods, 61))) == 2
    assert "no MuseSpark1xModel" in mods.plugin.refusals(None)[-1]


def test_register(mods, monkeypatch):
    plugin = mods.plugin
    monkeypatch.setattr(HookRegistry, "hooks", [])
    monkeypatch.delenv("MUSE_VR_VERIFY_MK", raising=False)
    monkeypatch.delenv("MUSE_VR_PLUGIN_DISABLE", raising=False)
    assert plugin.register() is False and HookRegistry.hooks == []
    monkeypatch.setenv("MUSE_VR_VERIFY_MK", "1")
    monkeypatch.setattr(
        plugin, "pins_match", lambda: (False, "sglang/srt/models/muse_spark.py differs")
    )
    assert (
        plugin.register() is False and HookRegistry.hooks == []
    )  # a moved source: stock, with the reason
    monkeypatch.setattr(plugin, "pins_match", lambda: (True, "ok"))
    monkeypatch.setattr(sgl_compat, "target", lambda dotted: "installed:" + dotted)
    assert plugin.register() is True
    ((target, hook, hook_type),) = HookRegistry.hooks
    assert (
        target == "installed:" + plugin.TARGET
        and hook is plugin._around_install
        and hook_type.name == "AROUND"
    )


@pytest.fixture
def compat_cache():
    sgl_compat.module_name.cache_clear()
    sgl_compat.target.cache_clear()
    yield
    sgl_compat.module_name.cache_clear()
    sgl_compat.target.cache_clear()


@pytest.mark.parametrize(
    "installed",
    ["sglang.srt.megakernels.runtime", "sglang.srt.kernel_plugins.nvidia.binding"],
)
def test_install_target_follows_the_binding(mods, monkeypatch, compat_cache, installed):
    monkeypatch.setattr(sgl_compat, "_exists", lambda name: name == installed)
    assert sgl_compat.target(mods.plugin.TARGET) == f"{installed}.install_megakernel"


def test_around_install(mods, monkeypatch):
    plugin, seen = mods.plugin, []
    monkeypatch.setattr(
        plugin, "install", lambda model, hooks: seen.append(("install", model, hooks))
    )

    def original(**kwargs):
        seen.append(("original", kwargs["is_draft"]))
        return "out"

    hooks = []
    assert (
        plugin._around_install(
            original, model="m", hf_config=None, capture_tail_hooks=hooks, is_draft=True
        )
        == "out"
    )
    assert seen == [("original", True)]
    plugin._around_install(
        original, model="m", hf_config=None, capture_tail_hooks=hooks, is_draft=False
    )
    assert seen[1:] == [("original", False), ("install", "m", hooks)]


class FakeDecoderLayer:
    def __init__(self, index):
        self.index = index

    def forward(self, positions, stream, attn_input, forward_batch):
        return ("stock", self.index)


class FakeMk:
    def __init__(self, compare=0):
        self.indices = list(range(1, 61))
        self.current = None
        self.calls = []
        self.compare_forwards = compare

    def begin_compare(self, forward_batch, positions, stream, attn_input):
        self.calls.append("begin_compare")
        self.current = {} if forward_batch == "verify" else None

    def compare_layer(
        self, index, original, positions, stream, attn_input, forward_batch
    ):
        self.calls.append(index)
        if index == self.indices[-1]:
            self.current = None
        return original(positions, stream, attn_input, forward_batch)

    def begin_sg(self, forward_batch, positions, stream, attn_input):
        self.calls.append("begin")
        self.current = {} if forward_batch == "verify" else None
        return self.current is not None

    def run_layer(self, index, x, res):
        self.calls.append((index, x, res))
        if index == self.indices[-1]:
            self.current = None
        return ("mk", index)


def run_loop(model, forward_batch):
    return [
        layer.forward("pos", "stream", "x", forward_batch) for layer in model.layers
    ]


def test_wrap_layers(mods):
    layers = mods.plugin.LAYERS
    model = types.SimpleNamespace(layers=[FakeDecoderLayer(i) for i in range(layers)])
    mk = FakeMk()
    mods.plugin._wrap_layers(model, mk)
    out = run_loop(model, "verify")
    assert out[0] == ("stock", 0) and out[61] == ("stock", 61)
    assert out[1:61] == [("mk", i) for i in range(1, 61)]
    assert mk.calls[0] == "begin" and mk.calls[1] == (
        1,
        "x",
        "stream",
    )  # x = the attention input, res = the stream
    mk.calls.clear()
    assert run_loop(model, "prefill") == [("stock", i) for i in range(layers)]
    assert mk.calls == ["begin"]


def test_capture_check(mods):
    mk = types.SimpleNamespace(last_refusal="not the bs-1 verify graph's metadata")
    check = mods.plugin._capture_check(mk)
    verify = types.SimpleNamespace(forward_mode=Mode("TARGET_VERIFY"), batch_size=1)
    with pytest.raises(RuntimeError, match="verify graph's metadata"):
        check(None, None, verify, 8)
    check(None, None, verify, 16)  # another graph
    check(
        None, None, types.SimpleNamespace(forward_mode=Mode("DECODE"), batch_size=1), 8
    )
    mk.last_refusal = ""
    check(None, None, verify, 8)


def test_wrap_layers_compare(mods):
    layers = mods.plugin.LAYERS
    model = types.SimpleNamespace(layers=[FakeDecoderLayer(i) for i in range(layers)])
    mk = FakeMk(compare=2)
    mods.plugin._wrap_layers(model, mk)
    assert run_loop(model, "verify") == [
        ("stock", i) for i in range(layers)
    ]  # the stock outputs go on
    assert mk.calls == ["begin_compare"] + list(range(1, 61))


# ---- the per-forward binding, the launches and the compare mode (runtime) ----
def metadata(pages=4):
    return types.SimpleNamespace(
        is_ragged_verify=False,
        max_seq_len_q=ROWS,
        page_table=torch.zeros(1, pages, dtype=torch.int32),
        swa_page_table=torch.zeros(1, pages, dtype=torch.int32),
        cache_seqlens_int32=torch.zeros(1, dtype=torch.int32),
    )


@pytest.fixture
def env(mods, monkeypatch):
    """A built SgMegakernel (``_build`` recorded), a trtllm_mha backend on the bs-1 verify graph's metadata."""
    backend = TRTLLMHAAttnBackend()
    md = metadata()
    backend.target_verify_metadata.update({"page_table": md.page_table, 1: md})
    backend.forward_metadata = md
    backend.token_to_kv_pool = "pool"
    monkeypatch.setattr(
        sys.modules["sglang.srt.model_executor.forward_context"],
        "get_attn_backend",
        lambda: backend,
    )
    capturing = types.SimpleNamespace(on=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing.on)
    mk = mods.runtime.SgMegakernel([object()] * 62, tp_rank=3, local_rank=3, comm=None)
    mk.kernels = types.SimpleNamespace(ws=types.SimpleNamespace(plan_pos=0))
    builds = []

    def build(sg_md, pos32, spec):
        builds.append(sg_md)
        return {"md": sg_md}

    mk._build = build
    batch = types.SimpleNamespace(forward_mode=Mode("TARGET_VERIFY"), batch_size=1)
    positions = torch.arange(1000, 1000 + ROWS, dtype=torch.int64)
    stream = torch.zeros(ROWS, HIDDEN, dtype=torch.float32)
    x = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
    return types.SimpleNamespace(
        mods=mods,
        mk=mk,
        backend=backend,
        md=md,
        builds=builds,
        capturing=capturing,
        args=[batch, positions, stream, x],
    )


def test_mapping(mods):
    runtime = mods.runtime
    m = runtime.SgMegakernel(
        [object()] * 62, tp_rank=5, local_rank=1, comm=None
    ).mapping
    assert (m.tp_rank, m.local_rank, m.tp_size, m.moe_ep_size, m.pp_size) == (
        5,
        1,
        8,
        8,
        1,
    )
    assert runtime.MODE.endswith("_lm2_sg_s4") and runtime.CHECK_MODE.startswith(
        "check_"
    )
    assert "_rl3_" in runtime.CHECK_MODE and "_rl1_" not in runtime.CHECK_MODE


def test_bind_once(env):
    mk = env.mk
    assert mk.begin_sg(*env.args) and mk.last_refusal == ""
    (sg_md,) = env.builds
    assert (
        sg_md.kv_lens_cuda_runtime is env.md.cache_seqlens_int32
        and sg_md.pool == "pool"
    )
    assert (
        sg_md.page_table.shape == (1, 4)
        and sg_md.swa_page_table.data_ptr() == env.md.swa_page_table.data_ptr()
    )
    assert mk.current == {"md": sg_md}
    assert mk.positions32.dtype == torch.int32 and mk.positions32.tolist() == list(
        range(1000, 1000 + ROWS)
    )
    env.args[1].add_(8)  # the next step's positions, same buffer
    env.capturing.on = True  # a capture reuses the warm-ups' binding
    assert mk.begin_sg(*env.args) and len(env.builds) == 1
    assert mk.positions32.tolist() == list(range(1008, 1008 + ROWS))


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda e: setattr(e.mk.kernels.ws, "plan_pos", 3), "copy plan"),
        (lambda e: setattr(e.args[0], "forward_mode", Mode("EXTEND")), "EXTEND batch"),
        (lambda e: setattr(e.args[0], "batch_size", 2), "batch of 2"),
        (lambda e: e.args.__setitem__(2, torch.zeros(16, HIDDEN)), "stream (16, 8192)"),
        (lambda e: e.args.__setitem__(3, torch.zeros(ROWS, HIDDEN)), "bf16"),
        (
            lambda e: setattr(e.backend, "forward_metadata", metadata()),
            "verify graph's metadata",
        ),
        (lambda e: setattr(e.md, "is_ragged_verify", True), "ragged True"),
        (lambda e: setattr(e.md, "swa_page_table", None), "SWA page table False"),
        (lambda e: setattr(e.backend, "page_size", 64), "page size 64"),
        (lambda e: setattr(e.capturing, "on", True), "inside a capture"),
    ],
)
def test_refused(env, change, reason):
    change(env)
    assert not env.mk.begin_sg(*env.args)
    assert reason in env.mk.last_refusal and env.mk.current is None and env.builds == []


def test_backend_type(env, monkeypatch):
    other = types.SimpleNamespace(forward_metadata=env.md)
    monkeypatch.setattr(
        sys.modules["sglang.srt.model_executor.forward_context"],
        "get_attn_backend",
        lambda: other,
    )
    assert (
        not env.mk.begin_sg(*env.args)
        and "attention backend SimpleNamespace" in env.mk.last_refusal
    )


def test_binding_limit(env):
    for _ in range(4):  # a new positions buffer each time: a new binding
        env.args[1] = torch.arange(ROWS, dtype=torch.int64)
        assert env.mk.begin_sg(*env.args)
    env.args[1] = torch.arange(ROWS, dtype=torch.int64)
    assert not env.mk.begin_sg(*env.args) and "binding limit" in env.mk.last_refusal
    assert env.mk.refused_bindings == 1 and len(env.builds) == 4


def test_run_layer(env):
    mk, launches = env.mk, []
    mk.kernels.run_swa = lambda blocks, x, res, pdl: launches.append(
        ("swa", blocks, x, res)
    )
    mk.kernels.run_full = lambda blocks, x, res, pdl: launches.append(
        ("full", blocks, x, res)
    )
    mk.sliding = [i % 4 != 3 for i in range(62)]
    mk.x_out = [f"x{p}" for p in range(60)]
    mk.res_out = [f"res{p}" for p in range(60)]
    assert mk.begin_sg(*env.args)
    mk.current = {i: f"blocks{i}" for i in range(1, 61)}
    x, res = torch.zeros(2), torch.ones(2)
    assert mk.run_layer(1, x, res) == ("res0", "x0")
    assert mk.run_layer(3, x, res) == ("res2", "x2")
    assert [(kind, blocks) for kind, blocks, _, _ in launches] == [
        ("swa", "blocks1"),
        ("full", "blocks3"),
    ]
    assert launches[0][2] is x and launches[0][3] is res
    assert mk.current is not None and mk.forwards == 0
    mk.run_layer(60, x, res)
    assert mk.current is None and mk.forwards == 1


def test_layer_kv(env, monkeypatch):
    maps = []
    kv = env.mods.kv
    monkeypatch.setattr(
        kv,
        "tensor_maps",
        lambda lib, k, v, swizzle64: maps.append(swizzle64) or torch.zeros(256),
    )
    mk = env.mk
    mk.kernels.lib = None
    mk.sliding = [i % 4 != 3 for i in range(62)]
    bufs = {
        i: (
            torch.zeros(256, 2, 64, dtype=torch.uint8),
            torch.zeros(256, 2, 64, dtype=torch.uint8),
        )
        for i in (1, 3)
    }
    pool = types.SimpleNamespace(
        layers_mapping={1: (1, True), 3: (0, False)},
        get_key_buffer=lambda i: bufs[i][0],
        get_value_buffer=lambda i: bufs[i][1],
    )
    md = types.SimpleNamespace(
        pool=pool, page_table=torch.zeros(1, 4, dtype=torch.int32)
    )
    md.swa_page_table = torch.zeros(1, 4, dtype=torch.int32)
    swa, full = mk._layer_kv(1, md), mk._layer_kv(3, md)
    assert swa.fields()["AhBlockTable"] == md.swa_page_table.data_ptr()
    assert full.fields()["FaTable"] == md.page_table.data_ptr() and maps == [
        False,
        True,
    ]
    md2 = types.SimpleNamespace(
        pool=pool, page_table=torch.zeros(1, 8, dtype=torch.int32)
    )
    md2.swa_page_table = torch.zeros(1, 8, dtype=torch.int32)
    assert (
        mk._layer_kv(1, md2) is swa
        and swa.table is md2.swa_page_table
        and len(maps) == 2
    )  # maps kept
    pool.layers_mapping[1] = (1, False)
    with pytest.raises(AssertionError):
        mk._layer_kv(1, md)


class FakePool:
    def __init__(self, sliding):
        self.layers_mapping = {i: (i, s) for i, s in enumerate(sliding)}
        self.k = {
            i: torch.zeros(2048, 2, 64, dtype=torch.uint8) for i in range(len(sliding))
        }
        self.v = {
            i: torch.zeros(2048, 2, 64, dtype=torch.uint8) for i in range(len(sliding))
        }

    def get_key_buffer(self, i):
        return self.k[i].view(torch.float8_e4m3fn)

    def get_value_buffer(self, i):
        return self.v[i].view(torch.float8_e4m3fn)


@pytest.fixture
def cmp_env(env):
    mk = env.mods.runtime.SgMegakernel(
        [object()] * 62, tp_rank=0, local_rank=0, comm=None, compare=2
    )
    assert mk.kwargs["early_x"] is False
    mk.kernels = env.mk.kernels

    def build(sg_md, pos32, spec):
        env.builds.append(sg_md)
        return {"_md": sg_md, **{i: f"blocks{i}" for i in range(1, 61)}}

    mk._build = build
    mk.sliding = [i % 4 != 3 for i in range(62)]
    mk.x_out = [torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16) for _ in range(60)]
    mk.res_out = [torch.zeros(ROWS, HIDDEN) for _ in range(60)]
    env.pool = FakePool(mk.sliding)
    env.backend.token_to_kv_pool = env.pool
    env.backend.req_to_token_pool = types.SimpleNamespace(
        req_to_token=torch.zeros(2, 1000, dtype=torch.int32)
    )
    eager = metadata(pages=3)  # an eager forward's own metadata
    eager.page_table[0] = torch.tensor([7, 8, 9], dtype=torch.int32)
    eager.swa_out_cache_loc = torch.arange(100, 108)
    env.backend.forward_metadata = eager
    env.args[0].out_cache_loc = torch.arange(300, 308)
    env.mk = mk
    return env


def test_compare_binding(cmp_env):
    env, mk = cmp_env, cmp_env.mk
    assert mk.begin_compare(*env.args), mk.last_refusal
    (sg_md,) = env.builds
    assert sg_md.page_table.shape == (
        1,
        8,
    )  # ceil(1000 / 128) pages: the longest request
    assert (
        sg_md.page_table[0, :3].tolist() == [7, 8, 9]
        and sg_md.kv_lens_cuda_runtime is mk.cmp.seq
    )
    assert mk.cmp.pos32.tolist() == list(range(1000, 1008))
    env.capturing.on = True
    assert not mk.begin_compare(*env.args) and "eager forwards only" in mk.last_refusal


def test_compare_layer(cmp_env):
    env, mk = cmp_env, cmp_env.mk
    assert mk.begin_compare(*env.args)
    launches = []

    def run(
        blocks, x, res, pdl=True
    ):  # the kernel: stream + 1, x as is, rewrites K row 3 of the step
        launches.append(pdl)
        pos = len(launches) - 1
        mk.res_out[pos].copy_(res + 1)
        mk.x_out[pos].copy_(x)
        layer = mk.indices[pos]
        loc = mk.cmp.loc[mk.sliding[layer]]
        env.pool.k[layer][loc[3], 0, 0] = 0x38  # FP8 1.0

    mk.kernels.run_swa = mk.kernels.run_full = run

    def stock(positions, stream, attn_input, batch):
        stream.add_(0)  # in place, as a fused residual update may be
        return stream.clone(), attn_input.clone()

    stream, x = env.args[2], env.args[3]
    for index in mk.indices:
        out = mk.compare_layer(index, stock, env.args[1], stream, x, env.args[0])
        assert (
            out[0] is not mk.res_out[mk.position_of[index]]
        )  # the stock outputs go on
    assert launches == [False] * 60 and mk.current is None and mk.compared == 1
    row1, row3 = mk.cmp_stats[1], mk.cmp_stats[3]
    assert row1["stream_max"] == 1.0 and row1["x_max"] == 0.0 and row1["nonfinite"] == 0
    assert row1["kv_bytes"] == 1 and row1["kv_max"] == 1.0 and row3["kv_bytes"] == 1
    assert (
        env.pool.k[1][103, 0, 0] == 0x38 and env.pool.k[3][303, 0, 0] == 0x38
    )  # SWA rows vs full rows
    (fwd,) = mk.cmp_forwards
    assert (
        fwd["first_position"] == 1000 and fwd["rows"]["1"] == row1 and not mk.cmp_rows
    )
    warmup = torch.zeros(
        ROWS, dtype=torch.int64
    )  # a boot dummy forward: all 8 rows at position 0
    assert (
        not mk.begin_compare(env.args[0], warmup, *env.args[2:])
        and "warm-up" in mk.last_refusal
    )
    assert mk.cmp_warmups == 1 and mk.current is None
    assert mk.begin_compare(*env.args)  # the second of two
    mk.current = None
    mk.compared = 2
    assert not mk.begin_compare(*env.args) and "compared 2" in mk.last_refusal


class FakeRadix(
    torch.nn.Module
):  # RadixAttention: (q, k, v, batch) -> the rank's 8 heads
    scaling = 0.125

    def forward(self, q, k, v, batch, save_kv_cache=True):
        return q * 2


class FakeQkv(
    torch.nn.Module
):  # QKVOGateParallelLinear: (output, bias), SGLang's [q | k | v | g] columns
    def forward(self, x):
        return x[:, :2304] * 1.5, None


class FakeSelfAttn(torch.nn.Module):
    scale_query_by = 5.5

    def __init__(self):
        super().__init__()
        self.qkv_proj, self.attn = FakeQkv(), FakeRadix()

    def forward(self, positions, x, batch):
        q = self.qkv_proj(x)[0][:, :1024]
        return self.attn(q, q[:, :128], q[:, :128], batch).repeat(1, 8)  # "o_proj"


class FakeBoundary(torch.nn.Module):
    def forward(self, stream, branch):
        stream.add_(branch.float())  # in place, as the fused boundary may
        return (stream * 0.5).to(torch.bfloat16), stream, stream * 0.5


class FakeGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(256, HIDDEN) / 64, requires_grad=False
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class FakeExperts(torch.nn.Module):
    def forward(self, x, ar_workspace=None):
        return x * 3


class FakeMlp(torch.nn.Module):
    router_multiplier, route_scale, top_k = 0.17677669529663687, 1.0, 4

    def __init__(self):
        super().__init__()
        self.gate, self.experts = FakeGate(), FakeExperts()
        self.pre_expert_proj = torch.nn.Linear(HIDDEN, 4096, bias=False).to(
            torch.bfloat16
        )
        self.pre_expert_proj.weight.requires_grad_(False)  # SGLang's inference weights

    def forward(self, x, router_input, batch):
        self.gate(router_input)
        m = self.experts(
            torch.nn.functional.linear(x, self.pre_expert_proj.weight),
            ar_workspace=object(),
        )
        return m[
            :, :1024
        ]  # rank 0's unpool columns: the gather deferred to the FFN boundary


class FakeStageLayer(torch.nn.Module):
    """``MuseSpark1xDecoderLayer.forward``'s stages, on the modules ``stages.StockTaps`` hooks."""

    def __init__(self):
        super().__init__()
        self.self_attn, self._fused_attn_boundary, self.mlp = (
            FakeSelfAttn(),
            FakeBoundary(),
            FakeMlp(),
        )
        for name in (
            "post_attn_norm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            setattr(self, name, types.SimpleNamespace(eps=1e-5))

    def forward(self, positions, stream, attn_input, batch):
        x, stream, router_input = self._fused_attn_boundary(
            stream, self.self_attn(positions, attn_input, batch)
        )
        stream = stream + self.mlp(x, router_input, batch).float().repeat(1, 8)
        return stream, stream.to(torch.bfloat16)


def test_compare_stages(cmp_env, tmp_path):
    env, layer = cmp_env, FakeStageLayer()
    runtime, stages = env.mods.runtime, env.mods.stages
    coef = types.SimpleNamespace(
        coefficients=lambda: (torch.ones(HIDDEN), torch.zeros(HIDDEN))
    )
    norm = types.SimpleNamespace(
        weight=torch.zeros(HIDDEN, dtype=torch.bfloat16),
        gain=lambda: torch.ones(HIDDEN),
    )
    ff = types.SimpleNamespace(
        expert_bias=torch.zeros(256),
        pre_expert_norm=types.SimpleNamespace(weight=torch.zeros(4096)),
        post_expert_norm_weight=torch.zeros(4096),
        experts=types.SimpleNamespace(fc31_input_scale=torch.ones(1)),
    )
    facade = types.SimpleNamespace(
        sgl_layer=layer,
        feed_forward=ff,
        add_attn=coef,
        add_ffn=coef,
        pre_ffn_norm=norm,
        post_ffn_norm=norm,
        pre_qkv_proj_norm=norm,
    )
    mk = runtime.SgMegakernel(
        [facade] * 62,
        tp_rank=0,
        local_rank=0,
        comm=None,
        mode=runtime.CHECK_MODE,
        compare=1,
        compare_out=str(tmp_path),
    )
    assert mk.stage_check
    k = mk.kernels = env.mk.kernels
    k.res1 = torch.zeros(ROWS, HIDDEN)
    mk.sliding, mk.x_out, mk.res_out = env.mk.sliding, env.mk.x_out, env.mk.res_out

    def make_swa(attn_args, attn_kwargs, moe, x_out, res_out, **kw):
        return types.SimpleNamespace(
            index=moe, check=kw["check_out"], attn=attn_kwargs.get("attn_out")
        )

    def make_full(w, f_args, f_kwargs, moe, x_out, res_out, **kw):
        return types.SimpleNamespace(index=moe, check=kw["check_out"], attn=None)

    k.make_swa_layer, k.make_full_layer_one, k.make_full_layer = (
        make_swa,
        make_full,
        make_full,
    )

    def build(sg_md, pos32, spec):  # as the vendored _build calls the builders
        return {
            i: (
                k.make_swa_layer((), {}, i, None, None)
                if mk.sliding[i]
                else k.make_full_layer_one(None, (), {}, i, None, None)
            )
            for i in mk.indices
        }

    mk._build = build
    w_pool, w_gate = layer.mlp.pre_expert_proj.weight, layer.mlp.gate.weight

    def kernel(
        blocks, x, res, pdl=True, qkv=None
    ):  # the stock stages, but the pool doubled and the logits multiplied
        f = torch.nn.functional
        if qkv is not None:  # the checkpoint column order, the gate rows negated
            out_qkv = (x[:, :2304] * 1.5).index_select(
                1, stages.checkpoint_cols(x.device)
            )
            out_qkv[:, stages.QKV_BLOCKS["g"]] *= -1
            qkv.copy_(out_qkv)
        attn = (x[:, :1024] * 1.5) * 2
        r1 = res + attn.repeat(1, 8).float()
        xffn = (r1 * 0.5).to(torch.bfloat16)
        pool = f.linear(xffn, w_pool)
        m = pool * 3
        b = m[:, :1024].repeat(1, 8)
        r2 = r1 + b.float()
        out = blocks.check
        for name, value in dict(
            MhOutXffn=xffn,
            MhOutPool=pool * 2,
            MhOutLogits=(f.linear(r1 * 0.5, w_gate) * 0.17677669529663687).t(),
            MhOutM=m,
            MhOutB=b,
            MhOutRes2=r2,
        ).items():
            out[name].copy_(value)
        if blocks.attn is not None:
            blocks.attn.copy_(attn)
        k.res1.copy_(r1)
        pos = mk.position_of[blocks.index]
        mk.res_out[pos].copy_(r2)
        mk.x_out[pos].copy_(r2.to(torch.bfloat16))

    k.run_swa = k.run_full = kernel
    k.swa_mk = types.SimpleNamespace(
        launch=lambda blk, io, pdl: kernel(blk, io.x, io.res, pdl, qkv=io.qkv)
    )
    env.backend.forward_metadata.cache_seqlens_int32.fill_(300)
    assert mk.begin_compare(*env.args), mk.last_refusal
    assert (
        k.make_swa_layer is make_swa and k.make_full_layer_one is make_full
    )  # the builders restored
    stream = torch.randn(ROWS, HIDDEN)
    x = torch.randn(ROWS, HIDDEN).to(torch.bfloat16)
    for index in mk.indices:
        mk.compare_layer(
            index, layer.forward, env.args[1], stream.clone(), x, env.args[0]
        )
    assert mk.compared == 1 and (tmp_path / "compare_rank0.json").exists()
    assert (tmp_path / "forwards_rank0.json").exists()
    row1, row3 = mk.cmp_stats[1], mk.cmp_stats[3]
    for key in (
        "st_q",
        "st_k",
        "st_v",
        "st_attn",
        "st_res1",
        "st_xffn",
        "st_m",
        "st_b",
        "st_res2",
        "stream_max",
    ):
        assert row1[key] < 1e-5, (key, row1[key])
    assert row1["st_logits_fit"] < 1e-5
    assert (
        abs(row1["st_pool"] - 1.0) < 1e-6
        and abs(row1["st_logits"] - (1 - 0.17677669529663687)) < 1e-5
    )
    assert abs(row1["logits_scale"] - 0.17677669529663687) < 1e-6
    assert abs(row1["st_g"] - 2.0) < 1e-6  # the negated gate rows
    assert (
        "st_attn" not in row3 and "st_q" not in row3 and row3["st_res1"] < 1e-5
    )  # a full layer: no attention output
    assert sorted(p.name for p in tmp_path.glob("stages_*.pt")) == [
        f"stages_rank0_layer{i}.pt" for i in (1, 2, 3)
    ]
    blob = torch.load(tmp_path / "stages_rank0_layer1.pt")
    assert (
        blob["layer"] == 1
        and blob["sliding"]
        and blob["scalars"]["softmax_scale"] == 0.125
    )
    assert (
        blob["stock"]["attn_q"].shape == (ROWS, 1024)
        and "m_reduced" not in blob["stock"]
    )
    assert blob["scalars"]["m_reduced"] and blob["params"]["alpha1"].shape == (HIDDEN,)
    assert torch.equal(
        blob["kernel"]["MhOutPool"],
        2 * torch.nn.functional.linear(blob["stock"]["xffn"], w_pool),
    )
    assert torch.equal(blob["inputs"]["res"], stream) and blob["kv"]["seq_len"] == 300
    assert blob["kv"]["k_rows"].shape == (300, 2, 64) and blob["kv"][
        "kv_positions"
    ].tolist() == list(range(300))


# ---- SGLang's KV buffers as the kernel's fields (kv) and the TP group (comm) ----
class FakeFn:
    def __init__(self, fn):
        self.fn, self.restype, self.argtypes = fn, None, None

    def __call__(self, *args):
        return self.fn(*args)


def test_check_build(mods):
    lib = types.SimpleNamespace(
        mode="m", lib=types.SimpleNamespace(msoMkSgKv=FakeFn(lambda: 128))
    )
    mods.kv.check_build(lib)
    lib.lib.msoMkSgKv = FakeFn(lambda: 0)  # a build without MSO_MK_SG_KV
    with pytest.raises(RuntimeError, match="page 0"):
        mods.kv.check_build(lib)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_check_buffer_accepts_sglang_layout(mods, dtype):
    mods.kv.check_buffer(torch.zeros(3 * 128, 2, 64, dtype=dtype))


@pytest.mark.parametrize(
    "buf",
    [
        torch.zeros(256, 128, dtype=torch.uint8),  # flattened rows
        torch.zeros(200, 2, 64, dtype=torch.uint8),  # not whole pages
        torch.zeros(256, 2, 64, dtype=torch.bfloat16),  # BF16 cache
        torch.zeros(256, 2, 128, dtype=torch.uint8)[:, :, :64],  # strided
    ],
)
def test_check_buffer_rejects(mods, buf):
    with pytest.raises(AssertionError):
        mods.kv.check_buffer(buf)


def test_placeholder(mods):
    pool, table = mods.kv.placeholder(torch.device("cpu"))
    assert pool.shape == (1, 2, 32, 64) and pool.dtype == torch.float8_e4m3fn
    assert table.shape == (2, 1) and table.dtype == torch.int32


def test_tensor_map_calls(mods):
    calls = []

    def encode(ptr, slots, out, swizzle):
        calls.append((ptr, slots, swizzle))
        return 0

    lib = types.SimpleNamespace(
        lib=types.SimpleNamespace(msoMkSgKvTensorMap=FakeFn(encode))
    )
    k, v = (
        torch.zeros(256, 2, 64, dtype=torch.uint8),
        torch.zeros(256, 2, 64, dtype=torch.uint8),
    )
    try:
        mods.kv.tensor_maps(lib, k, v, swizzle64=True)
    except (
        AssertionError
    ):  # the 128-B alignment of the device copy: CPU allocations may be 64-B aligned
        pass
    assert calls == [(k.data_ptr(), 256, 1), (v.data_ptr(), 256, 1)]


def test_layer_kv_fields(mods, monkeypatch):
    kv = mods.kv
    monkeypatch.setattr(
        kv,
        "tensor_maps",
        lambda lib, k, v, swizzle64: torch.zeros(256, dtype=torch.uint8),
    )
    k, v = (
        torch.zeros(256, 2, 64, dtype=torch.uint8),
        torch.zeros(256, 2, 64, dtype=torch.uint8),
    )
    table = torch.zeros(1, 4, dtype=torch.int32)
    swa = kv.LayerKv(None, k, v, table, sliding=True).fields()
    assert swa == dict(
        AhPool=k.data_ptr(),
        AhVPool=v.data_ptr(),
        AhBlockTable=table.data_ptr(),
        AhBtStride=0,
        AhTmap=swa["AhTmap"],
    )
    full = kv.LayerKv(None, k, v, table, sliding=False).fields()
    assert (
        set(full) == {"FaPool", "FaVPool", "FaTable", "FaTableStride", "FaTmap"}
        and full["FaTableStride"] == 0
    )
    with pytest.raises(AssertionError):
        kv.LayerKv(None, k, v, torch.zeros(1, 4, dtype=torch.int64), sliding=True)


@pytest.fixture(scope="module")
def group():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


def test_tp_comm(mods, group):
    comm = mods.comm
    c = comm.TpComm(group)
    assert (c.Get_size(), c.Get_rank()) == (1, 0)
    assert c.allgather(("mode", 3)) == [("mode", 3)]
    c.Barrier()
    comm._agree(c, "", "nothing")
    with pytest.raises(RuntimeError, match=r"x failed on ranks \[0\]"):
        comm._agree(c, "out of memory", "x")


def test_mc_view(mods):
    buf = object.__new__(mods.comm.SymmMcastBuffer)
    buf.nbytes, buf.mc_ptr = 4096, 0xABC000
    assert buf.get_mc_buffer((1024,), torch.float32, 0).data_ptr() == 0xABC000
    with pytest.raises(AssertionError):
        buf.get_mc_buffer((1025,), torch.float32, 0)


# ---- the vendored TensorRT-LLM files (vendor/README.md) ----
MODIFIED = {
    "megakernel/library.py",
    "megakernel/msoMegakernel.cu",
    "megakernel/stages/attnHalf.cuh",
    "megakernel/stages/fullAttention.cuh",
}


def test_vendored_files_name_their_origin():
    files = sorted(
        p
        for p in VENDOR.rglob("*")
        if p.suffix in (".py", ".cu", ".cuh")
        and p.parent != VENDOR
        and "__pycache__" not in p.parts
    )
    assert len(files) == 39
    for path in files:
        rel = path.relative_to(VENDOR).as_posix()
        head = path.read_text().split("\n")[:30]
        text = " ".join(line.lstrip("#/ ") for line in head)
        assert (
            f"kernels/{rel} at commit ab93b785a476f01d2f1e1907c930780fc6d7832e" in text
        ), rel
        assert ("Modified by NVIDIA" in text) == (rel in MODIFIED), rel
    assert (
        (VENDOR / "LICENSE.tensorrt_llm")
        .read_text()
        .startswith("Copyright (c) 2011-2026 NVIDIA CORPORATION")
    )
