"""The executor's QSA indexer against SGLang's stock indexer; run on SM10x (the
grouped4 widths score with the plugin's kernel on SM107 only, stock elsewhere).

SGLang's ``QSAIndexer._forward_impl`` and ``runtime/qsa_indexer.select`` run the same
decode-like step, on CUDA-graph metadata, on two copies of one pool. The ring rows, the
RoPE positions and the compressed rows the step writes must be bit-identical
(compressed slot 0 is the inert dump the rows completing no group race on), and each
row must select the same blocks: SGLang's fast_topk orders a row's indices by atomics,
so rows are compared sorted, as the grouped4 gate compares its top-k. Long rows hold
at least 1024 blocks and short ones at most 512: just past 512 a row's 512th score can
be a ReLU zero tied with others, where the selects may break the tie differently.
"""

import copy
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sgl_nvidia_mega.qwen3_8.runtime import qsa_indexer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="explicit SM10x qualification",
)

HEADS, DIM, ROTARY, RATIO, PAGE, BLOCKS, TOKEN_TOPK = 4, 128, 64, 4, 16, 512, 2048
CAPACITY = 16384  # compressed keys a row's page table spans (a 64K-token context)
PAGES = CAPACITY // PAGE
POSITIONS = CAPACITY * RATIO + 16
EPS = 1e-6


def _bits(t):
    return t.contiguous().view(torch.uint8)


class Pool:
    """The QSA pool members the indexer reads; request r owns compressed slots
    [r * CAPACITY, (r + 1) * CAPACITY), its page table rows r * PAGES + k."""

    index_state_dtype = torch.bfloat16
    qsa_compressed_page_size = PAGE
    qsa_index_kv_heads = 1
    qsa_index_head_dim = DIM

    def __init__(self, requests, compressed_dtype, g):
        slots = (requests + 1) * RATIO
        self.qsa_compressed_dtype = compressed_dtype
        self.ring = torch.randn(slots, 1, DIM, generator=g, device="cuda").to(
            torch.bfloat16
        )
        self.qsa_rope_position_buffer = torch.randint(
            0, POSITIONS, (slots, 3), generator=g, device="cuda"
        )
        self.compressed = torch.randn(
            (requests + 1) * CAPACITY, 1, DIM, generator=g, device="cuda"
        ).to(compressed_dtype)

    def get_qsa_key_state_buffer(self, layer_id):
        return self.ring

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed

    def copy(self):
        other = copy.copy(self)
        other.ring = self.ring.clone()
        other.qsa_rope_position_buffer = self.qsa_rope_position_buffer.clone()
        other.compressed = self.compressed.clone()
        return other

    def load(self, other):
        self.ring.copy_(other.ring)
        self.qsa_rope_position_buffer.copy_(other.qsa_rope_position_buffer)
        self.compressed.copy_(other.compressed)


def _indexer(g):
    """SGLang's QSAIndexer with an identity index projection (the hidden rows are qk)."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    def bf16(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    obj = object.__new__(QSAIndexer)
    torch.nn.Module.__init__(obj)
    obj.layer_id = 0
    obj.index_n_heads, obj.index_kv_heads, obj.index_head_dim = HEADS, 1, DIM
    obj.token_topk, obj.compress_ratio, obj.block_topk = TOKEN_TOPK, RATIO, BLOCKS
    obj.rotary_emb = SimpleNamespace(
        rotary_dim=ROTARY,
        cos_sin_cache=torch.randn(POSITIONS, ROTARY, generator=g, device="cuda"),
        is_neox_style=True,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )
    obj.index_qk_proj = lambda x: (x, None)
    obj.q_layernorm = SimpleNamespace(weight=0.1 * bf16(DIM), variance_epsilon=EPS)
    obj.k_layernorm = SimpleNamespace(weight=0.1 * bf16(DIM), variance_epsilon=EPS)
    obj._rope_axis_map_cache = None
    return obj


def _step(mode, requests, pool, g, *, short=(0,), long_prefix=None):
    """One step's graph metadata, the way the backend fills it at replay (``g``: a CPU
    generator): rows of a request are its consecutive positions, a row completing a
    group writes the group's compressed slot, every other row slot 0."""
    from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
    from sglang.srt.model_executor.forward_mode import ForwardMode

    window = 1 if mode is ForwardMode.DECODE else 4
    rows = requests * window
    prefix = torch.randint(
        1024 * RATIO, (CAPACITY - 1) * RATIO - window, (requests,), generator=g
    )
    for i in short:
        if i < requests:
            prefix[i] = torch.randint(0, 512 * RATIO - window, (1,), generator=g)
    if long_prefix is not None:
        prefix[:] = long_prefix
    req = torch.arange(1, requests + 1).repeat_interleave(window)
    pos = prefix.repeat_interleave(window) + torch.arange(window).repeat(requests)
    seq = pos + 1
    members = (pos[:, None] - torch.arange(RATIO - 1, -1, -1)[None, :]).clamp_min(0)
    write = torch.where(seq % RATIO == 0, req * CAPACITY + pos // RATIO, 0)

    def cuda(t, dtype):
        return t.to(dtype=dtype, device="cuda").contiguous()

    md = QSAIndexerMetadata(
        sequence_lengths=cuda(seq, torch.int32),
        token_to_batch_idx=cuda(torch.arange(rows), torch.int32),
        token_slot_table=cuda(torch.zeros(rows, 1), torch.int32),
        out_cache_loc=cuda(torch.zeros(rows), torch.int64),
        token_to_kv_pool=pool,
        compress_ratio=RATIO,
        block_topk=BLOCKS,
        req_pool_indices=cuda(req, torch.int32),
        is_cuda_graph=True,
        graph_write_locs=cuda(write, torch.int32),
        graph_compressed_page_table=cuda(
            req[:, None] * PAGES + torch.arange(PAGES)[None, :], torch.int32
        ),
        graph_compressed_lengths=cuda(seq // RATIO, torch.int32),
        graph_prefix_lengths=cuda(torch.zeros(rows), torch.int32),
        decode_logical_positions=cuda(pos, torch.int32),
        pending_ring_slots=cuda(req * RATIO + pos % RATIO, torch.int64),
        graph_ring_group_locs=cuda(req[:, None] * RATIO + members % RATIO, torch.int32),
        defer_block_expansion=mode is not ForwardMode.DRAFT_EXTEND_V2,
    )
    hidden = torch.randn(rows, (HEADS + 1) * DIM, generator=g).to(
        device="cuda", dtype=torch.bfloat16
    )
    rope = torch.stack([pos, pos // 2, pos // 3]).to(device="cuda")
    fb = SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=torch.arange(1, requests + 1, device="cuda"),
        spec_info=SimpleNamespace(draft_token_num=4),
    )
    return md, hidden, rope, fb


class _Backend:
    def __init__(self, md):
        self.md = md

    def set_mtp_shared_sparse_indices(self, state):
        pass

    def should_reuse_mtp_sparse_indices(self, fb):
        return False

    def should_capture_mtp_sparse_indices(self, fb):
        return False

    def get_indexer_metadata(self, layer_id, fb):
        return self.md


class _St:
    def __init__(self, fb):
        self.fb = fb
        self.record = []

    def ran(self, op, impl):
        self.record.append((op, impl))


def _stock_layer(*args):
    raise AssertionError("the covered step ran SGLang's indexer")


@pytest.fixture
def run(monkeypatch, tmp_path):
    """select() on a fresh IndexerKernels with SGLang's backend lookup faked."""
    monkeypatch.setattr(
        "sglang.srt.models.qwen4_exp._shards_rows_over_cp", lambda fb: False
    )
    kernels = qsa_indexer.IndexerKernels()
    kernels.source_dir = tmp_path
    kernels.sm107 = torch.cuda.get_device_capability() == (10, 7)

    def call(obj, md, hidden, rope, fb):
        backend = _Backend(md)
        monkeypatch.setattr(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            lambda: backend,
        )
        layer = SimpleNamespace(
            is_qsa=True,
            layer_id=0,
            indexer=obj,
            _compute_qsa_topk_indices=_stock_layer,
        )
        st = _St(fb)
        out = qsa_indexer.select(
            SimpleNamespace(qsa_indexer=kernels), layer, hidden, rope, st
        )
        return out, st.record

    call.kernels = kernels
    return call


def _check(out, ref, pool, pool_ref):
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(out.sort(dim=1).values, ref.sort(dim=1).values)
    assert torch.equal(_bits(pool.ring), _bits(pool_ref.ring))
    assert torch.equal(pool.qsa_rope_position_buffer, pool_ref.qsa_rope_position_buffer)
    assert torch.equal(_bits(pool.compressed[1:]), _bits(pool_ref.compressed[1:]))


def _cases():
    from sglang.srt.model_executor.forward_mode import ForwardMode as M

    sm107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
    grouped = "v7" if sm107 else "stock"
    return [
        # mode, requests, compressed dtype, expected engagement
        (M.TARGET_VERIFY, 4, torch.bfloat16, "fold+score_v4+select_v1"),
        (M.TARGET_VERIFY, 16, torch.bfloat16, "fold+score_v4+select_v1"),
        (M.TARGET_VERIFY, 32, torch.bfloat16, f"fold+score_{grouped}+select_v2"),
        (M.TARGET_VERIFY, 144, torch.bfloat16, f"fold+score_{grouped}+select_v2"),
        (M.TARGET_VERIFY, 160, torch.bfloat16, "fold+score_stock+select_v2"),
        (M.TARGET_VERIFY, 16, torch.float8_e4m3fn, "fold+score_stock+select_v1"),
        (M.DECODE, 8, torch.bfloat16, "fold+score_v5+select_v1"),
        (M.DECODE, 64, torch.bfloat16, "fold+score_v5+select_v1"),
        (M.DECODE, 96, torch.bfloat16, "fold+score_stock+select_stock"),
        (M.DRAFT_EXTEND_V2, 8, torch.bfloat16, "fold+score_v5+select_v1"),
        (M.DRAFT_EXTEND_V2, 32, torch.bfloat16, f"fold+score_{grouped}+select_v2"),
    ]


@pytest.mark.parametrize("mode,requests,dtype,impl", _cases())
def test_select_matches_the_stock_indexer(run, mode, requests, dtype, impl):
    g = torch.Generator(device="cuda").manual_seed(requests * 131 + mode.value)
    gc = torch.Generator().manual_seed(requests * 137 + mode.value)
    obj = _indexer(g)
    pool = Pool(requests, dtype, g)
    pool_ref = pool.copy()
    md, hidden, rope, fb = _step(mode, requests, pool, gc)
    md_ref = msgspec.structs.replace(md, token_to_kv_pool=pool_ref)

    ref = obj._forward_impl(hidden, rope, fb, md_ref)
    out, record = run(obj, md, hidden, rope, fb)
    torch.cuda.synchronize()
    assert record == [(qsa_indexer.OP, impl)]
    _check(out, ref, pool, pool_ref)


@pytest.mark.parametrize("prefix", [1100 * RATIO, 9000 * RATIO])
def test_single_request_draft_extend_mk_scorer(run, prefix):
    # The MK scorer's long path takes rows past 8192 blocks; shorter ones score per row.
    from sglang.srt.model_executor.forward_mode import ForwardMode

    g = torch.Generator(device="cuda").manual_seed(prefix)
    obj = _indexer(g)
    pool = Pool(1, torch.bfloat16, g)
    pool_ref = pool.copy()
    md, hidden, rope, fb = _step(
        ForwardMode.DRAFT_EXTEND_V2,
        1,
        pool,
        torch.Generator().manual_seed(prefix),
        short=(),
        long_prefix=prefix + 2,
    )
    md_ref = msgspec.structs.replace(md, token_to_kv_pool=pool_ref)
    ref = obj._forward_impl(hidden, rope, fb, md_ref)
    out, record = run(obj, md, hidden, rope, fb)
    torch.cuda.synchronize()
    assert record == [(qsa_indexer.OP, "fold+score_v0+select_v1")]
    _check(out, ref, pool, pool_ref)


def test_graph_replay_matches_the_stock_indexer(run):
    from sglang.srt.model_executor.forward_mode import ForwardMode

    g = torch.Generator(device="cuda").manual_seed(7)
    obj = _indexer(g)
    pool = Pool(8, torch.bfloat16, g)
    initial = pool.copy()
    md, hidden, rope, fb = _step(
        ForwardMode.TARGET_VERIFY, 8, pool, torch.Generator().manual_seed(7)
    )
    # The graph runner's eager warmup runs the width before capturing it.
    run(obj, md, hidden, rope, fb)
    pool.load(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, record = run(obj, md, hidden, rope, fb)
    assert record == [(qsa_indexer.OP, "fold+score_v4+select_v1")]
    hidden.copy_(torch.randn(hidden.shape, generator=g, device="cuda"))
    pool.load(initial)
    graph.replay()
    pool_ref = initial.copy()
    ref = obj._forward_impl(
        hidden, rope, fb, msgspec.structs.replace(md, token_to_kv_pool=pool_ref)
    )
    torch.cuda.synchronize()
    _check(out, ref, pool, pool_ref)


def test_compile_builds_the_static_layouts(tmp_path):
    g = torch.Generator(device="cuda").manual_seed(3)
    kernels = qsa_indexer.IndexerKernels()
    kernels.source_dir = tmp_path
    kernels.compile(_indexer(g), context_len=CAPACITY * RATIO)
    assert ("fold", torch.bfloat16, torch.bfloat16, DIM, True) in kernels._modules
    splits = {key for key in kernels._modules if key[0] == "split"}
    assert {key[1:4] for key in splits} == {
        qsa_indexer.split_config(rows, CAPACITY) for rows in range(1, 65)
    }
    assert {key[0] for key in kernels._kernels} == {"v0", "v4", "v5"}
    assert sorted(p.name.split("_")[1] for p in tmp_path.glob("*.cuh")) == [
        "cluster",
        "prologue",
    ]
