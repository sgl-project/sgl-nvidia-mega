# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor, make_fake_stream
from .contract import BuildConfig, LayerInputs, LayerOutputs, PreparedLayer


ARGUMENT_CONTRACT = [
    {
        "role": "plain_parent_x_raw",
        "kind": "tensor",
        "owner": "caller"
    },
    {
        "role": "plain_parent_residual_raw",
        "kind": "tensor",
        "owner": "caller"
    },
    {
        "role": "plain_parent_residual_out_raw",
        "kind": "tensor",
        "owner": "caller"
    },
    {
        "role": "plain_parent_normalized_raw",
        "kind": "tensor",
        "owner": "caller"
    },
    {
        "role": "plain_parent_wfa_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_wqb_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_wo_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_wlg_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_wgu_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_wd_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_w1_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_w2_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_gamma_kv",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_gamma_post_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_gamma_next_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_bias_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_alpha1",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_alpha2",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_rope_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_length",
        "kind": "tensor",
        "owner": "caller"
    },
    {
        "role": "plain_parent_table_raw",
        "kind": "tensor",
        "owner": "caller"
    },
    {
        "role": "plain_parent_cache_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_topk_raw",
        "kind": "tensor",
        "owner": "caller"
    },
    {
        "role": "plain_parent_arena_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_partials_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_stats_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_fc2_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_counters_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_lists_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_uc_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_mc_raw",
        "kind": "tensor",
        "owner": "prepared-layer"
    },
    {
        "role": "plain_parent_copy",
        "kind": "scalar",
        "owner": "caller"
    },
    {
        "role": "plain_parent_rank",
        "kind": "scalar",
        "owner": "caller"
    },
    {
        "role": "plain_parent_early_x",
        "kind": "scalar",
        "owner": "caller"
    },
    {
        "role": "plain_parent_x_only",
        "kind": "scalar",
        "owner": "caller"
    },
    {
        "role": "plain_parent_stride_rows",
        "kind": "scalar",
        "owner": "caller"
    },
    {
        "role": "plain_parent_layer_rows",
        "kind": "scalar",
        "owner": "caller"
    },
    {
        "role": "plain_parent_max_blocks",
        "kind": "scalar",
        "owner": "caller"
    },
    {
        "role": "plain_parent_clear_first",
        "kind": "scalar",
        "owner": "caller"
    },
    {'role': 'leader_widx_raw_unused_in_plain_specialization', 'kind': 'tensor', 'owner': 'prepared-layer'},
    {'role': 'leader_idx_kn_w_unused_in_plain_specialization', 'kind': 'tensor', 'owner': 'prepared-layer'},
    {'role': 'leader_idx_kn_b_unused_in_plain_specialization', 'kind': 'tensor', 'owner': 'prepared-layer'},
    {'role': 'leader_idx_rope_raw_unused_in_plain_specialization', 'kind': 'tensor', 'owner': 'prepared-layer'},
    {'role': 'leader_idx_pool_raw_unused_in_plain_specialization', 'kind': 'tensor', 'owner': 'prepared-layer'},
    {'role': 'leader_idx_blocks_raw_unused_in_plain_specialization', 'kind': 'tensor', 'owner': 'prepared-layer'},
    {'role': 'leader_idx_stride_bytes_unused_in_plain_specialization', 'kind': 'scalar', 'owner': 'prepared-layer'},
    {'role': 'leader_idx_pages_unused_in_plain_specialization', 'kind': 'scalar', 'owner': 'prepared-layer'},
    {'role': 'leader_candidate_pairs_histogram_and_completion_words_unused_in_plain', 'kind': 'tensor', 'owner': 'private-workspace'},
    {'role': 'actual_layer_kind_leader_front_compile_only_until_selection_ready', 'kind': 'constexpr', 'owner': 'prepared-layer'},
    {
        "role": "actual_caller_current_stream",
        "kind": "stream",
        "owner": "caller"
    },
    {
        "role": "programmatic_stream_serialization_policy",
        "kind": "constexpr",
        "owner": "caller"
    }
]


def compile_request(build: BuildConfig, pdl: bool) -> dict[str, object]:
    validate_build(build)
    if type(pdl) is not bool:
        raise TypeError('pdl must be bool')
    return {
        'entry': decoder_entry,
        'args': (
            make_fake_tensor(cutlass.BFloat16, (8, 6144), (6144, 1)),
            make_fake_tensor(cutlass.BFloat16, (8, 6144), (6144, 1)),
            make_fake_tensor(cutlass.BFloat16, (8, 6144), (6144, 1)),
            make_fake_tensor(cutlass.BFloat16, (8, 6144), (6144, 1)),
            make_fake_tensor(cutlass.Uint8, (164, 96, 2048),
                             (196608, 2048, 1)),
            make_fake_tensor(cutlass.Uint8, (144, 32, 2048), (65536, 2048, 1)),
            make_fake_tensor(cutlass.Uint8, (384, 32, 2048), (65536, 2048, 1)),
            make_fake_tensor(cutlass.Uint8, (16, 38, 2048), (77824, 2048, 1)),
            make_fake_tensor(cutlass.Uint8, (16, 96, 2048), (196608, 2048, 1)),
            make_fake_tensor(cutlass.Uint8, (384, 2, 2048), (4096, 2048, 1)),
            make_fake_tensor(cutlass.Uint8, (256, 2, 8, 55296), (884736, 442368, 55296, 1)),
            make_fake_tensor(cutlass.Uint8, (256, 8, 55296), (442368, 55296, 1)),
            make_fake_tensor(cutlass.BFloat16, (512,), (1,)),
            make_fake_tensor(cutlass.BFloat16, (6144,), (1,)),
            make_fake_tensor(cutlass.BFloat16, (6144,), (1,)),
            make_fake_tensor(cutlass.BFloat16, (256,), (1,)),
            make_fake_tensor(cutlass.Float32, (257,), (1,)),
            make_fake_tensor(cutlass.Float32, (257,), (1,)),
            make_fake_tensor(cutlass.Float32, (cute.sym_int64(symbol="rope_rows"), 128), (128, 1)),
            make_fake_tensor(cutlass.Int32, (1,), (1,)),
            make_fake_tensor(cutlass.Int32, (cute.sym_int64(symbol="page_table_rows"),), (1,)),
            make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol="cache_rows"), 576), (576, 1)),
            make_fake_tensor(cutlass.Int32, (8, 2048), (2048, 1)),
            make_fake_tensor(cutlass.Float32, (1852320,), (1,)),
            make_fake_tensor(cutlass.Float32, (262144,), (1,)),
            make_fake_tensor(cutlass.Float32, (1024,), (1,)),
            make_fake_tensor(cutlass.Float32, (1572864,), (1,)),
            make_fake_tensor(cutlass.Int32, (2400,), (1,)),
            make_fake_tensor(cutlass.Int32, (12480,), (1,)),
            make_fake_tensor(cutlass.Float32, (TRANSPORT_WORDS,), (1,)),
            make_fake_tensor(cutlass.Float32, (TRANSPORT_WORDS,), (1,)),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Boolean(False),
            cutlass.Boolean(False),
            cutlass.Int64(64),
            cutlass.Int64(0),
            cutlass.Int32(1024),
            cutlass.Boolean(False),
            make_fake_tensor(cutlass.Uint8, (256, 32, 2048), (65536, 2048, 1)),
            make_fake_tensor(cutlass.Float32, (128,), (1,)),
            make_fake_tensor(cutlass.Float32, (128,), (1,)),
            make_fake_tensor(cutlass.Float32, (cute.sym_int64(symbol="idx_rope_rows"), 64), (64, 1)),
            make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol="idx_pool_pages"), 8448), (8448, 1)),
            make_fake_tensor(cutlass.Int32, (cute.sym_int64(symbol="idx_table_rows"),), (1,)),
            cutlass.Int64(8448),
            cutlass.Int64(1024),
            make_fake_tensor(cutlass.Int32, (CAND_WORDS if build.include_indexer else 2400,), (1,)),
            False,
            make_fake_stream(),
            pdl,
        ),
        'argument_contract': ARGUMENT_CONTRACT,
    }


def _bind_entry_metadata(prepared: PreparedLayer, inputs: LayerInputs, outputs: LayerOutputs,
                         stream: object, pdl: bool, leader: bool) -> dict[str, object]:
    validate_build(prepared.build)
    if type(pdl) is not bool:
        raise TypeError('pdl must be bool')
    w, c, shared = prepared.weights, prepared.cache, prepared.workspace
    if c.get('leader') is not leader:
        raise ValueError('Prepared layer kind differs from binding request')
    indexer = leader_metadata_request(w, c, prepared.build) if leader else None
    topk = outputs.topk if leader else inputs.topk
    b = shared.buffers
    early, x_only = c.get('chained'), c.get('next_chained')
    if type(early) is not bool or type(x_only) is not bool:
        raise TypeError('Source chaining flags must be explicit booleans')
    if (inputs.x is None) != early:
        raise ValueError('Only chained consumers may use the actual arena carry')
    if (outputs.normalized is None) and not x_only:
        raise ValueError('Source-valid normalized output owner is required')
    if inputs.residual is None or outputs.residual is None or topk is None:
        raise ValueError('Actual residual owners and logical topk are required')
    if tensor_elements(topk) != 8 * 2048 or tensor_elements(inputs.sequence_lengths) != 1:
        raise ValueError('Pinned C1/M8 logical metadata extents required')
    if tensor_elements(inputs.residual) != 8 * 6144 or tensor_elements(outputs.residual) != 8 * 6144:
        raise ValueError('Actual residual tensor extents differ')
    for tensor in (inputs.residual, outputs.residual):
        require_compact(tensor, 'BFloat16', 8 * 6144)
    if inputs.x is not None:
        require_compact(inputs.x, 'BFloat16', 8 * 6144)
    if outputs.normalized is not None:
        require_compact(outputs.normalized, 'BFloat16', 8 * 6144)
    require_compact(topk, 'Int32', 8 * 2048)
    require_compact(inputs.sequence_lengths, 'Int32', 1)
    pool, table = c['mla_pool'], c['block_table']
    require_compact(pool, 'Uint8')
    require_compact(table, 'Int32')
    weight_bytes = {'w_fa': (174 if leader else 164) * 196608, 'w_qb': 9437184, 'w_o': 25165824,
                    'w_lg': 1245184, 'w_gu': 3145728, 'w_d': 1572864,
                    'w13': 226492416, 'w2': 113246208}
    for name, count in weight_bytes.items():
        require_compact(w[name], 'Uint8', count)
    for name, count in (('gamma_kv', 512), ('gamma_post', 6144), ('gamma_next', 6144), ('router_bias', 256)):
        require_compact(w[name], 'BFloat16', count)
    for name in ('alpha1', 'alpha2'):
        require_compact(w[name], 'Float32', 257)
    require_compact(w['rope'], 'Float32')
    if tensor_elements(w['rope']) % 128:
        raise ValueError('Interleaved RoPE rows require128Float32 elements')
    if tensor_elements(pool) % 576:
        raise ValueError('Raw576 pool owner has a non-row byte extent')
    max_blocks = tensor_elements(table)
    required_rows = c['layer_rows'] + 64
    if max_blocks <= 0 or required_rows * 576 > tensor_elements(pool):
        raise ValueError('Base-covering raw576 allocation is too small for runtime pages')
    if leader:
        rope_rows = tensor_elements(w['rope']) // 128
        pages, stride = indexer['idx_pages'], indexer['idx_stride_bytes']
        if tensor_elements(indexer['idx_rope']) != rope_rows * 64:
            raise ValueError('Indexer RoPE rows differ from the MLA RoPE rows')
        if tensor_elements(indexer['idx_blocks']) < rope_rows // 64:
            raise ValueError('Indexer block table shorter than the RoPE position range')
        if pages <= 0 or (pages - 1) * stride + 8448 > tensor_elements(indexer['idx_pool']) \
                or tensor_elements(indexer['idx_pool']) % 128:
            raise ValueError('Indexer pool smaller than its page range')
        if tensor_elements(indexer['idx_kn_w']) != 128 or tensor_elements(indexer['idx_kn_b']) != 128:
            raise ValueError('Indexer key-norm extents differ')
        if tensor_elements(w['w_fa']) != 174 * 196608 or tensor_elements(indexer['w_idx']) != 256 * 65536:
            raise ValueError('Leader Fa / indexer weight extents differ')
    return {
        'entry': decoder_entry,
        'args': (
            inputs.x if inputs.x is not None else b['arena'],
            inputs.residual, outputs.residual,
            outputs.normalized if outputs.normalized is not None else b['arena'],
            w['w_fa'], w['w_qb'], w['w_o'], w['w_lg'], w['w_gu'], w['w_d'], w['w13'], w['w2'],
            w['gamma_kv'], w['gamma_post'], w['gamma_next'], w['router_bias'],
            w['alpha1'], w['alpha2'], w['rope'],
            inputs.sequence_lengths, table, pool, topk,
            b['arena'], b['part_o'], b['part_stats'], b['fc2out'], b['counters'], b['fc2_lists'],
            shared.uc, shared.mc, cutlass.Int32(c['copy']), cutlass.Int32(c['tp_rank']),
            cutlass.Boolean(early), cutlass.Boolean(x_only), cutlass.Int64(c['stride_rows']),
            cutlass.Int64(c['layer_rows']), cutlass.Int32(max_blocks), cutlass.Boolean(False),
            indexer['w_idx'] if leader else w['w_qb'],
            indexer['idx_kn_w'] if leader else w['alpha1'],
            indexer['idx_kn_b'] if leader else w['alpha2'],
            indexer['idx_rope'] if leader else w['rope'],
            indexer['idx_pool'] if leader else pool,
            indexer['idx_blocks'] if leader else table,
            cutlass.Int64(indexer['idx_stride_bytes'] if leader else -8448),
            cutlass.Int64(indexer['idx_pages'] if leader else -1),
            b.get('idx_candidates', b['counters']), leader, stream, pdl,
        ),
        'argument_contract': ARGUMENT_CONTRACT,
    }


def runtime_request(prepared: PreparedLayer, inputs: LayerInputs, outputs: LayerOutputs,
                    stream: object, pdl: bool) -> dict[str, object]:
    leader = prepared.cache.get('leader')
    if type(leader) is not bool:
        raise ValueError('Explicit layer kind required')
    return _bind_entry_metadata(prepared, inputs, outputs, stream, pdl, leader)


def leader_qualification_request(prepared: PreparedLayer, inputs: LayerInputs, outputs: LayerOutputs,
                                  stream: object, pdl: bool) -> dict[str, object]:
    return _bind_entry_metadata(prepared, inputs, outputs, stream, pdl, True)


def launch(prepared: PreparedLayer, inputs: LayerInputs, outputs: LayerOutputs, stream: object) -> None:
    raise NotImplementedError('Trusted serialized parent launcher must consume runtime_request')


# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .decoder_entry import decoder_entry
from .preparation import (
    validate_build,
    tensor_elements,
    require_compact,
    CAND_WORDS,
    leader_metadata_request,
    TRANSPORT_WORDS,
)
