# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass
from .contract import BuildConfig, PreparedLayer


COPIES = 3
ARENA_WORDS = 617440
HOP_WORDS = 393216
TK_WORDS = 16384
LG_WORDS = 32768
Y_WORDS = 8 * 1312
COUNTER_WORDS = 800
CAND_LIST = 1 << 20
CAND_CAP = CAND_LIST + 8192
CAND_WORDS = COPIES * 8 * CAND_CAP * 2
TRANSPORT_WORDS = COPIES * (2 * HOP_WORDS + TK_WORDS + LG_WORDS + Y_WORDS)
BUFFER_FIELDS = (
    ('arena', 'float32', COPIES * ARENA_WORDS, 'negative_zero', 'Three-copy packed arena and interlayer carry'),
    ('part_o', 'float32', 262144, 'negative_zero', 'Same-generation MLA partials, last-reader merge reset'),
    ('part_stats', 'float32', 1024, 'negative_zero', 'Same-generation MLA maxima/sums, merge reset'),
    ('fc2out', 'float32', 64 * 8 * 3072, 'negative_zero', 'Packed BF16 routed outputs'),
    ('counters', 'int32', COPIES * COUNTER_WORDS, 'zero', 'Source release/acquire three-copy counters'),
    ('fc2_lists', 'int32', 4160 * 3, 'zero', 'FC2 reduction lists; the parent writes schedule.fc2_lists_payload()'),
)


@dataclass(frozen=True)
class SharedWorkspace:

    buffers: dict[str, object]
    uc: object
    mc: object
    build: BuildConfig


def validate_build(build: BuildConfig) -> None:
    if type(build.include_indexer) is not bool:
        raise TypeError('include_indexer must be bool')
    if (build.arch, build.static_flags, build.slot_mode) != ('sm_107a', 2752, 'reference_logical'):
        raise ValueError('Pinned specialization and logical-slot contract required')


def workspace_request(build: BuildConfig) -> dict[str, object]:
    validate_build(build)
    fields = BUFFER_FIELDS
    if build.include_indexer:
        fields += (('idx_candidates', 'int32', CAND_WORDS, 'zero',
                    'Three-copy eight-row candidate pairs plus RS-H histogram and CTA completion words'),)
    return {
        'schema': 'typed-shared-workspace.v1',
        'buffers': [dict(zip(('name', 'dtype', 'elements', 'initial', 'purpose'), row)) for row in fields],
        'multicast_bytes': 4 * TRANSPORT_WORDS,
    }


def tensor_elements(tensor: object) -> int:
    count = 1
    for size in tensor.shape:
        count *= int(size)
    return count


def require_compact(tensor: object, type_name: str, elements: int | None = None) -> None:
    if tensor.element_type.__name__ != type_name:
        raise TypeError('Unexpected source operand element type: ' + type_name)
    if len(tensor.shape) != len(tensor.stride):
        raise ValueError('Source owner shape and stride rank differ')
    count = 1
    for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride)):
        if int(size) <= 0:
            raise ValueError('Source owner dimensions must be positive')
        if int(size) > 1 and int(stride) != count:
            raise ValueError('Source raw owner must cover compact physical storage')
        count *= int(size)
    if elements is not None and count != elements:
        raise ValueError('Source operand extent differs')
    address = int(tensor.data_ptr)
    if not 0 < address < 2**64:
        raise ValueError('Source owner address must be a nonnull unsigned64 pointer')
    if address % 16:
        raise ValueError('Vector/bulk operands require actual16B alignment')

def prepare_shared(shared: object, build: BuildConfig) -> SharedWorkspace:
    validate_build(build)
    expected = workspace_request(build)
    if shared.spec != expected:
        raise ValueError('Parent workspace specification differs from source plan')
    if set(shared.buffers) != {row['name'] for row in expected['buffers']}:
        raise ValueError('Actual declared private workspace owners required')
    for row in expected['buffers']:
        require_compact(shared.buffers[row['name']], {'float32': 'Float32', 'int32': 'Int32'}[row['dtype']], row['elements'])
    if shared.uc is None or shared.mc is None:
        raise ValueError('Actual framework rank16 UC/MC mappings required')
    require_compact(shared.uc, 'Float32', TRANSPORT_WORDS)
    require_compact(shared.mc, 'Float32', TRANSPORT_WORDS)
    return SharedWorkspace(dict(shared.buffers), shared.uc, shared.mc, build)


def _prepare_layer_metadata(weights: dict[str, object], cache: dict[str, object], build: BuildConfig) -> PreparedLayer:
    validate_build(build)
    if type(cache.get('leader')) is not bool:
        raise ValueError('Explicit layer kind required')
    if cache.get('tp_size') != 16 or type(cache.get('tp_rank')) is not int or not 0 <= cache['tp_rank'] < 16:
        raise ValueError('Actual TP16 rank metadata required')
    if type(cache.get('copy')) is not int or not 0 <= cache['copy'] < COPIES:
        raise ValueError('Three-copy identity required')
    if type(cache.get('layer_id')) is not int or not 3 <= cache['layer_id'] <= 77:
        raise ValueError('Actual selected model layer identity required')
    shared = cache.get('candidate_shared')
    if not isinstance(shared, SharedWorkspace) or shared.build != build:
        raise ValueError('Prepared source-bound workspace required')
    pool = cache.get('mla_pool')
    require_compact(pool, 'Uint8')
    if type(cache.get('mla_base')) is not int or cache['mla_base'] != int(pool.data_ptr):
        raise ValueError('Raw576 pool owner must begin at actual source base; layer-offset alias is insufficient')
    if type(cache.get('stride_rows')) is not int or cache['stride_rows'] < 64 or cache['stride_rows'] % 64:
        raise ValueError('Page stride must be a positive multiple of64 rows')
    if type(cache.get('layer_rows')) is not int or cache['layer_rows'] < 0 or cache['layer_rows'] + 64 > cache['stride_rows']:
        raise ValueError('Layer rows must fit each runtime page')
    required = {'w_fa', 'w_qb', 'w_o', 'w_lg', 'w_gu', 'w_d', 'w13', 'w2', 'alpha1', 'alpha2',
                'gamma_kv', 'gamma_post', 'gamma_next', 'router_bias', 'rope'}
    if not required <= weights.keys():
        raise ValueError('Missing actual packed plain-layer weights')
    return PreparedLayer(dict(weights), dict(cache), shared, (), build)


def prepare_layer(weights: dict[str, object], cache: dict[str, object], build: BuildConfig) -> PreparedLayer:
    validate_build(build)
    if cache.get('leader') is True:
        return prepare_leader_for_qualification(weights, cache, build)
    if cache.get('leader') is not False:
        raise ValueError('Explicit layer kind required')
    return _prepare_layer_metadata(weights, cache, build)


def prepare_leader_for_qualification(weights: dict[str, object], cache: dict[str, object],
                                     build: BuildConfig) -> PreparedLayer:
    prepared = _prepare_layer_metadata(weights, cache, build)
    leader_metadata_request(weights, cache, build)
    return prepared


def leader_metadata_request(weights: dict[str, object], cache: dict[str, object],
                            build: BuildConfig) -> dict[str, object]:
    validate_build(build)
    if not build.include_indexer or cache.get('leader') is not True:
        raise ValueError('Leader metadata requires explicit indexer-enabled layer')
    shared = cache.get('candidate_shared')
    if not isinstance(shared, SharedWorkspace) or shared.build != build or 'idx_candidates' not in shared.buffers:
        raise ValueError('Actual source-bound candidate workspace required')
    required = {'w_idx', 'idx_kn_w', 'idx_kn_b', 'idx_rope'}
    if not required <= weights.keys():
        raise ValueError('Missing actual packed indexer weights')
    require_compact(weights['w_idx'], 'Uint8', 256 * 32 * 2048)
    for name in ('idx_kn_w', 'idx_kn_b'):
        require_compact(weights[name], 'Float32', 128)
    require_compact(weights['idx_rope'], 'Float32')
    if tensor_elements(weights['idx_rope']) == 0 or tensor_elements(weights['idx_rope']) % 64:
        raise ValueError('Indexer RoPE requires complete64-Float32 rows')
    pool, blocks = cache.get('idx_pool'), cache.get('idx_blocks')
    if pool is None or blocks is None:
        raise ValueError('Actual independent indexer pool and table required')
    require_compact(pool, 'Uint8')
    require_compact(blocks, 'Int32')
    if type(cache.get('idx_base')) is not int or cache['idx_base'] != int(pool.data_ptr):
        raise ValueError('Indexer raw owner must start at actual layer base')
    stride, pages = cache.get('idx_stride_bytes'), cache.get('idx_pages')
    if (type(stride) is not int or type(pages) is not int
            or not 8448 <= stride < 2**63 or stride % 128 or not 0 < pages < 2**63):
        raise ValueError('Indexer physical page metadata is invalid')
    span = (pages - 1) * stride + 8448
    size = tensor_elements(pool)
    if span >= 2**63 or size >= 2**63 or size < span or size % 128:
        raise ValueError('Indexer raw owner cannot cover full physical page span')
    if tensor_elements(blocks) <= 0:
        raise ValueError('Indexer page table must not be empty')
    return {
        'w_idx': weights['w_idx'], 'idx_kn_w': weights['idx_kn_w'],
        'idx_kn_b': weights['idx_kn_b'], 'idx_rope': weights['idx_rope'],
        'idx_pool': pool, 'idx_blocks': blocks, 'idx_stride_bytes': stride,
        'idx_pages': pages, 'idx_candidates': shared.buffers['idx_candidates'],
    }
