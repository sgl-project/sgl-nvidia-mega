# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (Apache-2.0, SGLang Team): python/sglang/srt/layers/attention/qsa/mqa.py.
# Changed: only the grouped scorers SGLang does not have; the per-row scorer and the
# prefill kernels are SGLang's (``sglang.srt.layers.attention.qsa.mqa``).
"""The QSA indexer's grouped TileLang decode scorers: the v7 grouped scorer (one K-tile
load per group of 4 rows) and the 4-row chain and pipelined variants."""

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped_v7_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        rows_per_group: int = 4,
        ctas_per_group: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        """The v3 grouped scorer (one K-tile load per group of rows, per-row
        query buffers, the per-row kernel's exact GEMM / ReLU / reduce_sum /
        scale) with a shorter dependency chain: CTA ``cta`` scores tiles
        cta, cta + C, ... and reads its first tile's page-table entries
        together with the context lengths, so the chain is
        max(length, page table) -> K (+ queries) instead of
        length -> queries -> page table -> K. Nothing but those four entries is
        read before the length check, and sub-pages past the length re-load the
        tile's first page (never stored). The second tile's (cta + C)
        page-table entries are read up front as well, so a CTA's second tile
        costs one K round trip."""
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        W = rows_per_group
        assert 1 <= W <= 4, W
        C = ctas_per_group
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.macro
        def load_tile(KCache, PageTable, kbuf, last, tile, last_page):
            for sp in T.unroll(sub_pages):
                T.copy(
                    KCache[
                        PageTable[last, T.min(tile * sub_pages + sp, last_page)],
                        :,
                        0,
                        :,
                    ],
                    kbuf[sp * page_size : (sp + 1) * page_size, :],
                )

        @T.macro
        def score_row(Logits, kbuf, qbuf, scores, reduced, row, row_len, tile, Scale):
            T.gemm(
                kbuf,
                qbuf,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            for token, head in T.Parallel(GROUP, heads):
                scores[token, head] = T.max(scores[token, head], 0.0)
            T.reduce_sum(scores, reduced, dim=1, clear=True)
            for token in T.Parallel(GROUP):
                position = tile * GROUP + token
                if position < row_len:
                    Logits[row, position] = reduced[token] / Scale

        @T.macro
        def score_tile(
            Logits,
            kbuf,
            q0,
            q1,
            q2,
            q3,
            scores,
            reduced,
            gx,
            l0,
            l1,
            l2,
            l3,
            tile,
            Scale,
        ):
            score_row(Logits, kbuf, q0, scores, reduced, gx * W, l0, tile, Scale)
            if W > 1:
                score_row(
                    Logits, kbuf, q1, scores, reduced, gx * W + 1, l1, tile, Scale
                )
            if W > 2:
                score_row(
                    Logits, kbuf, q2, scores, reduced, gx * W + 2, l2, tile, Scale
                )
            if W > 3:
                score_row(
                    Logits, kbuf, q3, scores, reduced, gx * W + 3, l3, tile, Scale
                )

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),
            PageTable: T.Tensor([batch, max_pages], T.int32),
            ContextLens: T.Tensor([batch], T.int32),
            Logits: T.Tensor([batch, max_model_len], T.float32),
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, W), C, threads=threads) as (gx, cta):
                q0 = T.alloc_shared([heads, head_dim], dtype)
                q1 = T.alloc_shared([heads, head_dim], dtype)
                q2 = T.alloc_shared([heads, head_dim], dtype)
                q3 = T.alloc_shared([heads, head_dim], dtype)
                k0 = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                group_len = T.alloc_var(T.int32)
                last = gx * W + (W - 1)
                pidx = T.alloc_local([sub_pages], T.int32)
                pidx2 = T.alloc_local([sub_pages], T.int32)
                for sp in T.unroll(sub_pages):
                    pidx[sp] = PageTable[
                        last, T.min(cta * sub_pages + sp, max_pages - 1)
                    ]
                for sp in T.unroll(sub_pages):
                    pidx2[sp] = PageTable[
                        last, T.min((cta + C) * sub_pages + sp, max_pages - 1)
                    ]
                group_len = ContextLens[last]
                num_tiles = T.ceildiv(group_len, GROUP)
                last_page = T.max(group_len - 1, 0) // page_size
                l0 = T.alloc_var(T.int32)
                l1 = T.alloc_var(T.int32)
                l2 = T.alloc_var(T.int32)
                l3 = T.alloc_var(T.int32)
                l0 = ContextLens[gx * W]
                l1 = l0
                l2 = l0
                l3 = l0
                if W > 1:
                    l1 = ContextLens[gx * W + 1]
                if W > 2:
                    l2 = ContextLens[gx * W + 2]
                if W > 3:
                    l3 = ContextLens[gx * W + 3]
                if group_len > min_scored_len and cta < num_tiles:
                    for sp in T.unroll(sub_pages):
                        T.copy(
                            KCache[
                                T.if_then_else(
                                    cta * sub_pages + sp <= last_page, pidx[sp], pidx[0]
                                ),
                                :,
                                0,
                                :,
                            ],
                            k0[sp * page_size : (sp + 1) * page_size, :],
                        )
                    T.copy(Q[gx * W, 0, :, :], q0)
                    if W > 1:
                        T.copy(Q[gx * W + 1, 0, :, :], q1)
                    if W > 2:
                        T.copy(Q[gx * W + 2, 0, :, :], q2)
                    if W > 3:
                        T.copy(Q[gx * W + 3, 0, :, :], q3)
                    for it in T.serial(T.ceildiv(num_tiles - cta, C)):
                        tile = cta + it * C
                        if it == 1:
                            for sp in T.unroll(sub_pages):
                                T.copy(
                                    KCache[
                                        T.if_then_else(
                                            tile * sub_pages + sp <= last_page,
                                            pidx2[sp],
                                            pidx2[0],
                                        ),
                                        :,
                                        0,
                                        :,
                                    ],
                                    k0[sp * page_size : (sp + 1) * page_size, :],
                                )
                        else:
                            if it > 1:
                                load_tile(KCache, PageTable, k0, last, tile, last_page)
                        score_tile(
                            Logits,
                            k0,
                            q0,
                            q1,
                            q2,
                            q3,
                            scores,
                            reduced,
                            gx,
                            l0,
                            l1,
                            l2,
                            l3,
                            tile,
                            Scale,
                        )

        return kernel

    # The grouped4 kernel with the v7 kernel's shorter load chain: the length and the
    # first page-table entries are read together, the key tile before the queries,
    # and sub-pages past the length re-read the tile's first page (never stored).
    # Every score is grouped4's GEMM, ReLU, head sum and scale on the same tile and
    # rows, so the logits are bit-identical.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped4_chain_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        ctas_per_row: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        GROUP = 64
        group_rows = 4
        score_heads = 4
        assert heads == 8
        assert GROUP % page_size == 0
        sub_pages = GROUP // page_size
        C = ctas_per_row
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, group_rows), C, threads=threads) as (
                request,
                cta,
            ):
                row_base = request * group_rows
                q_shared = T.alloc_shared([group_rows, score_heads, head_dim], dtype)
                q_gemm = T.reshape(q_shared, (group_rows * score_heads, head_dim))
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, group_rows * score_heads], T.float32)
                scores_3d = T.reshape(scores, (GROUP, group_rows, score_heads))
                reduced = T.alloc_fragment([GROUP, group_rows], T.float32)
                context_len = T.alloc_var(T.int32)
                table_row = T.alloc_var(T.int32)
                pidx = T.alloc_local([sub_pages], T.int32)
                # Graph verify rows share page prefixes and have increasing lengths.
                table_row = T.min(row_base + group_rows - 1, batch - 1)
                for sp in T.unroll(sub_pages):
                    pidx[sp] = PageTable[
                        table_row, T.min(cta * sub_pages + sp, max_pages - 1)
                    ]
                context_len = ContextLens[table_row]
                num_groups = T.ceildiv(context_len, GROUP)
                last_page = T.max(context_len - 1, 0) // page_size
                if context_len > min_scored_len and cta < num_groups:
                    for sp in T.unroll(sub_pages):
                        T.copy(
                            KCache[
                                T.if_then_else(
                                    cta * sub_pages + sp <= last_page, pidx[sp], pidx[0]
                                ),
                                :,
                                0,
                                :,
                            ],
                            k_shared[sp * page_size : (sp + 1) * page_size, :],
                        )
                    T.copy(
                        Q[
                            row_base : row_base + group_rows,
                            0,
                            0:score_heads,
                            0:head_dim,
                        ],
                        q_shared,
                    )
                    for step in T.serial(T.ceildiv(num_groups - cta, C)):
                        group = step * C + cta
                        if step > 0:
                            for sp in T.unroll(sub_pages):
                                T.copy(
                                    KCache[
                                        PageTable[
                                            table_row,
                                            T.min(group * sub_pages + sp, last_page),
                                        ],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[sp * page_size : (sp + 1) * page_size, :],
                                )
                        T.gemm(
                            k_shared,
                            q_gemm,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, qi, head in T.Parallel(
                            GROUP, group_rows, score_heads
                        ):
                            scores_3d[token, qi, head] = T.max(
                                scores_3d[token, qi, head], 0.0
                            )
                        T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                        for token, qi in T.Parallel(GROUP, group_rows):
                            position = group * GROUP + token
                            if row_base + qi < batch:
                                if (
                                    ContextLens[row_base + qi] > min_scored_len
                                    and position < ContextLens[row_base + qi]
                                ):
                                    Logits[row_base + qi, position] = (
                                        reduced[token, qi] / Scale
                                    )

        return kernel

    # The grouped4 chain kernel with its key tiles software-pipelined: a CTA's next
    # tiles are in flight (cp.async, num_stages buffers) while the current one is
    # scored, instead of one tile load per round trip. Every score is the chain
    # kernel's GEMM (same shapes, threads and warp policy), ReLU, head sum and scale
    # on the same tile and rows, and sub-pages past the length re-read the tile's
    # last page (never stored), so the logits are bit-identical; only when the key
    # loads are issued moves.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_decode_grouped4_pipe_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        ctas_per_row: int = 1,
        min_scored_len: int = 0,
        num_stages: int = 2,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        GROUP = 64
        group_rows = 4
        score_heads = 4
        assert heads == 8
        assert GROUP % page_size == 0
        assert num_stages >= 1
        sub_pages = GROUP // page_size
        C = ctas_per_row
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(batch, group_rows), C, threads=threads) as (
                request,
                cta,
            ):
                row_base = request * group_rows
                q_shared = T.alloc_shared([group_rows, score_heads, head_dim], dtype)
                q_gemm = T.reshape(q_shared, (group_rows * score_heads, head_dim))
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, group_rows * score_heads], T.float32)
                scores_3d = T.reshape(scores, (GROUP, group_rows, score_heads))
                reduced = T.alloc_fragment([GROUP, group_rows], T.float32)
                context_len = T.alloc_var(T.int32)
                table_row = T.alloc_var(T.int32)
                # Graph verify rows share page prefixes and have increasing lengths.
                table_row = T.min(row_base + group_rows - 1, batch - 1)
                context_len = ContextLens[table_row]
                num_groups = T.ceildiv(context_len, GROUP)
                last_page = T.max(context_len - 1, 0) // page_size
                if context_len > min_scored_len and cta < num_groups:
                    T.copy(
                        Q[
                            row_base : row_base + group_rows,
                            0,
                            0:score_heads,
                            0:head_dim,
                        ],
                        q_shared,
                    )
                    for step in T.Pipelined(
                        T.ceildiv(num_groups - cta, C), num_stages=num_stages
                    ):
                        group = step * C + cta
                        # Unrolled in Python: the pipeliner takes no dynamic loop around
                        # shared-memory copies.
                        for sp in range(sub_pages):
                            T.copy(
                                KCache[
                                    PageTable[
                                        table_row,
                                        T.min(group * sub_pages + sp, last_page),
                                    ],
                                    :,
                                    0,
                                    :,
                                ],
                                k_shared[sp * page_size : (sp + 1) * page_size, :],
                            )
                        T.gemm(
                            k_shared,
                            q_gemm,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, qi, head in T.Parallel(
                            GROUP, group_rows, score_heads
                        ):
                            scores_3d[token, qi, head] = T.max(
                                scores_3d[token, qi, head], 0.0
                            )
                        T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                        for token, qi in T.Parallel(GROUP, group_rows):
                            position = group * GROUP + token
                            if row_base + qi < batch:
                                if (
                                    ContextLens[row_base + qi] > min_scored_len
                                    and position < ContextLens[row_base + qi]
                                ):
                                    Logits[row_base + qi, position] = (
                                        reduced[token, qi] / Scale
                                    )

        return kernel
