# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Kimi K3.1 split DSA selection on CPU: the merge storage comes from SGLang's
``DcpTopKMerge.create`` (which may decline) and the replicated split launch gets
that storage."""

import ast
import contextlib
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k31/dsa"
# split.py imports its kernel module as ..sm103.dsa.split_merge
KERNEL = "split_test.sm103.dsa.split_merge"


class SplitMergeCompatibility(unittest.TestCase):
    def load_split(self):
        node = next(
            n
            for n in ast.parse((ROOT / "split.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "SplitMerge"
        )
        scope = {"TOPK": 2048, "PAGE": 64, "__package__": "split_test.dsa"}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "split.py", "exec"), scope
        )
        return scope["SplitMerge"]

    def test_factory_storage_and_replicated_kernel(self):
        for cand_k in (256, 384):
            factory_calls, launch_calls = [], []
            storage = types.SimpleNamespace(
                buffer=torch.zeros(2 * 8 * (16 * cand_k + 2048), dtype=torch.int64),
                multicast_ptr=123,
                peer_ptrs=torch.arange(16),
                counters=torch.zeros(8),
                union_scratch=torch.empty(8, 16 * 2048),
                fallback=torch.zeros(8),
                dcp_rank=5,
                usable=lambda rows: rows <= 8,
            )

            def create(*args):
                factory_calls.append(args)
                return storage

            host = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_dcp_sharded")
            host.DcpTopKMerge = types.SimpleNamespace(create=create)
            kernel = types.ModuleType(KERNEL)
            kernel.dcp_topk_merge = lambda *a, **k: launch_calls.append((a, k))
            with patch.dict(
                sys.modules, {host.__name__: host, kernel.__name__: kernel}
            ):
                group = types.SimpleNamespace(rank_in_group=5, world_size=16)
                merge = self.load_split()(group, 8)
                self.assertEqual(factory_calls, [(group, 8, 2048, 5)])
                self.assertTrue(merge.usable(8))
                self.assertFalse(merge.usable(9))
                out = torch.empty(4, 2048, dtype=torch.int32)
                self.assertIs(merge.run("candidates", "lengths", "pages", out), out)
                args, kwargs = launch_calls[0]
                self.assertEqual(args, ("candidates", "lengths", "pages"))
                self.assertTrue(kwargs["split"])
                self.assertEqual(
                    (kwargs["rank"], kwargs["span"], kwargs["cand_k"]), (5, 64, cand_k)
                )
                self.assertIs(kwargs["buffer"], storage.buffer)
                self.assertEqual(kwargs["out_fallback"].shape, (4,))

    def test_declined_factory_remains_unusable(self):
        host = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_dcp_sharded")
        host.DcpTopKMerge = types.SimpleNamespace(create=lambda *a: None)
        kernel = types.ModuleType(KERNEL)
        kernel.dcp_topk_merge = lambda *a, **k: self.fail("declined merge launched")
        with patch.dict(sys.modules, {host.__name__: host, kernel.__name__: kernel}):
            merge = self.load_split()(types.SimpleNamespace(rank_in_group=0), 8)
            self.assertFalse(merge.usable(1))
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                merge.run(None, None, None, None)


class SplitArming(unittest.TestCase):
    """``BuiltSparseVerify._build_split``: the split arms from the measured crossover on; below it, under DCP
    and without a usable merge it declines (an empty split)."""

    def build_split(self, context_length, dcp=1, usable=True):
        node = next(
            n
            for n in ast.parse((ROOT / "verify.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "BuiltSparseVerify"
        )
        method = next(n for n in node.body if getattr(n, "name", "") == "_build_split")
        from sgl_nvidia_mega.kimi_k31.dsa import split as real

        built = []
        split = types.ModuleType("split_test.dsa.split")
        split.SPLIT_MIN = real.SPLIT_MIN
        split.LocalIndexer = lambda device, **kw: built.append(kw) or "local"
        merge = types.SimpleNamespace(usable=lambda rows: usable)
        split.SplitMerge = lambda group, rows: built.append(rows) or merge
        group = types.SimpleNamespace(world_size=16, rank_in_group=3)
        host = types.ModuleType("sglang.srt.distributed.parallel_state")
        host.get_tp_group = lambda: group
        scope = {
            "__package__": "split_test.dsa",
            "logger": types.SimpleNamespace(info=lambda *a: None),
            "dcp_layout": lambda: (dcp, 0),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "verify.py", "exec"),
            scope,
        )
        owner = types.SimpleNamespace(widths=(8, 16), query_width=8)
        modules = {
            "split_test": types.ModuleType("split_test"),
            "split_test.dsa": types.ModuleType("split_test.dsa"),
            split.__name__: split,
            host.__name__: host,
        }
        with patch.dict(sys.modules, modules):
            return scope["_build_split"](owner, "cuda:0", context_length), built

    def test_the_threshold_is_the_measured_crossover(self):
        from sgl_nvidia_mega.kimi_k31.dsa.split import SPLIT_MIN

        self.assertEqual(SPLIT_MIN, 320000)

    def test_it_arms_from_the_crossover_and_declines_where_it_cannot_run(self):
        armed, built = self.build_split(320000)
        self.assertEqual(sorted(armed), [8, 16])
        self.assertEqual(armed[8][0], "local")
        self.assertEqual(built[0], 16)  # the merge is sized for the widest step
        self.assertEqual(
            [(kw["requests"], kw["world"], kw["rank"]) for kw in built[1:]],
            [(1, 16, 3), (2, 16, 3)],
        )
        for name, args in {
            "below the crossover": (319999,),
            "DCP": (320000, 8),
            "no usable merge": (320000, 1, False),
        }.items():
            with self.subTest(name):
                self.assertEqual(self.build_split(*args)[0], {})


if __name__ == "__main__":
    unittest.main()


class K31DFlashRuntimeRound(unittest.TestCase):
    """The K3.1 DFlash runtime on CPU, over fakes of SGLang's worker bindings and the
    megakernel's target and draft chain: what it launches, in which order, and what it
    hands back to SGLang's stock round."""

    BLOCK = 8

    def setUp(self):
        try:
            from sglang.srt.kernel_plugins.nvidia import (  # noqa: F401
                kimi_k31_dflash_binding,
            )
        except ImportError:
            self.skipTest("this SGLang has no DFlash runtime binding")
        from sgl_nvidia_mega.kimi_k31.dflash import protocol
        from sgl_nvidia_mega.kimi_k31.dflash import runtime as rt

        self.rt, self.calls = rt, []
        # No MLP sync unless a test says so (SGLang reads it from the parallel config).
        mlp_sync = patch.object(rt, "require_mlp_sync", return_value=False)
        mlp_sync.start()
        self.addCleanup(mlp_sync.stop)
        log = self.calls.append
        self.chain = types.SimpleNamespace(
            widths=(self.BLOCK,),
            draft_aux_rows=(self.BLOCK,),
            draft_head_requests=(1,),
            draft_head_sampling=True,
            attach_draft_head=lambda **kw: (
                log("attach_head")
                or (torch.arange(7), torch.zeros(1, 7, 16), torch.zeros(1, 7, 16))
            ),
            stage_draft_head_sampling=lambda **kw: log("stage_head"),
        )
        protocol.DRAFT_CHAINS.append(self.chain)
        self.addCleanup(protocol.DRAFT_CHAINS.remove, self.chain)
        self.target = types.SimpleNamespace(
            prepare_sampled_accept=lambda **kw: log("prepare_sampled") or True,
            sampled_accept_fits=lambda bs, logits=None: True,
            sampled_accept=lambda **kw: (
                log("sampled_accept")
                or (torch.tensor([2], dtype=torch.int32), torch.tensor([9]), None)
            ),
        )
        lp = types.SimpleNamespace(
            vocab_size=32,
            final_logit_softcapping=None,
            defer_plain_gather=False,
            is_deferred_shard=lambda x: log("is_deferred_shard") or True,
            deferred_argmax=lambda x: log("deferred_argmax") or torch.zeros(1, 8),
        )
        runner = types.SimpleNamespace(
            model=types.SimpleNamespace(
                lm_head=types.SimpleNamespace(num_embeddings_per_partition=128),
                get_input_embeddings=lambda: (
                    lambda ids: log("embed") or torch.zeros(*ids.shape, 4)
                ),
            ),
            req_to_token_pool=types.SimpleNamespace(req_to_token=None),
            runner_ids=types.SimpleNamespace(plan=self._plan),
            attn_backend="target attention",
        )
        self.verify_out = types.SimpleNamespace(
            logits_output=types.SimpleNamespace(
                next_token_logits="shard", hidden_states=torch.zeros(8, 4)
            ),
            can_run_cuda_graph=True,
            routed_experts_output=None,
            indexer_topk_output=None,
            kv_weight_version_record=None,
        )
        self.bindings = {
            "block_size": self.BLOCK,
            "device": "cpu",  # the worker holds the server arg
            "mask_token_id": 3,
            "noise_embed_scale": 1.0,
            "batched_small_copy": False,
            "draft_model": types.SimpleNamespace(
                candidate_selector=types.SimpleNamespace(top_k=16),
                draft_config=types.SimpleNamespace(is_domino=False),
            ),
            "draft_model_runner": types.SimpleNamespace(
                runner_ids=types.SimpleNamespace(space=lambda kind: None),
                token_to_kv_pool="draft pool",
                forward=lambda fb: (
                    log("draft_forward") or types.SimpleNamespace(can_run_graph=True)
                ),
            ),
            "draft_parallel_layout": None,
            "draft_block_spec_info": None,
            "target_worker": types.SimpleNamespace(
                model_runner=runner,
                forward_batch_generation=lambda **kw: log("verify") or self.verify_out,
            ),
            "target_logits_processor": lp,
            "sim_acc_rng": None,
            "need_mamba_verify_commit": True,
            "compute_compact_draft_seq_lens": lambda lens: log("compact_lens") or lens,
            "fill_compact_seq_lens_cpu_bound": lambda **kw: log("compact_bound"),
            "rebuild_compact_draft_cache": lambda **kw: log("compact_rebuild"),
            "accept_block": lambda **kw: (
                log(("accept_block", kw["target_predict"]))
                or (
                    torch.tensor([1]),
                    torch.tensor([2]),
                    torch.tensor([7]),
                    torch.zeros(1, 8),
                    None,
                    None,
                )
            ),
        }
        self.graphs = graphs = types.SimpleNamespace(
            graph=types.SimpleNamespace(
                cuda_graph_config=types.SimpleNamespace(
                    decode=types.SimpleNamespace(bs=[1])
                )
            )
        )
        # The measured setup's server args; a test changes one before building.
        self.spec = types.SimpleNamespace(
            speculative_draft_window_size=1024,
            speculative_dflash_draft_kv_ring=False,
            speculative_use_block_verification=False,
        )
        self.kda_state = ("linear", "speculative state")
        kda_state = patch.object(rt.KdaCommit, "_kda_state", lambda kda: self.kda_state)
        kda_state.start()
        self.addCleanup(kda_state.stop)
        verify_input = types.SimpleNamespace(
            prepare_for_verify=lambda *a, **kw: log("prepare_verify") or ("fb", True)
        )
        for name, value in {
            "megakernel_agree": lambda value, what: log(("agree", what)),
            "built_megakernel": lambda model: self.chain,
            "get_parallel": lambda: types.SimpleNamespace(tp_rank=0),
            "get_exec": lambda: graphs,
            "get_spec": lambda: self.spec,
            "adjust_verify_logits": lambda *a, **kw: log("adjust"),
            "_prepare_dflash_draft_block_unchecked": lambda **kw: log("prepare_block"),
            "ForwardBatch": lambda **kw: log("forward_batch") or kw,
            "DFlashVerifyInput": lambda **kw: verify_input,
            "draft_tp_context": lambda layout: contextlib.nullcontext(),
            "apply_dflash_simulated_acceptance": lambda **kw: (
                log("simulate") or torch.tensor([11])
            ),
            "SIMULATE_ACC_LEN": 3.69,
            "enable_num_token_non_padded": lambda: False,
            "GenerationBatchResult": lambda **kw: kw,
        }.items():
            mocked = patch.object(rt, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def _plan(self, **kw):
        log = self.calls.append
        return types.SimpleNamespace(
            reads_from=lambda *a, **kw: types.SimpleNamespace(
                bind=lambda fb, translator: log("bind")
            ),
            bind=lambda fb, translator: log("bind (shared mapping)"),
            write_ids=lambda reader: torch.zeros(8, dtype=torch.int64),
        )

    def _binding(self, bindings=None):
        bindings = bindings or self.bindings
        return types.SimpleNamespace(pull=lambda name: bindings[name])

    def _runtime(self):
        runtime = self.rt.K31DFlashRuntime(self._binding(), self.target)
        log = self.calls.append
        runtime.draft_sampler(lm_head="lm_head")
        runtime._kda._megakernel_kda_commit = lambda **kw: log("kda_commit") or True
        runtime._aux._append_target_hidden_megakernel = lambda **kw: log("draft_aux")
        return runtime

    def _batch(self, **sampling):
        info = dict(
            is_all_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones(1, 1),
            top_ks=torch.tensor([20]),
            penalizer_orchestrator=None,
            has_custom_logit_processor=False,
            step_penalties=None,
            logit_bias=None,
            watermark=None,
            sampling_mask_batch_indices=None,
        )
        info.update(sampling)
        return types.SimpleNamespace(
            seq_lens=torch.tensor([5]),
            seq_lens_cpu=torch.tensor([5]),
            seq_lens_sum=5,
            req_pool_indices=torch.tensor([0]),
            spec_info=types.SimpleNamespace(
                bonus_tokens=torch.tensor([4]), nxt_kv_lens_cpu=None
            ),
            sampling_info=types.SimpleNamespace(**info),
            has_grammar=False,
            return_logprob=False,
            can_run_decode_cuda_graph=True,
        )

    def _round(self, runtime, batch):
        published = []

        def publish(new_seq_lens):
            self.calls.append("publish")
            published.append(new_seq_lens)

        with patch.object(torch.Tensor, "record_stream"):
            out = runtime.decode_round(batch, on_publish=publish, grammar_barrier=None)
        return out, published

    def test_init_builds_the_sampled_accept_and_arms_the_deferral_before_capture(self):
        runtime = self._runtime()
        self.assertEqual(
            self.calls,
            [("agree", "DFlash draft chain offers"), "prepare_sampled", "attach_head"],
        )
        self.assertTrue(self.bindings["target_logits_processor"].defer_plain_gather)
        self.assertEqual(runtime._round_bs, {1})
        self.assertEqual(runtime._aux._draft_kv_pool, "draft pool")
        self.assertEqual(runtime._kda._target_attn_backend, "target attention")

    def test_a_sampled_round_runs_the_measured_order(self):
        runtime = self._runtime()
        del self.calls[:]
        out, published = self._round(runtime, self._batch())
        self.assertEqual(
            self.calls,
            [
                "prepare_block",
                "embed",
                "compact_lens",
                "compact_bound",
                "compact_rebuild",
                "forward_batch",
                "bind",
                "stage_head",
                "draft_forward",
                "prepare_verify",
                "verify",
                "is_deferred_shard",
                "sampled_accept",
                "simulate",
                "kda_commit",
                "publish",
                "draft_aux",
            ],
        )
        # published once, between the KDA commit and the aux, with the simulated lengths
        self.assertEqual([t.tolist() for t in published], [[11]])
        self.assertEqual(out["new_seq_lens"].tolist(), [11])
        self.assertEqual(out["accept_lens"].tolist(), [3])
        self.assertIsNone(out["side_stream_done"])
        self.assertIsNone(self.verify_out.logits_output.next_token_logits)

    def test_a_greedy_round_takes_sglangs_accept_on_the_shard_argmax(self):
        runtime = self._runtime()
        del self.calls[:]
        self._round(runtime, self._batch(is_all_greedy=True))
        accepts = [c for c in self.calls if c[0] in ("accept_block", "sampled_accept")]
        self.assertEqual([c[0] for c in accepts], ["accept_block"])
        self.assertIn("deferred_argmax", self.calls)
        self.assertNotIn("simulate", self.calls)  # SGLang's accept simulates itself

    def test_rounds_it_does_not_serve_go_to_sglang_before_any_launch(self):
        runtime = self._runtime()
        cases = {
            "grammar": dict(has_grammar=True),
            "logprob": dict(return_logprob=True),
            "bs 2": dict(seq_lens=torch.tensor([5, 6])),
        }
        sampling = {
            "top_p": dict(need_top_p_sampling=True),
            "top_k": dict(need_top_k_sampling=True),
            "penalties": dict(step_penalties=object()),
            "logit bias": dict(logit_bias=torch.zeros(1)),
            "custom logit processor": dict(has_custom_logit_processor=True),
            "watermark": dict(watermark=object()),
            "sampling mask": dict(sampling_mask_batch_indices=[0]),
        }
        for name, fields in [*cases.items(), *sampling.items()]:
            with self.subTest(name):
                batch = self._batch(**(fields if name in sampling else {}))
                if name in cases:
                    batch.__dict__.update(fields)
                del self.calls[:]
                out, published = self._round(runtime, batch)
                self.assertIsNone(out)
                self.assertEqual((self.calls, published), ([], []))

    def test_the_decode_graph_flag_counts_only_under_mlp_sync(self):
        """SGLang sets can_run_decode_cuda_graph only under MLP sync; without it the
        flag stays False and the round still runs here."""
        for mlp_sync, served in ((False, True), (True, False)):
            with (
                self.subTest(mlp_sync=mlp_sync),
                patch.object(self.rt, "require_mlp_sync", return_value=mlp_sync),
            ):
                runtime = self._runtime()
                batch = self._batch()
                batch.can_run_decode_cuda_graph = False
                del self.calls[:]
                out, published = self._round(runtime, batch)
                self.assertEqual(out is not None, served)
                self.assertEqual(bool(self.calls), served)

    def test_a_kernel_limit_sends_sampled_rounds_to_sglang_and_keeps_greedy_ones(self):
        """Block verification, or no sampled-accept build: the megakernel sampled
        accept cannot run, the greedy accept still can."""
        for name, change in {
            "block verification": lambda: setattr(
                self.spec, "speculative_use_block_verification", True
            ),
            "no sampled accept": lambda: setattr(
                self.target, "prepare_sampled_accept", lambda **kw: False
            ),
        }.items():
            with (
                self.subTest(name),
                patch.object(self, "spec", copy.copy(self.spec)),
                patch.object(self, "target", copy.copy(self.target)),
            ):
                change()
                self.verify_out.logits_output.next_token_logits = "shard"
                self.verify_out.logits_output.hidden_states = torch.zeros(8, 4)
                runtime = self._runtime()
                del self.calls[:]
                out, _ = self._round(runtime, self._batch())
                self.assertEqual((out, self.calls), (None, []))
                out, _ = self._round(runtime, self._batch(is_all_greedy=True))
                self.assertIsNotNone(out)
                self.assertIn("kda_commit", self.calls)

    def test_a_layout_the_kernels_do_not_serve_sends_every_round_to_sglang(self):
        """No head for the captured graphs or the block, no aux, the draft KV ring, a
        sliding-window draft pool, a recurrent state the KDA commit is not laid out
        for: nothing fails at startup, and every round runs SGLang's, logged once."""
        bindings = self.bindings
        cases = {
            "decode graphs [1, 2, 4]": lambda: setattr(
                self.graphs.graph.cuda_graph_config.decode, "bs", [1, 2, 4]
            ),
            "block 5": lambda: bindings.__setitem__("block_size", 5),
            "no aux": lambda: setattr(self.chain, "draft_aux_rows", ()),
            "ring": lambda: setattr(
                self.spec, "speculative_dflash_draft_kv_ring", True
            ),
            "sliding window": lambda: setattr(
                bindings["draft_model_runner"].runner_ids, "space", lambda kind: "swa"
            ),
            "KDA state": lambda: setattr(self, "kda_state", None),
        }
        for name, change in cases.items():
            with self.subTest(name):
                saved = (
                    dict(self.chain.__dict__),
                    dict(self.spec.__dict__),
                    dict(bindings),
                    dict(bindings["draft_model_runner"].runner_ids.__dict__),
                    self.kda_state,
                )
                change()
                with self.assertLogs(self.rt.logger, "WARNING") as logs:
                    runtime = self._runtime()
                    del self.calls[:]
                    out, published = self._round(runtime, self._batch())
                    self._round(runtime, self._batch())
                self.assertEqual((out, self.calls, published), (None, [], []))
                (warning,) = logs.output  # once, at the draft graph's capture
                self.assertIn("every round runs SGLang's round", warning)
                self.chain.__dict__.update(saved[0])
                self.spec.__dict__.update(saved[1])
                bindings.update(saved[2])
                bindings["draft_model_runner"].runner_ids.__dict__.update(saved[3])
                self.kda_state = saved[4]
                self.graphs.graph.cuda_graph_config.decode.bs = [1]
        # The head stays off the draft graph only where it cannot serve it.
        self.graphs.graph.cuda_graph_config.decode.bs = [1, 2, 4]
        runtime = self.rt.K31DFlashRuntime(self._binding(), self.target)
        self.assertIsNone(runtime.draft_sampler(lm_head="lm_head"))
        self.graphs.graph.cuda_graph_config.decode.bs = [1]
        # Without a draft graph (cuda graphs off) the worker never asks for the head.
        runtime = self.rt.K31DFlashRuntime(self._binding(), self.target)
        out, _ = self._round(runtime, self._batch())
        self.assertIsNone(out)

    def test_the_shared_draft_mapping_without_a_draft_window(self):
        self.spec.speculative_draft_window_size = None
        runtime = self._runtime()
        del self.calls[:]
        out, _ = self._round(runtime, self._batch())
        self.assertIsNotNone(out)
        self.assertNotIn("compact_lens", self.calls)
        self.assertIn("bind (shared mapping)", self.calls)

    def test_gathered_logits_of_a_greedy_batch_take_the_adjusted_argmax(self):
        """SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=0: no sampled accept is built, sampled
        rounds run SGLang's, a greedy round accepts on the gathered logits."""
        lp = self.bindings["target_logits_processor"]
        lp.is_deferred_shard = lambda x: self.calls.append("is_deferred_shard") or False
        plain = types.SimpleNamespace(is_set=lambda: True, get=lambda: False)
        with patch.object(
            self.rt,
            "envs",
            types.SimpleNamespace(SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=plain),
        ):
            runtime = self._runtime()
        self.assertNotIn("prepare_sampled", self.calls)
        self.assertFalse(lp.defer_plain_gather)
        self.assertIsNone(self._round(runtime, self._batch())[0])
        logits = torch.zeros(8, 32)
        logits[:, 5] = 1.0
        self.verify_out.logits_output.next_token_logits = logits
        del self.calls[:]
        self._round(runtime, self._batch(is_all_greedy=True))
        ((_, target_predict),) = [c for c in self.calls if c[0] == "accept_block"]
        self.assertEqual(target_predict.tolist(), [5] * 8)
        self.assertLess(
            self.calls.index("adjust"),
            self.calls.index(("accept_block", target_predict)),
        )

    def test_no_recurrent_commit_and_a_domino_draft_follow_sglangs_round(self):
        self.bindings["need_mamba_verify_commit"] = False
        self.bindings["draft_model"].draft_config.is_domino = True
        runtime = self._runtime()
        del self.calls[:]
        out, _ = self._round(runtime, self._batch())
        self.assertIsNotNone(out)
        self.assertNotIn("kda_commit", self.calls)
        # SGLang's round accepts a Domino draft greedily, sampled batch or not.
        self.assertNotIn("sampled_accept", self.calls)
        self.assertIn("deferred_argmax", self.calls)

    def test_a_chain_without_its_aux_or_head_builds_no_sampled_accept(self):
        """The sampled accept (a collective build out of the KV budget) serves only
        rounds on the chain; a chain without the aux or the head serves none."""
        for name, offer in {
            "no aux": "draft_aux_rows",
            "no head": "draft_head_requests",
        }.items():
            with self.subTest(name):
                saved = dict(self.chain.__dict__)
                setattr(self.chain, offer, ())
                del self.calls[:]
                self._runtime()
                self.assertNotIn("prepare_sampled", self.calls)
                self.chain.__dict__.update(saved)
        del self.calls[:]
        self._runtime()
        self.assertIn("prepare_sampled", self.calls)


class K31DFlashRuntimeNaming(unittest.TestCase):
    def test_the_target_names_the_runtime_only_with_a_draft_on_the_chain(self):
        """Without a draft on the megakernel chain (another TP size, another draft) the
        runtime has nothing to run its rounds on: SGLang's DFlash round runs."""
        try:
            from sgl_nvidia_mega.kimi_k31.verify_spec import BuiltK31Verify
        except ImportError as e:  # the GPU-only kernel packages
            self.skipTest(f"verify_spec does not import here: {e}")
        from sgl_nvidia_mega.kimi_k31 import verify_spec
        from sgl_nvidia_mega.kimi_k31.dflash.protocol import DRAFT_CHAINS

        name = BuiltK31Verify.dflash_runtime.fget
        saved = list(DRAFT_CHAINS)
        self.addCleanup(lambda: DRAFT_CHAINS.__setitem__(slice(None), saved))
        with patch.object(verify_spec, "RUNTIME", True):
            DRAFT_CHAINS.clear()
            self.assertEqual(name(None), "")
            DRAFT_CHAINS.append(object())
            self.assertEqual(name(None), K31DFlashRuntimeGuard.PATH)
        with patch.object(verify_spec, "RUNTIME", False):  # K31MK_DFLASH_RUNTIME=0
            self.assertEqual(name(None), "")


class K31DFlashRuntimeGuard(unittest.TestCase):
    """SGLang's guard (sglang.srt.kernel_plugins.nvidia.kimi_k31_dflash_binding) against
    the runtime's declared bindings and guards, on the measured server arguments
    (G0/G2: TP16, page 64, block 8, FP8 draft KV, draft window 1024, decode graph bs
    [1]) and on setups it does not serve."""

    PATH = "sgl_nvidia_mega.kimi_k31.dflash.runtime:K31DFlashRuntime"

    def setUp(self):
        try:
            from sglang.srt.kernel_plugins.nvidia import kimi_k31_dflash_binding
        except ImportError:
            self.skipTest("this SGLang has no DFlash runtime binding")
        from sglang.srt.kernel_plugins.nvidia import binding
        from sglang.srt.runtime_context import get_parallel, reset_context

        from sgl_nvidia_mega.kimi_k31.dflash.runtime import K31DFlashRuntime

        self.sgl, self.runtime = kimi_k31_dflash_binding, K31DFlashRuntime
        self.addCleanup(reset_context)
        self._publish()
        for patched in (
            get_parallel().override(tp_rank=0),
            patch.object(self.sgl, "is_cuda", return_value=True),
            patch.object(self.sgl, "mambaish_config", return_value=object()),
            patch.object(binding, "_gather_tp", side_effect=lambda v, group=None: [v]),
            patch.object(self.sgl, "built_megakernel"),
        ):
            patched.__enter__()
            self.addCleanup(patched.__exit__, None, None, None)
        self.sgl.built_megakernel.return_value = types.SimpleNamespace(
            dflash_runtime=self.PATH
        )
        config = types.SimpleNamespace(dtype=torch.bfloat16)
        self.worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(model_config=config),
            draft_model_runner=types.SimpleNamespace(
                model_config=types.SimpleNamespace(
                    hf_config=types.SimpleNamespace(architectures=["DFlash2DraftModel"])
                )
            ),
            block_size=8,
            use_compact_draft_cache=True,
            draft_kv_ring=None,
            selector=object(),
            _selector_sampling_enabled=True,
            lilicorr=None,
            _is_domino=False,
            _block_verification=False,
            _full_embed_gpu=None,
            _use_aux_graph=True,
        )

    def _publish(self, tp_size=16, graph_bs=(1,)):
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
            with_phase,
        )
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        publish(
            ServerArgs(
                model_path="dummy",
                tp_size=tp_size,
                page_size=64,
                kv_cache_dtype="fp8_e4m3",
                speculative_algorithm="DFLASH",
                speculative_num_draft_tokens=8,
                speculative_draft_kv_cache_dtype="fp8_e4m3",
                speculative_draft_window_size=1024,
                enable_nvidia_megakernel=True,
                cuda_graph_config=with_phase(
                    default_cuda_graph_config(), "decode", bs=list(graph_bs)
                ),
            ),
            role="test",
        )

    def test_no_setup_declines_the_runtime(self):
        """The runtime declares no setup guards: SGLang binds it on the measured setup,
        at TP8 and with decode graphs [1, 2, 4] alike; the target build and the draft
        chain decide what runs (K31DFlashRuntimeRound)."""
        self.assertEqual(self.runtime.guards, {})
        for tp_size, graph_bs in ((16, (1,)), (8, (1,)), (16, (1, 2, 4))):
            with self.subTest(tp_size=tp_size, graph_bs=graph_bs):
                self._publish(tp_size=tp_size, graph_bs=graph_bs)
                guard = self.sgl.DFlashGuard(self.runtime.guards)
                self.assertEqual(guard.unmet(self.worker), [])
                self.sgl.DFlashBinding(self.worker, frozenset(self.runtime.bindings))


def _load(path, names, **scope):
    """The named top-level definitions of ``path`` (under kimi_k31), run without its GPU imports."""
    body = ast.parse((ROOT.parent / path).read_text()).body
    keep = [
        n
        for n in body
        if getattr(n, "name", None) in names
        or any(
            getattr(t, "id", None) in names
            for t in getattr(n, "targets", [getattr(n, "target", None)])
        )
    ]
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), scope)
    return scope


class K31VerifyQ5(unittest.TestCase):
    """Q5 is served by the draft token count, with no env: the R5 setup (GB300, TP8/DCP8, Q5, PD decode) is admitted
    and gets the R5 kernel choices; Q5 is not tied to TP8/DCP8."""

    def test_the_draft_token_count_admits_q5(self):
        from sgl_nvidia_mega.kimi_k31 import verify_contract

        reasons = _load(
            "verify_spec.py",
            {"config_disable_reasons", "_disaggregation_reasons", "DCP_SIZES"},
            supported_query_width=verify_contract.supported_query_width,
        )["config_disable_reasons"]
        r5 = types.SimpleNamespace(
            speculative_algorithm="DFLASH",
            speculative_num_draft_tokens=5,
            enable_linear_replayssm_spec=True,
            enable_linear_replayssm=False,
            context_length=1048576,
            cuda_graph_config=types.SimpleNamespace(
                decode=types.SimpleNamespace(backend="full")
            ),
            attention_backend=None,
            prefill_attention_backend="dsa",
            decode_attention_backend="dsa",
            tp_size=8,
            dcp_size=8,
            attn_cp_size=1,
            enable_prefill_cp=False,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            disaggregation_decode_host_receive_threshold=1,
        )
        with patch.dict("os.environ", clear=True):
            self.assertEqual(reasons(r5, None), [])
            self.assertEqual(
                verify_contract.verify_keys(
                    5, verify_contract.served_request_batches(5)
                ),
                tuple((b, 5, 5 * b) for b in range(1, 9)),
            )
            for name, change, refused in (
                ("TP16/DCP1", {"tp_size": 16, "dcp_size": 1}, False),
                ("TP16/DCP16", {"tp_size": 16, "dcp_size": 16}, False),
                ("DCP below TP", {"dcp_size": 4}, True),
                ("prefill CP", {"enable_prefill_cp": True}, True),
                ("Q4", {"speculative_num_draft_tokens": 4}, True),
            ):
                with self.subTest(name):
                    cfg = copy.copy(r5)
                    vars(cfg).update(change)
                    self.assertEqual(bool(reasons(cfg, None)), refused)

    def test_q5_rows_run_the_next_wider_projection_tile(self):
        """A width with no tactic of its own runs the next wider entry's single N tile, which reproduces R5's Q5
        tiles (rows 10/15/20/25/30/35 on 16/16/24/32/32/40); up to 8 rows, and past the widest entry, the 8-row
        fallback stays."""
        scope = _load(
            "sm103/k3/gemm/projection.py",
            {
                "GB300_TACTICS",
                "GB300_FALLBACK",
                "FALLBACK_MAX_ROWS",
                "_FALLBACK_LOGGED",
                "_gb300_tactic",
                "SM103_TP16_TACTICS",
                "SM103_TP8_TACTICS",
                "_SM103_TP16_LOGGED",
                "projection_tactic",
            },
            torch=torch,
            logger=types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        )
        tactic, tp8, gb300 = (
            scope["projection_tactic"],
            scope["SM103_TP8_TACTICS"],
            scope["GB300_TACTICS"],
        )
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            for kind, rows in (("mla", 3648), ("kda", 6288)):
                for batch, tile in {
                    10: 16,
                    15: 16,
                    20: 24,
                    25: 32,
                    30: 32,
                    35: 40,
                    40: 40,
                }.items():
                    with self.subTest(kind=kind, batch=batch):
                        want = gb300[kind].get(tile) or tp8[(kind, rows, tile)]
                        self.assertEqual(tactic(kind, batch, 0, rows=rows), want)
                self.assertEqual(
                    tactic(kind, 5, 0, rows=rows), scope["GB300_FALLBACK"][kind]
                )
                self.assertEqual(
                    tactic(kind, 72, 0, rows=rows), scope["GB300_FALLBACK"][kind]
                )
            # the measured TP16 Q8 B1 entry and the TP16 pitch's other widths
            self.assertEqual(
                tactic("mla", 8, 0, rows=2880),
                scope["SM103_TP16_TACTICS"][("mla", 2880, 8)],
            )
            self.assertEqual(tactic("mla", 24, 0, rows=2880), gb300["mla"][32])

    def test_q5_breq3_runs_four_ctas_per_kda_head(self):
        scope = _load(
            "sm103/k3/kda/verify.py",
            {"SM103_TP8_HEADS", "SM103_TP8_VERIFY", "verify_schedule"},
            torch=torch,
        )
        schedule = scope["verify_schedule"]
        with (
            patch.dict("os.environ", clear=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            self.assertEqual(schedule(3, 5, 0, heads=12), (0, 8, 2, 4))
            self.assertEqual(schedule(2, 5, 0, heads=12), (0, 8, 2, 8))


class K31DcpQueryGather(unittest.TestCase):
    """Under DCP the sparse verify gathers the query early wherever the one-shot merge serves the rows, and
    multicasts it directly wherever the group has multicast; elsewhere it keeps the group's all-gather."""

    def test_the_early_gather_follows_the_merge(self):
        method = next(
            n
            for c in ast.parse((ROOT / "verify.py").read_text()).body
            if isinstance(c, ast.ClassDef) and c.name == "PreparedSparseVerifyMla"
            for n in c.body
            if getattr(n, "name", "") == "early_gather"
        )
        host = types.SimpleNamespace(_TOPK_MERGE=None)
        scope = {
            "DCP_SHARDED": host,
            "_EARLY_QAG_LOGGED": [],
            "logger": types.SimpleNamespace(info=lambda *a: None),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "verify.py", "exec"),
            scope,
        )
        early = scope["early_gather"]
        with patch.dict("os.environ", clear=True):
            self.assertFalse(early(None, 40))  # no one-shot merge: NCCL candidates
            host._TOPK_MERGE = types.SimpleNamespace(usable=lambda rows: rows <= 32)
            self.assertTrue(early(None, 32))
            self.assertFalse(early(None, 40))

    def test_the_direct_gather_needs_one_clique_with_multicast_on_every_rank(self):
        import functools

        from sgl_nvidia_mega.kimi_k31.dsa.mla import _CP_MAX_ROWS

        made = []

        def create_state(group, rank, rows, words, device=None):
            made.append((rows, words))
            return types.SimpleNamespace(
                symm_mem_hdl=types.SimpleNamespace(multicast_ptr=7),
                comm_buff=torch.ones(1),
            )

        def scope(clique, peer_multicast):
            def all_reduce(flag, op, group):
                flag.fill_(min(int(flag.item()), peer_multicast))

            return _load(
                "dsa/verify.py",
                {"_direct_query_state"},
                cache=functools.cache,
                torch=types.SimpleNamespace(
                    tensor=torch.tensor,
                    cuda=types.SimpleNamespace(
                        is_current_stream_capturing=lambda: False,
                        synchronize=lambda device: None,
                    ),
                    distributed=types.SimpleNamespace(
                        all_reduce=all_reduce, ReduceOp=types.SimpleNamespace(MIN="min")
                    ),
                ),
                in_the_same_node_as=lambda group: [True, False],
                is_one_nvlink_clique=lambda group, device: clique,
                create_state=create_state,
                _CP_MAX_ROWS=_CP_MAX_ROWS,
                logger=types.SimpleNamespace(info=lambda *a: None),
            )["_direct_query_state"]

        group = type(
            "Group",
            (),
            {
                "cpu_group": "cpu",
                "device_group": "dev",
                "rank_in_group": 0,
                "device": 0,
            },
        )()
        self.assertIsNone(scope(False, 1)(group, "cuda:0"))
        self.assertEqual(made, [])
        self.assertIsNone(scope(True, 0)(group, "cuda:0"))
        state = scope(True, 1)(group, "cuda:0")
        self.assertIsNotNone(state)
        self.assertEqual(state.comm_buff.item(), 0)  # the padding heads start zero
        self.assertEqual(made[-1], (_CP_MAX_ROWS, 128 * 576 // 2))


def _gpu_scorer_ready():
    """An SM100-family GPU with DeepGEMM and the CUDA driver bindings: the one-block scorer runs."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cuda.bindings.driver  # noqa: F401
        import deep_gemm  # noqa: F401
    except ImportError:
        return False
    return True


class K31VerifyScoreOneBlock(unittest.TestCase):
    """The DSA scorers give a request's draft rows one Q block where DeepGEMM would split them and the block's
    three TMEM stages fit, on SM100-family GPUs only; elsewhere DeepGEMM's own call runs. The FP32 logits equal
    DeepGEMM's bit for bit (GPU)."""

    def test_one_block_where_deepgemm_splits_and_tmem_fits(self):
        from sgl_nvidia_mega.kimi_k31.sm103.dsa import score

        for capability, served in (
            ((10, 3), {5}),
            ((10, 0), {5}),
            ((9, 0), set()),
            ((12, 0), set()),
        ):
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                for next_n in range(1, 17):
                    with self.subTest(capability=capability, next_n=next_n):
                        self.assertEqual(
                            score.serves("cuda:0", next_n), next_n in served
                        )

    def test_unserved_widths_call_deepgemm_and_served_ones_launch_one_block(self):
        from sgl_nvidia_mega.kimi_k31.sm103.dsa import score

        def inputs(batch, next_n, pages=9, table_pages=5):
            return (
                torch.zeros(batch, next_n, 32, 128, dtype=torch.float8_e4m3fn),
                torch.zeros(pages, 64, 1, 132, dtype=torch.uint8),
                torch.zeros(batch * next_n, 32, dtype=torch.float32),
                torch.zeros(batch, next_n, dtype=torch.int32),
                torch.zeros(batch, table_pages, dtype=torch.int32),
                torch.zeros(149, 2, dtype=torch.int32),
                table_pages * 64 - 3,
            )

        stock = []
        deep_gemm = types.ModuleType("deep_gemm")
        deep_gemm.fp8_paged_mqa_logits = lambda *a, **k: stock.append((a, k)) or "stock"
        launches, maps = [], []
        driver = types.SimpleNamespace(
            CUtensorMapDataType=types.SimpleNamespace(
                CU_TENSOR_MAP_DATA_TYPE_UINT8="u8",
                CU_TENSOR_MAP_DATA_TYPE_FLOAT32="f32",
            ),
            CUtensorMapSwizzle=types.SimpleNamespace(
                CU_TENSOR_MAP_SWIZZLE_128B="128B", CU_TENSOR_MAP_SWIZZLE_NONE="none"
            ),
        )

        def record_map(dtype, base, dims, strides, box, swizzle):
            maps.append((dtype, dims, strides, box, swizzle))
            return len(maps) - 1

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
            patch.object(
                score, "load_kernel", lambda index, next_n: ("kernel", next_n)
            ),
            patch.object(score, "driver", lambda: driver),
            patch.object(score, "_map", record_map),
            patch.object(score, "launch", lambda *a: launches.append(a)),
            patch.object(
                torch.cuda,
                "current_stream",
                lambda device: types.SimpleNamespace(cuda_stream=7),
            ),
        ):
            for next_n in (1, 4, 8):
                args = inputs(2, next_n)
                self.assertEqual(
                    score.fp8_paged_mqa_logits(*args, clean_logits=False), "stock"
                )
                self.assertIs(stock[-1][0][0], args[0])
                self.assertEqual(stock[-1][1], {"clean_logits": False})
            self.assertEqual(launches, [])

            args = inputs(3, 5)
            logits = score.fp8_paged_mqa_logits(*args, clean_logits=False)
            self.assertEqual(len(stock), 3)
            ((kernel, grid, params, stream),) = launches
            self.assertEqual((kernel, grid, stream), (("kernel", 5), (148, 1, 1), 7))
            # rows, the row stride (317 keys up to the 256-key split), the table stride
            self.assertEqual([p.value for p in params[:3]], [15, 512, 5])
            self.assertEqual(logits.shape, (15, 317))
            self.assertEqual(logits.stride(), (512, 1))
            q, sf_kv, kv, sf_kv_again, w = params[7:]
            # FP8: the KV-scale map fills the unused sf_q slot, as DeepGEMM's
            self.assertEqual(sf_kv, sf_kv_again)
            self.assertEqual(
                maps[q], ("u8", (128, 15 * 32), (128,), (128, 5 * 32), "128B")
            )
            self.assertEqual(
                maps[kv], ("u8", (128, 64, 9), (128, 8448), (128, 64, 1), "128B")
            )
            self.assertEqual(maps[sf_kv], ("f32", (64, 9), (8448,), (64, 1), "none"))
            self.assertEqual(maps[w], ("f32", (32, 15), (128,), (32, 5), "none"))
            page32 = list(inputs(3, 5))
            page32[1] = torch.zeros(9, 32, 1, 132, dtype=torch.uint8)
            with self.assertRaises(ValueError):  # not the page-64 FP8 index-K
                score.fp8_paged_mqa_logits(*page32)

    def test_both_dsa_scorers_score_through_the_dispatcher(self):
        """The replicated verify selection and the context-split local indexer pass DeepGEMM's arguments to it."""
        from sgl_nvidia_mega.kimi_k31.sm103.dsa import score

        calls = []

        def scorer(*args, **kwargs):
            calls.append((args, kwargs))
            return torch.zeros(args[0].shape[0] * args[0].shape[1], 300)

        step = types.SimpleNamespace(
            split=0,
            dcp_size=2,
            batch_requests=2,
            query_width=5,
            local_contexts="local",
            score_contexts="stock",
            score_schedule="schedule",
        )
        query, gates = torch.zeros(10, 32, 128), torch.zeros(10, 32, 1)
        cache, table = torch.zeros(7 * 64, 132), torch.zeros(2, 6, dtype=torch.int32)
        selection = _load(
            "dsa/verify.py",
            {"VerifySelection"},
            torch=torch,
            score=types.SimpleNamespace(fp8_paged_mqa_logits=scorer),
        )["VerifySelection"]
        owner = object.__new__(selection)
        owner.split = None
        owner.select_dcp = lambda scores, t, s: ("dcp", scores.shape)
        self.assertEqual(
            selection.run(owner, query, gates, cache, table, step), ("dcp", (10, 300))
        )
        args, kwargs = calls[-1]
        self.assertEqual(
            (args[0].shape, args[1].shape, args[2].shape),
            ((2, 5, 32, 128), (7, 64, 1, 132), (10, 32)),
        )
        self.assertEqual(args[3:], ("local", table, "schedule", 6 * 64))
        self.assertEqual(kwargs, {"clean_logits": False})

        from sgl_nvidia_mega.kimi_k31.dsa.split import LocalIndexer

        topk = types.ModuleType("sglang.kernels.ops.attention.dsv4.topk")
        topk.topk_transform_paged_candidates_v2 = lambda *a, **k: None
        indexer = object.__new__(LocalIndexer)
        vars(indexer).update(
            prepared_for=step,  # prepared this forward
            local_pages=4,
            local_contexts="contexts",
            local_table="table",
            schedule="split schedule",
            local_lengths=None,
            candidates="candidates",
            world=16,
            rank=3,
            overflow_rows=None,
        )
        with (
            patch.dict(sys.modules, {topk.__name__: topk}),
            patch.object(score, "fp8_paged_mqa_logits", scorer),
        ):
            self.assertEqual(
                indexer.run(query, gates, cache, table, step), "candidates"
            )
        args, kwargs = calls[-1]
        self.assertEqual(args[3:], ("contexts", "table", "split schedule", 4 * 64))
        self.assertEqual(kwargs, {"clean_logits": False})

    @unittest.skipUnless(
        _gpu_scorer_ready(), "needs an SM100-family GPU, DeepGEMM and cuda-python"
    )
    def test_logits_equal_deepgemm_bit_for_bit(self):
        """Breq 1..8 x contexts 1K..400K, lengths ending mid-page and mid-split, eager; every logit a request's
        last row reaches (the kernel computes each request through its last row's context, as DeepGEMM)."""
        for batch in range(1, 9):
            for keys in (1024, 4096 + 17, 32768 + 100, 131072 + 33, 400000):
                with self.subTest(batch=batch, keys=keys):
                    self._assert_equal(
                        *self._scores(batch, keys, seed=batch * 7 + keys % 97)
                    )

    @unittest.skipUnless(
        _gpu_scorer_ready(), "needs an SM100-family GPU, DeepGEMM and cuda-python"
    )
    def test_a_captured_graph_replays_deepgemm_bits(self):
        from sgl_nvidia_mega.kimi_k31.sm103.dsa import score

        args = self._case(4, 131072 + 33, seed=3)
        # built before capture, as the sparse verify build does
        score.load_kernel(torch.cuda.current_device(), 5)
        graph, out = torch.cuda.CUDAGraph(), {}
        with torch.cuda.graph(graph):
            out["logits"] = score.fp8_paged_mqa_logits(*args, clean_logits=False)
        args[0].copy_(torch.randn(args[0].shape, device="cuda").to(torch.float8_e4m3fn))
        graph.replay()
        torch.cuda.synchronize()
        import deep_gemm

        self._assert_equal(
            deep_gemm.fp8_paged_mqa_logits(*args, clean_logits=False),
            out["logits"],
            args[3],
        )

    def _case(self, batch, keys, seed, next_n=5):
        import deep_gemm

        g = torch.Generator(device="cuda").manual_seed(seed)
        lens = torch.tensor(
            [
                [keys - next_n + 1 + j + 101 * b for j in range(next_n)]
                for b in range(batch)
            ],
            dtype=torch.int32,
            device="cuda",
        )
        pages = -(-int(lens.max()) // 64)
        pool = batch * pages + 8
        codes = torch.randn(pool, 64 * 128, device="cuda", generator=g).to(
            torch.float8_e4m3fn
        )
        scales = torch.rand(pool, 64, device="cuda", generator=g) * 0.1 + 0.01
        cache = torch.cat(
            (codes.view(torch.uint8), scales.view(torch.uint8)), dim=1
        ).view(pool, 64, 1, 132)
        table = (
            torch.randperm(pool, device="cuda", generator=g)[: batch * pages]
            .view(batch, pages)
            .int()
        )
        q = torch.randn(batch, next_n, 32, 128, device="cuda", generator=g).to(
            torch.float8_e4m3fn
        )
        weights = torch.randn(batch * next_n, 32, device="cuda", generator=g)
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            lens, 64, deep_gemm.get_num_sms()
        )
        return q, cache, weights, lens, table, schedule, pages * 64

    def _scores(self, batch, keys, seed):
        import deep_gemm

        from sgl_nvidia_mega.kimi_k31.sm103.dsa import score

        args = self._case(batch, keys, seed)
        ref = deep_gemm.fp8_paged_mqa_logits(*args, clean_logits=False)
        return ref, score.fp8_paged_mqa_logits(*args, clean_logits=False), args[3]

    def _assert_equal(self, ref, out, lens):
        self.assertEqual(ref.shape, out.shape)
        self.assertEqual(ref.stride(), out.stride())
        reach = (
            lens[:, -1:].expand_as(lens).reshape(-1, 1)
        )  # each request's last row's context
        valid = torch.arange(ref.shape[1], device=ref.device)[None, :] < reach
        self.assertTrue(
            torch.equal(ref.view(torch.int32)[valid], out.view(torch.int32)[valid])
        )
