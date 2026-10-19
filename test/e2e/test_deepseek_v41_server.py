# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash with --enable-nvidia-megakernel on GB300, TP4, DSpark, across
the switch point of the post hops: up to 8 requests (48 rows) the routed-expert pair
and SGLang's fused posts, from 9 requests (54 rows) the two-shot hops; 12 and 16
requests step across it and back. Each server's startup lines show the covered
batches run the plugin; GSM8K runs across the switch point (gross failures); the
tokens of covered steps are scored again by the normal prefill path (KL
consistency) on short prompts and on prompts past the indexer's candidate span,
with a passkey to find there. Fail-stop: a stall or a crash on one rank, inside the
plugin's hop or anywhere else in the routed step, stops every rank and no token of
the failed step reaches the client; the tests print the time to stop.

SGLANG_NVIDIA_MEGAKERNEL_E2E_URL runs the serving tests against a server that is already up, so
the same checks also score the switch-off arm."""

import json
import os
import re
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import psutil
import requests
import sglang.test.kl_test_utils as kl_test_utils
from deepseek_v41_server import ARGS, ENV, MODEL, URL
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kl_test_utils import (
    compare_kl_divergence,
    get_input_ids,
)
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS_ENV
from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
# Few-shot GSM8K-200 (8 shots, 512 new tokens), for gross failures only; compare with
# the switch-off arm's own band for anything finer
GSM8K_BOUND = float(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BOUND", "0.85"))
# Decode or verify against prefill, mean KL over 8 LongBench-v2 prompts
KL_BOUND = 0.03
# Prompts past the candidate span (2048 blocks of 8 positions): every verify step's
# index sources run two-level
LONG_PROMPT_TOKENS = (18000, 21000, 24000, 27000)
LONG_KL_BOUND = 0.06
PASSKEY = "4861372950"
# The fault tests' watchdog; the plugin's waits end at half of it
WATCHDOG_S = 20
FAULT_PROMPT = "Write a long essay about lighthouses."
URL_ = URL or DEFAULT_URL_FOR_TEST
# A prepared LongBench-v2 prompt cache, for hosts without the dataset or with a read-only SGLang tree
if os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_LONGBENCH_CACHE"):
    kl_test_utils.CACHE_DIR = os.environ["SGLANG_NVIDIA_MEGAKERNEL_E2E_LONGBENCH_CACHE"]


def _logprobs(*, input_ids, max_new_tokens, logprob_start_len):
    """Greedy /generate of every prompt from an empty cache, with logprobs."""
    requests.post(
        URL_ + "/flush_cache", params={"timeout": 30}, timeout=40
    ).raise_for_status()
    r = requests.post(
        URL_ + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": logprob_start_len,
        },
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()


def _long_prompts():
    """Natural-text prompts past the candidate span: the KL test's LongBench-v2
    prompts joined end to end, in overlapping windows."""
    ids = get_input_ids(tokenizer_path=MODEL, num_samples=48, trust_remote_code=True)
    text = [t for p in ids for t in p]
    out, at = [], 0
    for n in LONG_PROMPT_TOKENS:
        out.append(text[at : at + n])
        at += n // 2
    assert all(len(p) == n for p, n in zip(out, LONG_PROMPT_TOKENS)), "corpus too short"
    return out


def _launch(args=ARGS, extra_env=None, extra_args=(), log=None, backend=True):
    # the product configuration: every slot the package ships, not a caller's override
    env = {
        **{k: v for k, v in os.environ.items() if k != SLOTS_ENV},
        **ENV,
        **(extra_env or {}),
    }
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=args + ([FLAG_CLI] if backend else []) + list(extra_args),
        env=env,
        return_stdout_stderr=log,
    )


def _gsm8k(parallel):
    host, port = re.match(r"https?://([^:/]+):(\d+)", URL_).groups()
    args = SimpleNamespace(
        num_shots=8,
        data_path=os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K"),
        num_questions=200,
        parallel=parallel,
        max_new_tokens=512,
        host=host,
        port=int(port),
    )
    metrics = run_eval_few_shot_gsm8k(args)
    print(f"gsm8k parallel={parallel} {metrics=}")
    return metrics


class _Stream(threading.Thread):
    """One streamed greedy request; records (arrival time, completion tokens, text)."""

    def __init__(self, url, prompt, max_new_tokens):
        super().__init__(daemon=True)
        self.url, self.prompt, self.max_new_tokens = url, prompt, max_new_tokens
        self.chunks, self.error = [], None

    def run(self):
        try:
            with requests.post(
                self.url + "/generate",
                json={
                    "text": self.prompt,
                    "stream": True,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": self.max_new_tokens,
                        "ignore_eos": True,
                    },
                },
                stream=True,
                timeout=(10, 120),
            ) as r:
                for line in r.iter_lines():
                    if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    self.chunks.append(
                        (
                            time.time(),
                            d.get("meta_info", {}).get("completion_tokens"),
                            d.get("text", ""),
                        )
                    )
        except Exception as e:  # the server dies under the request
            self.error = repr(e)


def _wait_exit(process, timeout_s):
    """Seconds until the server and every child exited, or None."""
    t0 = time.time()
    procs = [process] + [
        psutil.Process(c.pid)
        for c in psutil.Process(process.pid).children(recursive=True)
    ]
    while time.time() - t0 < timeout_s:
        if (
            all(
                not p.is_running() or p.status() == psutil.STATUS_ZOMBIE
                for p in procs[1:]
            )
            and process.poll() is not None
        ):
            return time.time() - t0
        time.sleep(0.2)
    return None


def _rank_process(process, rank):
    for c in psutil.Process(process.pid).children(recursive=True):
        try:
            if f"scheduler_TP{rank}" in c.name() or any(
                f"scheduler_TP{rank}" in a for a in c.cmdline()
            ):
                return c
        except psutil.Error:
            continue
    raise AssertionError(f"no scheduler_TP{rank} process")


class TestMegakernelVerify(CustomTestCase):
    """DSpark verify steps of 1-64 requests (6-384 rows) run the plugin."""

    # SGLang's startup line, once per TP rank: every shipped target slot built where it
    # was measured (hop A at batch 1-2, the fast top-k at every routed width, the Engram
    # prefetch at batch 1, the two-shot hops from batch 9), every width up to 384 rows
    ACTIVE = re.compile(
        r"NVIDIA megakernel: active \(DeepSeek-V4\.1: slots \["
        r"'attn', 'attn_indexer', 'attn_qkv_a', 'attn_qkv_a_bf16', 'aux_mean_stats', "
        r"'early_trigger', "
        r"'embed', 'engram', 'gate', 'gate_wide', 'hc_prenorm', 'hop_a', "
        r"'hop_a_batch2', 'hop_b', 'hop_b_start_trigger', 'index_q_mxfp8_in', "
        r"'kv_before_wait', 'moe', 'req_pos_once', "
        r"'sched_ahead', 'shared_ll', 'two_shot'\]; "
        r"hop A at \[6, 12\] rows, fast gate at \[6, 12, 18, 24, 30, 36, 42, 48\] rows, "
        r"Engram prefetch at \[6\] rows \(built, checked bit for bit\); boundary: "
        r"hop A at \[6, 12\] rows \(built\); hop B on; two-shot at 49-384 rows "
        r"\(built\); .*; widths "
        + re.escape(str(list(range(6, 385, 6))))
        + r"; tolerance numerics;"
    )
    # hop A at batch 2 and the fast top-k past 16 rows, per rank, from their dispatch
    SERVED = re.compile(
        r"NVIDIA megakernel: (hop A attention post serves 12|fast gate top-k serves 24) "
        r"rows"
    )
    # the captured decode graphs of 1 to 16 requests (this server's ceiling) run it
    CAPTURED = re.compile(
        r"NVIDIA megakernel: DeepSeek-V4\.1 runs the decode graphs of batch sizes "
        r"\[1, [^\]]*\b16\]"
    )
    TP = 4

    @classmethod
    def setUpClass(cls):
        cls.process, cls.logs = None, ()
        if URL:
            return
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"], log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def test_the_tested_batches_run_the_plugin(self):
        # routing is fixed at startup; without this the other tests could pass on the normal path alone
        log = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_LOG")
        text = (
            Path(log).read_text()
            if log
            else "".join(Path(f.name).read_text() for f in self.logs)
        )
        self.assertEqual(len(self.ACTIVE.findall(text)), self.TP, text[-4000:])
        self.assertEqual(len(self.CAPTURED.findall(text)), self.TP, text[-4000:])
        served = [m.group(1) for m in self.SERVED.finditer(text)]
        self.assertEqual(len(served), 2 * self.TP, text[-4000:])

    def test_gsm8k_across_the_two_shot_switch(self):
        # 12 and 16 requests: batches move between <= 8 (routed pair, SGLang's fused
        # posts) and >= 9 (two-shot hops)
        scores = {p: _gsm8k(p)["accuracy"] for p in (12, 16)}
        for parallel, accuracy in scores.items():
            self.assertGreater(accuracy, GSM8K_BOUND, f"parallel {parallel}")

    def test_covered_steps_match_prefill(self):
        # 8 requests in one batch: every verify step runs at a covered width, and the
        # prefill that scores the tokens again is the normal path
        assert_logprobs_match(
            URL_,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_samples=8,
            max_new_tokens=256,
            trust_remote_code=True,
        )

    def test_long_prompts_match_prefill(self):
        prompts = _long_prompts()
        decoded = _logprobs(input_ids=prompts, max_new_tokens=128, logprob_start_len=-1)
        outputs = [
            [x[0] for x in r["meta_info"]["output_token_logprobs"]] for r in decoded
        ]
        full = [p + r["output_ids"] for p, r in zip(prompts, decoded)]
        prefill = _logprobs(input_ids=full, max_new_tokens=0, logprob_start_len=0)
        inputs = [
            [x[0] for x in r["meta_info"]["input_token_logprobs"]][-len(o) :]
            for r, o in zip(prefill, outputs)
        ]
        compare_kl_divergence(
            input_logprobs=inputs,
            output_logprobs=outputs,
            ACC_THRESHOLDS={MODEL: {"kl_div": LONG_KL_BOUND}},
            model_name=MODEL,
            test_name="long prompts",
        )

    def test_passkey_past_the_candidate_span(self):
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tok = get_tokenizer(MODEL, trust_remote_code=True)
        prompt = _long_prompts()[1]
        cut = len(prompt) // 5
        needle = tok.encode(
            f" The passkey is {PASSKEY}. Remember it: {PASSKEY}. ",
            add_special_tokens=False,
        )
        question = tok.encode(
            " What is the passkey? The passkey is", add_special_tokens=False
        )
        r = requests.post(
            URL_ + "/generate",
            json={
                "input_ids": prompt[:cut] + needle + prompt[cut:] + question,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=600,
        )
        text = r.json()["text"]
        print(f"passkey answer: {text!r}")
        self.assertIn(PASSKEY, text)


class TestMegakernelFault(CustomTestCase):
    """One rank fails during routed verify steps (batch 1, 6 rows: every slot of the
    plugin runs); every rank must stop, and the client must get no token of the failed
    step: what it received is a prefix of a fault-free run of the same greedy request."""

    # "off" runs the stall and crash cases without the megakernel: the baseline for the
    # time to full exit when a rank stops outside the plugin's own waits
    SWITCH_ON = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_SWITCH", "on") != "off"

    def _serve(self, extra_env=None, extra_args=()):
        out, err = (
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        self.addCleanup(out.close)
        self.addCleanup(err.close)
        process = _launch(
            extra_env=extra_env,
            extra_args=["--watchdog-timeout", str(WATCHDOG_S), *extra_args],
            log=(out, err),
            backend=self.SWITCH_ON,
        )
        self.addCleanup(terminate_and_kill_process_tree, process)
        return process, (out, err)

    def _control(self, tokens):
        r = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": FAULT_PROMPT,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["text"]

    def _check(self, name, process, logs, stream, fault_t, control, expect_null_faults):
        t0 = time.time()
        stop_s = _wait_exit(process, 300)
        self.assertIsNotNone(stop_s, f"{name}: the server kept running")
        stream.join(30)
        last_t, last_n, text = stream.chunks[-1] if stream.chunks else (t0, 0, "")
        # an injected stall has no host-side time: measure from the last token the client got
        since = fault_t if fault_t is not None else last_t
        after = [
            c for c in stream.chunks if fault_t is not None and c[0] > fault_t + 0.5
        ]
        print(
            f"FAULT {name}: chunks={len(stream.chunks)} last_tokens={last_n} "
            f"time_to_stop_s={t0 + stop_s - since:.1f} (from {'the fault' if fault_t else 'the last token'}) "
            f"tokens_after_fault={len(after)} client_error={stream.error}"
        )
        self.assertTrue(
            stream.error or not stream.chunks or stream.chunks[-1][1] < 1024,
            f"{name}: the request completed",
        )
        self.assertTrue(
            control.startswith(text[: len(control)])
            or text.startswith(control[: len(text)]),
            f"{name}: a token the fault-free run does not have reached the client",
        )
        self.assertEqual(after, [], f"{name}: tokens arrived after the fault")
        for f in logs:
            f.seek(0)
        log = "".join(f.read() for f in logs)
        if expect_null_faults:
            # the plugin stops with a store to the null page: the only faults are at 0x0
            faults = re.findall(
                r"faulting addr: (0x[0-9a-fA-F]+)|illegal memory access", log
            )
            self.assertTrue(faults, log[-2000:])

    def test_the_plugins_hop_stalls_on_one_rank(self):
        # rank 1 skips its hop-A publish after 5000 hops (past startup, eager steps):
        # its peers' consumers reach the deadline, fault at 0x0, and every rank stops
        if not self.SWITCH_ON:
            self.skipTest("the plugin's own hop needs the plugin")
        process, logs = self._serve(
            {"SGLANG_NVIDIA_MEGAKERNEL_FAULT_INJECT": "1:hop_a:5000"},
            ["--disable-decode-cuda-graph"],
        )
        control = self._control(32)
        s = _Stream(DEFAULT_URL_FOR_TEST, FAULT_PROMPT, 1024)
        s.start()
        self._check("hop_a", process, logs, s, None, control, expect_null_faults=True)

    def _signal_rank(self, sig):
        process, logs = self._serve()
        control = self._control(256)
        s = _Stream(DEFAULT_URL_FOR_TEST, FAULT_PROMPT, 1024)
        s.start()
        while len(s.chunks) < 8 and s.is_alive():  # inside routed verify steps
            time.sleep(0.01)
        rank = _rank_process(process, 1)
        fault_t = time.time()
        os.kill(rank.pid, sig)
        self.addCleanup(lambda: rank.is_running() and os.kill(rank.pid, signal.SIGKILL))
        self._check(
            signal.Signals(sig).name,
            process,
            logs,
            s,
            fault_t,
            control,
            expect_null_faults=False,
        )

    def test_a_rank_stalls_anywhere_in_the_step(self):
        # SIGSTOP freezes rank 1 wherever it is: its peers wait in the plugin's hop or
        # in SGLang's own collectives of the routed step (the MoE post's push plane,
        # the attention post at 12-48 rows, the draft's exchanges)
        self._signal_rank(signal.SIGSTOP)

    def test_a_rank_dies_in_the_step(self):
        self._signal_rank(signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
