# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Matched B300 TP8/q6 CI: native stock draft, TRTLLM MLA, DCP off.

Two entry points share one sequential stock/plugin pair. GSM8K-200 permits at
most a 0.005 absolute score drop (one answer), fixed before the candidate runs.
The separate serving logprob check retains the 0.5 bound for a shared generated
prefix. Performance reports three warmed repeats at C1/C4 with natural
acceptance; it is end-to-end serving, not an equal-work kernel benchmark.
This is bounded 32K-input coverage, not new 262K or ordinary-q1 qualification.
"""

import importlib.metadata
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

MODEL = "moonshotai/Kimi-K3"
DRAFT = "RadixArk/Kimi-K3-DSpark"
URL = "http://127.0.0.1:30000"
PROMPT = "The first ten positive integers are 1, 2, 3,"
SLOTS = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
COMMON_ARGS = [
    "--trust-remote-code",
    "--language-only",
    "--tp-size",
    "8",
    "--dcp-size",
    "1",
    "--dtype",
    "bfloat16",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "64",
    "--attention-backend",
    "trtllm_mla",
    # Admission checks the phase fields directly, not their inherited backend.
    "--prefill-attention-backend",
    "trtllm_mla",
    "--decode-attention-backend",
    "trtllm_mla",
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--mem-fraction-static",
    "0.85",
    "--context-length",
    "40960",
    "--max-running-requests",
    "4",
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--speculative-draft-model-path",
    DRAFT,
    "--random-seed",
    "42",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 12}',
    "--decode-log-interval",
    "4",
    "--watchdog-timeout",
    "600",
]


def _check_admission(args, output):
    """Resolve the real plugin launch configuration before either weight load."""
    with (output / "admission.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from sglang.srt.plugins import load_plugins; "
                "from sglang.srt.server_args import prepare_server_args; "
                "load_plugins(); prepare_server_args(sys.argv[1:]).resolve_once(); "
                "print('KIMI_B300_ADMISSION=PASS', flush=True)",
                "--model-path",
                MODEL,
                *args,
                "--enable-nvidia-megakernel",
            ],
            env={**os.environ, SLOTS: "sm148"},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=True,
        )


def _request(endpoint, payload=None):
    response = (
        requests.get(URL + endpoint, timeout=30)
        if payload is None
        else requests.post(URL + endpoint, json=payload, timeout=900)
    )
    response.raise_for_status()
    return response.json()


def _generate(prompts, *, tokens=128):
    result = _request(
        "/generate",
        {
            "input_ids": prompts,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        },
    )
    assert len(result) == len(prompts)
    for row in result:
        meta = row["meta_info"]
        assert meta["completion_tokens"] == tokens, row
        assert meta["finish_reason"]["type"] == "length", row
        assert row["text"].strip(), row
        assert meta["spec_verify_ct"] > 0, "DSpark did not verify"
        assert meta["spec_num_proposed_drafts"] == 5 * meta["spec_verify_ct"]
        assert meta["output_token_logprobs"]
        assert all(math.isfinite(entry[0]) for entry in meta["output_token_logprobs"])
    return result


def _measure(prompts):
    start = time.perf_counter()
    result = _generate(prompts)
    elapsed = time.perf_counter() - start
    output = sum(row["meta_info"]["completion_tokens"] for row in result)
    verifications = sum(row["meta_info"]["spec_verify_ct"] for row in result)
    return {
        "seconds": elapsed,
        "output_tokens": output,
        "tokens_per_second": output / elapsed,
        "accept_length_including_bonus": output / verifications,
        "accepted_drafts": sum(
            row["meta_info"]["spec_num_correct_drafts"] for row in result
        ),
        "verify_steps": verifications,
    }


def _route_check(log, enabled):
    active = "NVIDIA megakernel: active (Kimi K3 packed-MXFP4:"
    assert (active in log) == enabled, "wrong stock/plugin execution path"
    if enabled:
        assert (
            "Kimi K3 packed-MXFP4 runs the decode graphs of batch sizes [1, 2]" in log
        )
        # No other model (including the draft) may activate a plugin.
        assert all(
            "Kimi K3 packed-MXFP4:" in name
            for name in re.findall(r"NVIDIA megakernel: active \(([^\n]+)", log)
        )
    observed = {
        (int(bs), graph)
        for bs, graph in re.findall(
            r"Decode batch[^\n]*?#running-req: (\d+),[^\n]*?cuda graph: (True|False)",
            log,
        )
    }
    assert {(1, "True"), (2, "True"), (3, "False"), (4, "False")} <= observed, observed


@pytest.fixture(scope="module")
def matched(tmp_path_factory):
    # Native dependencies are deliberately not stubbed or silently skipped.
    import torch
    from huggingface_hub import hf_hub_download
    from sglang.test.sgl_eval_utils import run_sgl_eval
    from sglang.test.test_utils import (
        _wait_for_gpu_idle_in_ci,
        popen_launch_server,
        terminate_and_kill_process_tree,
    )
    from transformers import AutoTokenizer

    assert torch.cuda.device_count() == 8, "requires eight B300 GPUs on one host"
    hardware = []
    for index in range(8):
        prop = torch.cuda.get_device_properties(index)
        assert "B300" in prop.name and "GB300" not in prop.name, prop.name
        assert (prop.major, prop.minor, prop.multi_processor_count) == (10, 3, 148)
        hardware.append({"name": prop.name, "sms": prop.multi_processor_count})
    output = (
        Path(
            os.environ.get("MEGA_E2E_OUTPUT_DIR")
            or tmp_path_factory.mktemp("megakernel-e2e")
        )
        / "kimi_k3"
    )
    output.mkdir(parents=True, exist_ok=True)
    revisions = {
        # Keep the snapshot path: resolve() would follow config.json into blobs/.
        model: Path(hf_hub_download(model, "config.json")).parent.name
        for model in (MODEL, DRAFT)
    }
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in revisions.values())
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=revisions[MODEL], trust_remote_code=True
    )
    prompt = tokenizer.encode(PROMPT, add_special_tokens=False)
    filler = tokenizer.encode(
        "A plain record contains no secret code.\n", add_special_tokens=False
    )
    needle = tokenizer.encode("\nThe secret code is 71432.\n", add_special_tokens=False)
    question = tokenizer.encode(
        "\nWhat is the secret code? Answer with just the digits.\n",
        add_special_tokens=False,
    )
    long_prompt = (filler * (32768 // len(filler) + 1))[:16000] + needle
    long_prompt += (filler * (32768 // len(filler) + 1))[
        : 32768 - len(long_prompt) - len(question)
    ] + question
    # Reject inherited experimental knobs rather than silently testing a different recipe.
    unexpected = [
        key
        for key in os.environ
        if key.startswith(
            (
                "SGLANG_NVIDIA_MEGAKERNEL_",
                "SGLANG_SIMULATE_",
                "SGLANG_DSPARK_",
                "SGLANG_KIMI_",
            )
        )
    ]
    assert not unexpected, f"unset experimental overrides before running: {unexpected}"
    args = COMMON_ARGS + [
        "--revision",
        revisions[MODEL],
        "--speculative-draft-model-revision",
        revisions[DRAFT],
    ]
    import sglang

    sources = {}
    for name, path in (
        ("framework", Path(sglang.__file__).resolve().parents[2]),
        ("plugin", Path(__file__).resolve().parents[2]),
    ):
        sources[name] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    (output / "identity.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "models": revisions,
                "hardware": hardware,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "sglang",
                        "sgl-nvidia-mega",
                        "flashinfer-python",
                        "nvidia-cutlass-dsl",
                    )
                },
                "args": args,
                "plugin_environment": {SLOTS: "sm148"},
            },
            indent=2,
        )
    )
    _check_admission(args, output)
    results = {}
    for arm in ("stock", "plugin"):
        enabled = arm == "plugin"
        directory = output / arm
        directory.mkdir(exist_ok=True)
        process = None
        _wait_for_gpu_idle_in_ci(timeout=120)
        try:
            with (directory / "server.log").open("w") as log:
                process = popen_launch_server(
                    MODEL,
                    URL,
                    timeout=3600,
                    other_args=args
                    + (["--enable-nvidia-megakernel"] if enabled else []),
                    env={SLOTS: "sm148"},
                    return_stdout_stderr=(log, log),
                )
                info = _request("/server_info")
                assert info["enable_nvidia_megakernel"] == enabled
                assert info["speculative_dspark_block_size"] == 5
                assert info["prefill_attention_backend"] == "trtllm_mla"
                assert info["decode_attention_backend"] == "trtllm_mla"
                (directory / "server-info.json").write_text(json.dumps(info, indent=2))
                probes = []
                # Warm capture, cross into eager B3/B4, retire requests, return to B1.
                for batch in (1, 2, 3, 4, 1):
                    probes.append(_generate([prompt] * batch))
                long_result = _generate([long_prompt])
                assert long_result[0]["meta_info"]["prompt_tokens"] == 32768
                assert "71432" in long_result[0]["text"], long_result
                (directory / "serving.json").write_text(
                    json.dumps({"transitions": probes, "long": long_result})
                )
                quality = run_sgl_eval(
                    SimpleNamespace(
                        eval_name="gsm8k",
                        base_url=URL,
                        model=MODEL,
                        num_examples=200,
                        num_threads=4,
                        repeat=1,
                        temperature=0,
                        seed=42,
                        max_tokens=2048,
                        sgl_eval_thinking=False,
                        sgl_eval_out_dir=directory / "gsm8k",
                    )
                )
                (directory / "quality.json").write_text(json.dumps(quality, indent=2))
                perf = {}
                for batch in (1, 4):
                    _generate([prompt] * batch)
                    perf[batch] = [_measure([prompt] * batch) for _ in range(3)]
                results[arm] = {"quality": quality, "perf": perf, "probe": probes[0][0]}
                (directory / "result.json").write_text(
                    json.dumps(results[arm], indent=2)
                )
        finally:
            if process is not None:
                terminate_and_kill_process_tree(process)
        _route_check((directory / "server.log").read_text(), enabled)
    print(f"Kimi matched artifacts: {output}")
    return results


def test_kimi_k3_b300_accuracy(matched):
    stock, plugin = (matched[arm] for arm in ("stock", "plugin"))
    baseline, candidate = stock["quality"]["accuracy"], plugin["quality"]["accuracy"]
    print(
        f"GSM8K-200 stock={baseline:.4f} plugin={candidate:.4f} delta={candidate - baseline:+.4f}"
    )
    assert math.isfinite(baseline) and 0 < baseline <= 1, "invalid stock baseline"
    assert math.isfinite(candidate) and baseline - 0.005 - 1e-12 <= candidate <= 1
    reference = stock["probe"]["meta_info"]["output_token_logprobs"]
    actual = plugin["probe"]["meta_info"]["output_token_logprobs"]
    compared = 0
    for left, right in zip(reference, actual):
        if left[1] != right[1]:
            break  # Subsequent logprobs would have different conditioning prefixes.
        assert abs(left[0] - right[0]) <= 0.5, (left, right)
        compared += 1
    assert compared, "no shared generated prefix for the numerical smoke check"
    print(f"Shared-prefix logprob comparison: {compared} tokens; unchanged 0.5 bound")


def test_kimi_k3_b300_performance(matched):
    for batch in (1, 4):
        samples = {
            arm: [row["tokens_per_second"] for row in matched[arm]["perf"][batch]]
            for arm in matched
        }
        assert all(
            math.isfinite(value) and value > 0
            for values in samples.values()
            for value in values
        )
        stock, plugin = samples["stock"], samples["plugin"]
        ratio = statistics.median(plugin) / statistics.median(stock)
        print(
            json.dumps(
                {
                    "concurrency": batch,
                    "tokens_per_second": samples,
                    "median_ratio": ratio,
                    "acceptance": {arm: matched[arm]["perf"][batch] for arm in matched},
                }
            )
        )
        # Overlapping ranges are inconclusive, not a speedup claim. A completely
        # slower candidate needs investigation, not an automatically relaxed gate.
        assert max(plugin) >= min(stock), (
            f"C{batch}: all plugin repeats slower than all stock repeats"
        )
