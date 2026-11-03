# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU checks of registered startup wiring; provider execution is mocked."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sgl_nvidia_mega.kimi_k3 import compat, providers, spec

PREFETCH = "SGLANG_KIMI_K3_MEGAKERNEL_PREFETCH"
EMBEDDING = "SGLANG_KIMI_K3_MEGAKERNEL_EMBEDDING"


class TestStartupOptions(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop(PREFETCH, None)
        os.environ.pop(EMBEDDING, None)
        self.validate = self.enterContext(
            patch.object(compat, "validate_runtime_compatibility")
        )
        self.enterContext(
            patch.object(
                spec, "get_model", return_value=SimpleNamespace(context_length=262144)
            )
        )
        self.memory = self.enterContext(
            patch.object(providers, "_cuda_memory", return_value={})
        )
        self.build = self.enterContext(patch.object(providers, "build_resources"))

    def build_registered(self, q):
        self.build.return_value = SimpleNamespace(tokens_per_request=q)
        context = spec.BuildContext(tuple(range(1, 65)), q, 30.0, object())
        return spec.SPEC.build(SimpleNamespace(language_model=object()), None, context)

    def test_options_forward_independently_for_decode_and_verify(self):
        for q in (1, 6):
            for prefetch, embedding in (
                (None, None),
                ("0", "0"),
                ("1", "0"),
                ("0", "1"),
                ("1", "1"),
            ):
                with self.subTest(q=q, prefetch=prefetch, embedding=embedding):
                    for key, value in ((PREFETCH, prefetch), (EMBEDDING, embedding)):
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                    built = self.build_registered(q)
                    kwargs = self.build.call_args.kwargs
                    options = {
                        key: value
                        for key, value in kwargs.items()
                        if key in ("enable_prefetch", "enable_embedding")
                    }
                    expected = {}
                    if prefetch == "1":
                        expected["enable_prefetch"] = True
                    if embedding == "1":
                        expected["enable_embedding"] = True
                    self.assertEqual(options, expected)
                    self.assertEqual(kwargs["widths"], spec.token_widths(q))
                    self.assertEqual(kwargs["tokens_per_request"], q)
                    self.assertEqual(kwargs["context_length"], 262144)
                    self.assertEqual(built.tokens_per_request, q)
                    self.assertNotIn("dcp_owner", kwargs)
                    self.assertNotIn("mla_core", kwargs)

    def test_invalid_options_fail_before_runtime_preparation(self):
        for key in (PREFETCH, EMBEDDING):
            for value in ("", "true", "false", "2", "-1", " 1"):
                with (
                    self.subTest(key=key, value=value),
                    patch.dict(os.environ, {key: value}),
                ):
                    with self.assertRaisesRegex(ValueError, key + " must be 0 or 1"):
                        self.build_registered(6)
        self.validate.assert_not_called()
        self.memory.assert_not_called()
        self.build.assert_not_called()

    def test_build_snapshots_options_without_mutating_existing_resources(self):
        first = self.build_registered(1)
        first_kwargs = self.build.call_args.kwargs.copy()
        with patch.dict(os.environ, {PREFETCH: "1", EMBEDDING: "1"}):
            second = self.build_registered(6)
        self.assertIsNot(first.resources, second.resources)
        self.assertEqual(self.build.call_count, 2)
        self.assertNotIn("enable_prefetch", first_kwargs)
        self.assertNotIn("enable_embedding", first_kwargs)
        self.assertIs(self.build.call_args.kwargs["enable_prefetch"], True)
        self.assertIs(self.build.call_args.kwargs["enable_embedding"], True)
        # Changing the environment is consumed by a subsequent build only.
        self.assertEqual(first.tokens_per_request, 1)
        self.assertEqual(second.tokens_per_request, 6)

    def test_capture_width_api_preserves_default_and_selects_explicit_rows(self):
        from sglang.srt.kernel_plugins.nvidia import binding

        for q, captures, expected in (
            (1, None, (2, 4)),
            (6, None, (12, 24)),
            (1, (), spec.WIDTHS),
            (1, (9, 17), (*range(1, 10), 16, 17, 32, 64)),
            (6, (1, 7, 13, 61), (6, 12, 18)),
        ):
            with self.subTest(q=q, captures=captures):
                context = spec.BuildContext(
                    (2, 4) if q == 1 else (12, 24), q, 30.0, object()
                )
                self.build.return_value = SimpleNamespace(tokens_per_request=q)
                order = []

                def prepare(*args, trace=order, query=q, **kwargs):
                    trace.append("prepare")
                    return SimpleNamespace(tokens_per_request=query)

                self.build.side_effect = prepare

                def gather(value, group=None, trace=order):
                    trace.append("agree")
                    return [value] * 8

                with patch.object(
                    binding, "_gather_tp", side_effect=gather
                ) as gathered:
                    built = spec.build_with_capture_widths(
                        SimpleNamespace(language_model=object()),
                        None,
                        context,
                        capture_token_rows=captures,
                    )
                self.assertEqual(order, ["agree", "prepare"])
                self.assertEqual(
                    gathered.call_args.args[0],
                    (
                        captures is not None,
                        q,
                        captures,
                        expected if captures is not None else context.widths,
                    ),
                )
                self.assertEqual(built.widths, expected)
                self.assertEqual(self.build.call_args.kwargs["widths"], expected)
                self.assertEqual(self.build.call_args.kwargs["tokens_per_request"], q)
                for optional in (
                    "enable_prefetch",
                    "enable_embedding",
                    "mla_core",
                    "dcp",
                ):
                    self.assertNotIn(optional, self.build.call_args.kwargs)
                self.assertEqual(context.widths, (2, 4) if q == 1 else (12, 24))

    def test_capture_rank_disagreement_stops_before_provider_preparation(self):
        from sglang.srt.kernel_plugins.nvidia import binding

        for captures in (None, (17,)):
            with (
                self.subTest(captures=captures),
                patch.object(
                    binding,
                    "_gather_tp",
                    side_effect=lambda value, group=None: (
                        [value] * 7 + [("different",)]
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "ranks.*differ"),
            ):
                spec.build_with_capture_widths(
                    SimpleNamespace(language_model=object()),
                    None,
                    spec.BuildContext((2, 4), 1, 30.0, object()),
                    capture_token_rows=captures,
                )
        self.validate.assert_not_called()
        self.memory.assert_not_called()
        self.build.assert_not_called()

    def test_capture_api_invalid_rows_stop_before_preparation(self):
        for rows in ([1], (0,), (True,), (1.5,)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                spec.build_with_capture_widths(
                    SimpleNamespace(language_model=object()),
                    None,
                    spec.BuildContext((2, 4), 1, 30.0, object()),
                    capture_token_rows=rows,
                )
        self.validate.assert_not_called()
        self.build.assert_not_called()

    def test_registered_build_remains_default_without_extra_collectives(self):
        from sglang.srt.kernel_plugins.nvidia import binding

        self.assertIs(spec.SPEC.build, spec._build)
        with patch.object(
            binding,
            "_gather_tp",
            side_effect=AssertionError("new collective on default path"),
        ):
            self.assertEqual(self.build_registered(1).widths, spec.WIDTHS)
            self.assertEqual(self.build_registered(6).widths, tuple(range(6, 65, 6)))
