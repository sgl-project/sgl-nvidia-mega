# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU selection contracts; no kernel execution or numerical qualification."""

import unittest

from sgl_nvidia_mega.kimi_k3.shape_policy import (
    DECODE_WIDTHS,
    baseline_capture_widths,
    prepared_widths,
    token_widths,
)


class TestCaptureWidths(unittest.TestCase):
    def test_ordinary_baseline_plus_configured_rows(self):
        self.assertEqual(baseline_capture_widths(1, ()), DECODE_WIDTHS)
        self.assertEqual(baseline_capture_widths(1, (2, 4, 128)), DECODE_WIDTHS)
        self.assertEqual(
            baseline_capture_widths(1, (33, 17, 33, 9)),
            (*range(1, 10), 16, 17, 32, 33, 64),
        )

    def test_verify_rounds_token_rows_not_request_counts(self):
        self.assertEqual(
            baseline_capture_widths(6, (1, 7, 13, 59, 61)), (6, 12, 18, 60)
        )
        self.assertEqual(baseline_capture_widths(6, (6, 12, 24)), (6, 12, 24))
        self.assertEqual(baseline_capture_widths(6, (64, 128)), (6,))

    def test_all_supported_query_modes_and_boundary_rows(self):
        for q in (1, 4, 5, 6, 7, 8):
            for row in range(1, 130):
                with self.subTest(q=q, row=row):
                    selected = baseline_capture_widths(q, (row,))
                    self.assertEqual(prepared_widths(q, selected), selected)
                    self.assertTrue(
                        all(1 <= value <= 64 and value % q == 0 for value in selected)
                    )
                    self.assertTrue(
                        {value for value in DECODE_WIDTHS if value % q == 0}
                        <= set(selected)
                    )
                    rounded = ((row + q - 1) // q) * q
                    self.assertEqual(rounded in selected, rounded <= 64)

    def test_invalid_inputs_reject(self):
        for rows in (None, [1], "1", 1, (0,), (-1,), (True,), (1.0,), ("1",)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                baseline_capture_widths(1, rows)
        for q in (0, 2, 3, 9, True, 1.0):
            with self.subTest(q=q), self.assertRaises(ValueError):
                baseline_capture_widths(q, (1,))

    def test_existing_default_and_subset_policies_are_unchanged(self):
        self.assertEqual(token_widths(1), DECODE_WIDTHS)
        self.assertEqual(prepared_widths(1, (2, 4)), (2, 4))
        self.assertEqual(token_widths(6), tuple(range(6, 65, 6)))
        self.assertEqual(prepared_widths(6, (12, 24)), (12, 24))
