# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Deployment query widths and prepared token rows; no runtime dependencies."""

DECODE_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
VERIFY_QUERIES = (4, 5, 6, 7, 8)
MAX_ROWS = 64
MAX_CONTEXT = 262144


def token_widths(
    tokens_per_request: int, *, configured: bool = False
) -> tuple[int, ...]:
    """Keep legacy defaults; explicit q1 capture sizes may use every row to 64.

    Representable rows are not blanket numerical qualification. Broader ordinary
    width comparisons retain failures; configured=True is only adapter opt-in.
    The already-registered width 64 remains admitted, not newly qualified here.
    """
    if type(tokens_per_request) is not int or tokens_per_request not in (
        1,
        *VERIFY_QUERIES,
    ):
        raise ValueError("Kimi K3 requires q1 decode or fixed q4-8 verification")
    if tokens_per_request == 1:
        return tuple(range(1, MAX_ROWS + 1)) if configured else DECODE_WIDTHS
    return tuple(range(tokens_per_request, MAX_ROWS + 1, tokens_per_request))


def prepared_widths(
    tokens_per_request: int, widths: tuple[int, ...] | None = None
) -> tuple[int, ...]:
    """Explicit q1 rows may use 1..64; omitted rows retain legacy preparation.

    The spec separately controls whether its framework width handoff opts in;
    a dense BuildContext range is not itself an explicit graph configuration.
    """
    supported = token_widths(tokens_per_request, configured=widths is not None)
    if widths is None:
        return supported
    if (
        not isinstance(widths, tuple)
        or not widths
        or any(type(rows) is not int or rows not in supported for rows in widths)
        or len(set(widths)) != len(widths)
    ):
        raise ValueError("Kimi K3 prepared widths must be a nonempty supported subset")
    return tuple(sorted(widths))


def baseline_capture_widths(
    tokens_per_request: int, capture_token_rows: tuple[int, ...]
) -> tuple[int, ...]:
    """Baseline plus explicit capture rows, restricted to the active query mode.

    Capture sizes are token rows, not request counts. Verification rounds up to
    complete requests; rows above capacity and ordinary-only rows are excluded.
    """
    token_widths(tokens_per_request)
    if not isinstance(capture_token_rows, tuple) or any(
        type(rows) is not int or rows <= 0 for rows in capture_token_rows
    ):
        raise ValueError("capture_token_rows must be a tuple of positive integers")
    q = tokens_per_request
    rounded = {((rows + q - 1) // q) * q for rows in capture_token_rows}
    return tuple(
        sorted(
            rows
            for rows in {*DECODE_WIDTHS, *rounded}
            if rows <= MAX_ROWS and rows % q == 0
        )
    )


def mla_max_seq_len(context_length: int, tokens_per_request: int) -> int:
    """Bound MLA scheduling, without changing native tail-inclusive KV lengths."""
    token_widths(tokens_per_request)
    if type(context_length) is not int or not 1 <= context_length <= MAX_CONTEXT:
        raise ValueError(f"Kimi K3 requires context length in 1..{MAX_CONTEXT}")
    # Native target verification adds q to its graph's context scheduling bound.
    # This is a conservative hint, not extra admitted context or a cache resize.
    maximum = context_length + (tokens_per_request if tokens_per_request > 1 else 0)
    if maximum > (1 << 31) - 1:
        raise ValueError("Kimi K3 MLA scheduling length exceeds signed int32")
    return maximum
