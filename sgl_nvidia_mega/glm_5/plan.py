# SPDX-License-Identifier: Apache-2.0
"""Host-only supported request shapes and compiled kernel selection."""

from dataclasses import dataclass

# (draft_tokens, tp_size, batch_size). Eight physical rows serve either shape.
CONFIGS = tuple(
    (draft, tp, batch) for tp in (4, 8) for draft, batch in ((4, 1), (4, 2), (8, 1))
)
MLA_CONFIGS = MOE_CONFIGS = CONFIGS


@dataclass(frozen=True)
class Variant:
    """Internal compilation identity; request shape is kept in ExecutionPlan."""

    op: str
    rows: int
    tp_size: int

    def __post_init__(self):
        if (
            self.op not in ("mla", "moe")
            or self.rows not in (4, 8)
            or self.tp_size not in (4, 8)
        ):
            raise ValueError(f"unsupported GLM5 kernel variant: {self}")

    @property
    def mtp_label(self) -> int:
        return 314 if self.rows == 4 else 718

    @property
    def symbol(self) -> str:
        if self.op == "mla":
            return f"glm5_megamla_mtp{self.mtp_label}_tp{self.tp_size}_kernel"
        suffix = "_tp4" if self.tp_size == 4 else ""
        return f"megamoe_w8a16_fp8_mtp{self.mtp_label}{suffix}_kernel"

    @property
    def source_name(self) -> str:
        return f"{self.op}_w8a16_fp8.cu"


@dataclass(frozen=True)
class ExecutionPlan:
    draft_tokens: int = 4
    tp_size: int = 4
    batch_size: int = 1

    def __post_init__(self):
        if (self.draft_tokens, self.tp_size, self.batch_size) not in CONFIGS:
            raise ValueError(
                "GLM5 supports draft_tokens=4/8 and tp_size=4/8; "
                "batch_size=2 is supported only with draft_tokens=4"
            )

    @property
    def rows(self) -> int:
        return self.draft_tokens * self.batch_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.batch_size, self.draft_tokens, 6144

    def variant(self, op: str) -> Variant:
        return Variant(op, self.rows, self.tp_size)
