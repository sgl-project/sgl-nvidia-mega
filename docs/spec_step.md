# Speculative step graph (any EAGLE v2 model)

Off by default; each switch installs its SGLang hooks only when set to `1`.

| Switch | Effect |
| --- | --- |
| `SGLANG_NVIDIA_MEGAKERNEL_SPEC_STEP_GRAPH` | One CUDA graph per topk-1 decode step (draft, target verify, draft extend and the work between them), batch size 1 to 8 |
| `SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY` | The step graph's input copies in one launch |
| `SGLANG_SPEC_PACKED_VERIFY_BROADCAST` | One TP broadcast of the verify result instead of three |
| `SGLANG_SPEC_LEAN_SAMPLED_VERIFY` | Chain sampled verify without the expanded temperatures and zero-filled draft probs |
| `SGLANG_SPEC_CHUNKED_VERIFY_SAMPLING` | With the lean verify: the sampling passes split over vocabulary chunks |
| `SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL` | The topk-1 draft-extend argmax, probs and DSA seed gather in two launches |
| `SGLANG_SPEC_FUSED_RELAY` | The relay stash in one scatter, row views for a single request |

Every fusion's outputs are bitwise identical to the path it replaces.
