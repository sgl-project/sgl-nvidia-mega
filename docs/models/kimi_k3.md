# Kimi K3 (packed MXFP4)

| GPU | Tensor parallel | Coverage |
| --- | --- | --- |
| GB300 (SM103, 152 SMs); 148-SM B300 (SM103) and B200/GB200 (SM100) with `sm148` | 8 | Decode rows 1–8, 16, 32 and 64; fixed q4–q8 verification within 64 token rows. Admission and tested configurations differ; see [`spec.py`](../../sgl_nvidia_mega/kimi_k3/spec.py). The 148-SM admission has no TP8 serving validation yet; the same gate was validated on Kimi K3.1 at TP16 on GB200 |

The plugin covers the 93-layer language model (24 MLA and 69 KDA layers) at the listed widths; every other call runs the standard path.

## Requirements

- Kimi K3: 8 x GB300 (SM103, 152 SMs), or 8 x 148-SM B300 (SM103) or B200 (SM100) with `sm148` in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (for Kimi K3 it only lowers the SM floor), BF16, context length up to 262144 within the
  native model limit, TP8, DP1 and DCP off,
  FP8 E4M3 KV cache with page size 64, `trtllm_mla` for prefill and decode,
  `flashinfer_mxfp4` with default precision, and full decode CUDA graphs
- Kimi K3 DSpark requires the target-verification interfaces in
  [SGLang #45406](https://github.com/sgl-project/sglang/pull/45406). The tested
  configuration uses `RadixArk/Kimi-K3-DSpark` with five draft proposals (q6);
  the draft model and acceptance remain on SGLang's stock path.
- Kimi K3 checks for CuTe DSL 4.8.0, a coherent FlashInfer Python/cubin/JIT-cache
  wheel trio and the private ABI used by its prepared kernels. The general
  dependency list in the README is not a promise of compatibility with arbitrary newer
  versions; see `sgl_nvidia_mega/kimi_k3/compat.py`.

Kimi K3 validates SGLang's finite build deadline and compiles it into all six
reachable TP8 remote-wait paths; on expiry a rank stores through address 0x0 and
traps.

## Validation

The Kimi K3 adapter is experimental. Initial ordinary-decode validation used
context length 8192 on GB300/TP8: every listed CUDA-graph width and padding
boundary passed dummy-weight
capture/replay, B1 and B16 passed stock-versus-plugin real-weight canaries, and a
specialized-route/captured-replay canary passed, and the GSM8K quality protocol
matched its frozen stock-path reference score. This is bounded functional and
quality evidence; it does not claim performance, unlisted configurations or production
readiness.

Earlier matched real-weight functional checks covered ordinary decode and
five-draft/q6 DSpark at context length 262144, with DCP off and stock draft
execution. Inputs span 32768, 131072, 240000 and 255872 tokens, plus a
262000-input/128-output deployment-edge request. Long B1 and mixed-B2 requests,
short B4 controls, chunked prefill, page boundaries, retirement, request-slot
reuse and full decode graph replay passed the predefined numerical checks.
Long-input output tokens matched stock; short controls can differ within the
numerical bounds. This is not global bitwise equivalence, an AgentX benchmark,
long-context task-quality validation or a performance claim. Other fixed draft
lengths have component coverage, not equivalent long-context serving evidence.

These results belong to the revisions tested, not automatically to every later
composition. Combined default-OFF ordinary-decode and fixed-q6 checks passed
within the recorded scopes, including near-cap requests, mixed lengths, reuse
and graph replay. A separate registered-default width-64 control matched the
published and private implementations exactly for all 64 sequences and 4,096
selected-token log probabilities. That control does not establish native-stock
accuracy or resolve the five numerical failures in the broader configured-width
experiment. Those failures, including at the already-admitted width 64, are not
waived by component tests or the default-OFF additional-width API. These are
bounded results, not a blanket numerical or performance PASS.

Optional prefetch, embedding and selected-width preparation have bounded
correctness evidence and remain dormant. The TokenSpeed MLA alternative has
matched ordinary-decode and fixed-q6 evidence at TP8, DCP off and context length
8192; TRTLLM remains the default. DCP resources and execution interfaces are
included for explicit integration, with matched ordinary-decode and fixed-q6
evidence at TP8 and context length 8192, including native acceptance, state
commit, hidden-state handoff and graph replay. The DCP adapters require additional
framework interfaces; stock-framework imports and explicit unsupported-activation
rejection are separate CPU contracts, not GPU qualification. No DCP feature
admission is added to the registered spec, and no DCP long-context, optional
combined-ON, default activation or performance claim follows from these checks.

Enabling the plugin does not itself enable DSpark: ordinary decode remains
ordinary decode; five-draft/q6 is the qualified speculative configuration when
DSpark is explicitly configured. An admitted shape is not a claim that every
batch, context and optional-provider combination has passed full-model testing.

#### Kimi K3 optional startup providers

The hybrid runtime and DSpark interfaces are upstream in SGLang through
[#45406](https://github.com/sgl-project/sglang/pull/45406), merge
`aa3e34aaebf880d72482cea74dbbfef54da728e2`. API v2 is required. The optional
interfaces below do not expand the registered hardware, model or width guards.

Native SGLang embedding and no cross-layer weight prefetch remain the defaults.
For experiments, set either of these plugin-owned switches before launching
the server with `--enable-nvidia-megakernel`:

| Environment variable | Unset or `0` | `1` |
| --- | --- | --- |
| `SGLANG_KIMI_K3_MEGAKERNEL_PREFETCH` | No optional prefetch preparation/resources | Prepare cross-layer weight prefetch using the existing selective TP8 token-row policy |
| `SGLANG_KIMI_K3_MEGAKERNEL_EMBEDDING` | Native embedding; no optional gather preparation/resources | Prepare the specialized gather for supported native embedding layouts; retain native fallback otherwise |

Use the same settings on every tensor-parallel rank. Only `0` and `1` are
accepted; other values fail during build, before optional resource preparation.
The plugin reads these switches at build time, not in the decode/verification
loop. Restart the server to change them; they are not live graph toggles.
Supplied `inputs_embeds`, vocabulary sharding and native communication semantics
remain unchanged. Neither option enables DCP, alternate MLA or wider batch
admission.

Each option has bounded ordinary-decode and q6 serving correctness evidence.
The prefetch serving screen and embedding component measurements did not
establish a useful benefit in their tested scopes; neither justifies default
activation. Both remain OFF. Their separate proofs do not qualify both options
together or establish arbitrary workload, long-context ON or performance coverage.

#### Kimi K3 optional capture-width build API

`sgl_nvidia_mega.kimi_k3.spec.build_with_capture_widths` is an explicit
adapter-facing alternative to `SPEC.build`. The registered build remains
unchanged; no CLI or environment setting automatically enables this policy.

Separately, [SGLang #49376](https://github.com/sgl-project/sglang/pull/49376)
automatically bounds `BuildContext.widths` by the resolved request limit for
eligible fixed-q6/TP8/DCP-off configurations before calling `SPEC.build`.
That policy retains reachable specialized eager widths, keeps full preparation
when a safe bound is unavailable, and does not invoke this optional API.

Call the alternative on every TP rank, passing the usual model, HF config and
`BuildContext`, plus keyword `capture_token_rows`. `None` preserves the existing
width policy. A tuple selects baseline token rows `1–8,16,32,64` plus the supplied
capture rows. Sizes must be positive integers; duplicates are deduplicated,
and sizes above capacity are excluded. An empty tuple selects the baseline.
Use actual configured capture rows, not a dense capacity range inferred from
`BuildContext.widths`. For q6, request batches 1/2/4 correspond to token rows
6/12/24. Verification rounds up to whole requests, excludes rounded rows above
64 and excludes ordinary-only baseline rows incompatible with its query width.
This does not add ordinary decode to a verification-only provider.

The API checks cross-rank agreement before preparation and snapshots the selected
widths into a new build context. All ranks must call the same build API; this is
not a live toggle. It does not implicitly enable another provider. The prefetch
and embedding environment settings above are still read independently; alternate
MLA and DCP remain unselected.
Normal SGLang startup continues to use `SPEC.build`; an adapter must explicitly
choose the alternative when activating it later.

CPU selection/forwarding tests do not establish GPU correctness or performance.
Wider full-model numerical checks retain unresolved failures; this API is
experimental and default-OFF, not qualification or a fix for those failures.

#### Kimi K3 dormant attention and DCP interfaces

These lower-level interfaces retain tested implementations for explicit future
integration. They are not server switches and do not change `SPEC.build`.

| Capability | Plugin interface | Registered default / integration boundary |
| --- | --- | --- |
| Alternate MLA | `providers.build_resources(..., mla_core="tokenspeed")` | `mla_core="trtllm"`; an adapter must select and prepare the alternative before graph capture |
| DCP8 resources | `providers.build_resources(..., mla_core="tokenspeed", dcp=dcp_resources.DcpBuildOptions())` | `dcp=None`; requires the matching native TP/DCP groups before preparation |
| DCP ordinary decode | `dcp_execution.PreparedDcpHybrid` | Requires the separate SGLang DCP binding extension and its owned pools/buckets |
| DCP fixed-q6 verification | `dcp_verify_execution.PreparedDcpVerify` | Requires the separate SGLang DCP verification binding extension, including native acceptance and state commit |

Stock SGLang at the merge above does not contain those DCP extensions. Importing
the plugin does not require them; attempting the unsupported DCP handoff rejects
explicitly rather than substituting the ordinary pool layout. Callers own native
metadata preparation, graph-safe updates and resource lifetimes. The plugin's
readiness/ownership checks must not be bypassed. These are startup integration
choices, not live backend or graph toggles.

TokenSpeed selection requires `tokenspeed-mla==2.0.2`; DCP requires this selector
explicitly, not the default TRTLLM core. Supplying `dcp` alone is rejected.

The bounded attention/DCP results described above do not qualify arbitrary
combinations with prefetch, embedding, additional widths or long-context DCP.
Their dormant inclusion does not enable them or imply a performance benefit.

## Tests

Kimi K3 full-serving qualification is separate from the unit tests.
