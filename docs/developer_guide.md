# Developer Guide

This guide is for developers who change a megakernel in this repository or add one for a new model. It covers how the plugin reaches SGLang's forward, what data a covered step gets, how the DeepSeek-V4.1 package is put together, and the rules a new model follows. For usage and the design summary, see the [README](../README.md).

**No SGLang modeling code calls a megakernel.** SGLang's megakernel layer ([`python/sglang/srt/megakernels/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/megakernels), added in [sgl-project/sglang#44945](https://github.com/sgl-project/sglang/pull/44945)) replaces the loaded model's `forward` with a dispatch wrapper. For each forward, the plugin decides whether to run its own step or hand the call back to the model's `forward`. The model files stay unchanged, so the standard path is always there for steps the plugin does not cover.

## 1. How the plugin plugs into SGLang

```text
pip install -e .   (in a clone of this repository)
  └─ entry point "nvidia_megakernel" ─> plugin.register()       every process, before any model is built
--enable-nvidia-megakernel
  └─ argument resolution: guards.handle_nvidia_megakernel()     before the weights load
  └─ ModelRunner: runtime.install_megakernel()                  after the load, before KV cache sizing
       spec.build() ─> BuiltMegakernel; model.forward = dispatch wrapper
  └─ every forward: built.plan() ─> Step | None
       Step ─> built.forward(step, original, ...)    None ─> the model's own forward
  └─ after decode CUDA graph capture: the capture check
```

### Entry point and registration

`pyproject.toml` declares a general SGLang plugin:

```toml
[project.entry-points."sglang.srt.plugins"]
nvidia_megakernel = "sgl_nvidia_mega.plugin:register"
```

SGLang's plugin loader calls `register()` in every process. It maps each Hugging Face architecture to a `"module:attribute"` spec path ([`sgl_nvidia_mega/plugin.py`](../sgl_nvidia_mega/plugin.py)):

```python
SPECS = (
    ("DeepseekV4ForCausalLM", "sgl_nvidia_mega.deepseek_v41.spec:SPEC"),
    ("DeepseekV4ForCausalLMDSpark", "sgl_nvidia_mega.deepseek_v41.draft:SPEC"),
)

def register() -> None:
    import sglang.srt.megakernels.spec as megakernels  # an ImportError only logs a warning
    for architecture, path in SPECS:
        megakernels.register(architecture=architecture, path=path)
    ...  # test-only hooks
```

Spec modules are imported lazily, only on a server started with `--enable-nvidia-megakernel`.

### Argument resolution: before the weights load

`handle_nvidia_megakernel` ([`megakernels/guards.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/megakernels/guards.py)) runs as a resolution step. It refuses the setup, listing every reason it finds, unless all of these hold:

- a GPU is visible, and `--monokernel-backend` is not also set;
- the plugin is installed and allowed by `SGLANG_PLUGINS`;
- a spec is registered for the model's architecture. It must import, be a `MegakernelSpec`, match `MEGAKERNEL_API_VERSION` and list this GPU's ISA;
- no feature in use is outside `spec.features` (`Feature` lists the SGLang features refused unless a spec accepts them: pipeline parallelism, DP attention, LoRA, PD disaggregation, KV offload, ...);
- the resolved settings meet `spec.requires`: TP size, decode CUDA graph backend, `server_args`, `hf_config`, `env`, dtype; the watchdog leaves room for the cross-GPU wait deadline;
- the spec's own `config_disable_reasons(cfg, hf_config)` returns nothing.

### Install: after the load, before KV cache sizing

`ModelRunner` calls `install_megakernel` ([`megakernels/runtime.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/megakernels/runtime.py)) after every post-load change to the model, for the target and for a draft model. It runs these steps:

1. **One spec on every rank.** Every rank must find the same spec. A draft model without a registered spec keeps its own forward.
2. **Load-time checks.** The requirements SGLang can check without the server arguments are re-checked (TP size, decode CUDA graph backend, HF config, environment). Each rank runs `spec.model_disable_reasons(model, hf_config)` on the loaded model, and checks the GPU has `requires.resident_blocks` SMs and that the model's `forward` is not already replaced.
3. **Build.** It calls `spec.build(model, hf_config, BuildContext(...))`, where:
   - `widths` = `range(tokens_per_request, max_rows + 1, tokens_per_request)`, so for DSpark one width per batch size;
   - `deadline_s` is the deadline for every cross-GPU wait, below `--watchdog-timeout`;
   - `binding` is SGLang's runtime state (section 2).
4. **Agreement.** Every rank must report the same `built.widths`, and a spec with `max_rows > 0` must build at least one.
5. **Dispatch.** It refuses online weight updates for this model.
   - A build with no widths (a draft spec offering only `draft_sampler`) stops here and keeps the model's `forward`.
   - Otherwise it logs `NVIDIA megakernel: active (...)` and replaces `model.forward` with the dispatch wrapper. With decode CUDA graphs on, it also registers the capture check.

### Dispatch: every forward

The wrapper keeps the model's signature:

```python
step = built.plan(input_ids, forward_batch, *args, **kwargs)   # host only
if step is None:
    return original(input_ids, positions, forward_batch, *args, **kwargs)
return built.forward(replace(step, view=binding.forward_view()), original,
                     input_ids, positions, forward_batch, *args, **kwargs)
```

- **`plan` is a host-only decision** from state every rank shares (for DeepSeek-V4.1: the forward mode, the row count and capture mode), so every rank decides alike. Under CUDA graphs it runs at capture, and a replayed graph keeps the decision made then.
- **Only decode and verify steps are covered.** `Step.mode` is `"decode"` or `"target_verify"`; prefill and extend always run the model's own forward. SGLang's breakable and full prefill CUDA graphs capture the inner `layer_model.forward`, which the wrapper does not see, so covering prefill would be a framework extension: an extend step mode, prefill graphs routed through the dispatch, and prefill state in the binding.
- **`forward` takes and returns what the model's forward does.** It must be capturable in a CUDA graph. `original` is the model's bound forward, for checks such as parity.

### Capture check

Once every decode CUDA graph is captured, SGLang:
- first checks that the KV pool and the Engram history, sealed at their first read, were not reallocated since (`binding.assert_live()`);
- refuses a runner none of whose graphs run the megakernel;
- logs the batch sizes that do.

The contract a build returns ([`megakernels/spec.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/megakernels/spec.py), abridged):

```python
class BuiltMegakernel(Protocol):
    widths: tuple[int, ...]   # the rows it built, possibly fewer than requested
    draft_sampler: Optional[Callable[..., Callable[[int], Optional[Callable]]]]
    def summary(self) -> str: ...
    def plan(self, input_ids, forward_batch, *args, **kwargs) -> Optional[Step]: ...
    def forward(self, step: Step, original: Callable, *args, **kwargs) -> Any: ...
```

## 2. What a step gets

[`megakernels/binding.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/megakernels/binding.py) is the one place a megakernel reads SGLang's runtime state:

- **`BuildContext.binding`** is a `MegakernelBinding`. It gives `forward_view()`, `engram_history()` and `engram_host_table(engram)` (one of the model's Engram layers).
  - Each source is sealed at its first read; the capture check raises if it was reallocated since.
- **`Step.view`** is the running forward's `ForwardView`, built at capture for a graph step:
  - `metadata` is the attention backend's forward metadata;
  - `kv_pool` (`KvPoolView`) gives the pool's layout and per-layer key buffers;
  - it also exposes the backend callables a step may still call, such as `attention`, `store_cache` and `low_ratio_sources`.
- **What a megakernel may touch beyond the model it replaces** is listed in `TOLERATED_CALLABLES`, `TOLERATED_WRITES` and `TOLERATED_READS`; anything else stays SGLang's own:
  - SGLang's kernel ops (`sglang.kernels.ops`);
  - the binding above;
  - those three lists.

### The contract test

The plugin reads SGLang by name, so a rename upstream must fail a test here, not a GPU server at startup:

- **Declared reads.** Every module that imports from `sglang` declares what it reads in `SGLANG_READS`, a map from an SGLang file to the names it uses (an example):

  ```python
  SGLANG_READS = {
      "srt/megakernels/spec.py": ("Step", "mode", "rows", "widths", "plan", "forward"),
      "srt/models/deepseek_v4.py": ("model", "start_layer", "end_layer", "layers"),
  }
  ```

- **[`test/unit/test_contract.py`](../test/unit/test_contract.py)** checks, statically against the installed SGLang:
  - that every SGLang import is declared;
  - that SGLang still provides every declared name;
  - that the hook targets resolve;
  - that the tolerated lists in `binding.py` are the ones pinned here;
  - that every `ForwardView` / `KvPoolView` callable a module declares is on the tolerated list.

  Run it against each SGLang version this package supports, and against SGLang main.

### Draft sampler

The object a draft model's spec builds can carry a `draft_sampler` factory instead of, or next to, covered steps. SGLang's DSpark draft sampler ([`dspark_draft_sampler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/dspark_components/dspark_draft_sampler.py)) handles it:

- It calls the factory once with `(*, gamma, max_bs, greedy_draft, noise, draft_tokens, step_logits, step_temperatures)`.
- The factory returns, per batch size, either `None` (SGLang's own proposal) or `step(hidden_states, anchor_tokens) -> (tokens, confidence_features)`.
- The served sizes must agree on every rank, and a step must produce identical tokens on every rank.
- A step writes `draft_tokens[: bs * gamma]` and the sampled rows' `step_logits`, and advances `noise.offset` once per sampled block.

In this repository, [`deepseek_v41/draft.py`](../sgl_nvidia_mega/deepseek_v41/draft.py) is that spec.

## 3. Worked example: `deepseek_v41`

| File | Role |
| --- | --- |
| [`spec.py`](../sgl_nvidia_mega/deepseek_v41/spec.py) | `SPEC`: the `Requirements` (ISA, TP size, dtype, FULL decode CUDA graphs, server arguments, HF config fields, environment), the accepted `Feature`s, `Numerics` (the per-kernel tolerances measured by `test/kernels`), `max_rows`, and `build`. It stays light because the launcher imports it; `build` imports the executor and the kernels. |
| [`plan.py`](../sgl_nvidia_mega/deepseek_v41/plan.py) | `Slots` per width: which kernel fills each slot of a step of that many rows, fixed at build. The row bounds are the measured ones. |
| [`runtime/__init__.py`](../sgl_nvidia_mega/deepseek_v41/runtime/__init__.py) | `build`: each package builds over SGLang's loaded weights (every rank agrees), then the plan per width, then the executor. Also the slot registry. |
| [`runtime/executor.py`](../sgl_nvidia_mega/deepseek_v41/runtime/executor.py) | `MegakernelDecode`: `plan` covers a captured target-verify step of a built width; `forward` runs the step. |

### `plan` and `forward`

- **`plan`** returns `Step(mode="target_verify", rows=rows)` for a target-verify step whose row count was built, while SGLang captures its decode CUDA graphs (or eagerly under a test-only switch), and `None` otherwise.
- **`forward`** follows `DeepseekV4ForCausalLM.forward`, but runs the plugin's step functions over SGLang's loaded modules. It never calls a model-level or decoder-layer `forward`:

```python
for i in range(m.start_layer, m.end_layer):
    layer = m.layers[i]
    ...  # Engram, and the draft's capture of this layer's input
    a_in = boundary.pre(layer, "attn", st, aux)
    part = attention.step(layer, a_in, st)
    m_in = boundary.post(layer, "attn", part, st)
    ...
    out = moe.step(layer, m_in, st)
    boundary.post(layer, "ffn", out, st)
```

### Step functions and slots

- **Stock code first.** Each step function's stock code issues what SGLang's forward issues for the covered configuration. For example, [`moe/step.py`](../sgl_nvidia_mega/deepseek_v41/moe/step.py) follows `DeepseekV2MoE.forward`.
- **A slot swaps in a kernel.** It replaces one piece of a step function with a plugin kernel only where that kernel built and the width's `Slots` select it:

  ```python
  bound = st.bound.get("moe") if st.slots.routed else None
  slot = bound.slots.get(layer.layer_id) if bound is not None else None
  ...
  if slot is not None:
      routed = slot.run(x, topk_output, routed_pre_quant)        # the plugin's routed GEMM pair
  else:
      routed = mlp.experts(x, topk_output)                        # SGLang's routed experts
  ```

- **Choosing slots:**
  - `SHIPPED_SLOTS` run by default; `OPT_IN_SLOTS` run only when listed.
  - `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides the list for tests and bisecting.
  - An empty value runs every step function's stock code.
  - Unknown names and missing dependencies are refused at resolution (`slot_reasons`).
  - A requested slot that builds at no width stops the server.
- **Parity check (test only).** `SGLANG_NVIDIA_MEGAKERNEL_PARITY=1` runs SGLang's own forward on the same eager step first and logs the largest difference of the logits.

## 4. Adding a model

### Layout

```text
sgl_nvidia_mega/<model>/
├── __init__.py   # the HF config fields the kernels are built for
├── spec.py       # MegakernelSpec: requirements, features, numerics, max_rows, build
├── plan.py       # slots per width
├── runtime/      # build, executor (plan / forward), per-step state
├── <sublayer>/   # step functions: stock code first, slots after
└── sm<NN>/       # kernels and their launch code; no SGLang imports
```

### Steps

1. **Register** the architecture in `SPECS` in [`plugin.py`](../sgl_nvidia_mega/plugin.py).
2. **Write the spec.** State in `Requirements` every setting the kernels assume, and accept only the `Feature`s the step handles. Return the checks `Requirements` cannot state from `config_disable_reasons` (before load) and `model_disable_reasons` (after load). Set `api_version` to the `MEGAKERNEL_API_VERSION` it is written against.
3. **Build the stock path first.** The executor and step functions reproduce SGLang's forward with every slot off. Check it with the parity switch and the end-to-end tests before adding any kernel.
4. **Add kernels one slot at a time.** Each slot needs:
   - allocation and compilation in the build;
   - a width bound from measurement in `plan.py`;
   - a kernel test against SGLang's or FlashInfer's op;
   - its tolerance in `Numerics`;
   - unit tests for where it serves and where it declines.

### Test tiers

| Directory | Runs on | Covers |
| --- | --- | --- |
| `test/unit/` | CPU | the plugin entry, the spec checks, the step loop and slot dispatch, the SGLang contract |
| `test/kernels/` | GPU | each kernel against SGLang's or FlashInfer's op, the cross-GPU wait deadlines (fail-stop) |
| `test/e2e/` | GPU | serving accuracy with the switch on and off |

### Rules

These are the [README's design rules](../README.md#design), applied to a new model:

- **Built at load.** Allocate and compile everything for every covered width in `build`. Nothing is compiled or allocated inside a step.
- **Graph-safe coverage.** `plan` decides from host state that is the same on every rank, such as the forward mode, the row count and whether graphs are being captured.
- **Fail-stop synchronization.** Every wait on another GPU uses `BuildContext.deadline_s`. A rank that misses it stops the step, so no token from a failed step is returned.
- **No silent fallback.** Refuse an unsupported setup with every reason, and make every rank agree on what was built. A step `plan` does not cover runs SGLang's forward by design; it is not a fallback from a failed build.
