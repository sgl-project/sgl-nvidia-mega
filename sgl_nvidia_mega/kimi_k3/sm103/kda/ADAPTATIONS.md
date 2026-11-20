# KDA split-cache verification adaptation

The imported payload at plugin parent
`e8cd3731294c8a536c5d3478116088d36af86532` has 43 Python files and aggregate
SHA-256 `c3fde49433b5a43efd9cd75f68db1ce09d41581506492938a29664255d6327c2`.
The inventory/digest test still covers all 43 files. This adaptation changes
only `kda/kernel.py` and `kda/prepared.py` inside that payload.

| File | Parent SHA-256 |
| --- | --- |
| `kernel.py` | `86912ad48d82ac2775b8b13281fa9f9448885dca6bba4823785d12f1618ff533` |
| `prepared.py` | `ace7847cee106fa6472dbe7a249e605f0636ddd03cfa9f84d93e9f1f4e1da5c0` |

`KdaSplit(split_cache=True)` reads committed convolution and FP32 recurrent
state, then writes each token's recurrent checkpoint and dense convolution
window to separate scratch pools. Physical slots and scratch rows have separate
indices; scratch row zero is valid. Persistent pools are read-only. SGLang's
existing acceptance scatter remains the sole commit owner.

`PreparedKda.prepare_verify_caches` and `run_verify` expose this mode for
TP8 fixed q4-8, request counts 1 through `floor(64 / q)`, BF16
projections/convolution state, FP32
recurrent state, and gate mode 1. They compile before execution and validate
the fixed layouts without moving cache contents. Existing decode and
joined-cache verification interfaces remain unchanged.

The adaptation changes only seed addressing, checkpoint destinations,
convolution-window publication, and host preparation/validation. Fused F_B,
gate/convolution/normalization math, recurrence, reduction order, schedules,
and PDL ordering are retained. With split-cache branches removed, the original
kernel body and compute helpers must remain structurally identical to the
parent. GPU validation must additionally compare every checkpoint and output
bitwise with the unchanged joined-cache path at each admitted q, and exercise stock commit,
the following verification step, padding, reordered indices, and graph replay.

The digest update records source identity; it does not itself establish GPU
compilation, numerical parity, state-lifecycle correctness, or serving support.
The q4-8 host extension changes prepared-policy admission and checkpoint-capacity
validation only; it leaves `kernel.py` unchanged. Historical q4 evidence does not
qualify the newly admitted query widths or request counts.
