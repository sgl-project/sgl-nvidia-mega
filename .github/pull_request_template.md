<!-- PR title: [Type] scope: title, or [Type] [Hardware] scope: title.
Type: Feat, Fix, Refactor, Perf, Doc, Test, CI, or Chore.
Scope: any module or model name, e.g. deepseek_v41, common, or ci.
Hardware: optional single tag without spaces, in uppercase (NVIDIA, GB300).
Reverts: use GitHub's default title, Revert "original title".
Examples: [Fix] deepseek_v41: Fix the pad-row KV write; [Perf] [GB300] deepseek_v41: Tune the routed-expert GEMM tiles. -->

## Motivation

<!-- Describe the purpose and goals of this pull request. -->

## Modifications

<!-- Detail the changes made in this pull request. -->

## Accuracy Tests

<!-- For each affected model/hardware configuration, give matched accuracy results against native SGLang with Mega disabled, using the same SGLang SHA and evaluation setup. Report the comparison verdict and evidence; do not use a fixed absolute score as the acceptance criterion. See CONTRIBUTING.md. -->

## Speed Tests and Profiling

<!-- For each affected model/hardware configuration, give E2E performance results against the same native SGLang baseline, including relevant fallback widths. Report throughput/latency, relative change, repeated-run variation, commands and any measurement-only overrides. -->

## Checklist

- [ ] `pre-commit run --all-files` passes.
- [ ] Tests under `test/` cover the change.
- [ ] Required SGLang bindings passed normal SGLang CI and are merged, or no binding change is needed.
- [ ] The tested SGLang and Mega SHAs and integration run link are named above.
- [ ] Accuracy and performance E2E results against the matched baseline are attached for each affected supported model/hardware configuration.
- [ ] Required E2E cases ran on the appropriate hardware; missing coverage is listed explicitly.
- [ ] A new model adds one line to `.github/CODEOWNERS` that names its owner.

## Review and Merge Process

1. The `lint` and `Check PR title` checks must pass.
2. Ask a code owner from `.github/CODEOWNERS` to review.
3. Merge with "Squash and merge": the PR title becomes the commit title on `main`.
