# `suffix` / `stop` Feature Validation Summary

Date: 2026-07-24
Branch: `feat/2026-07-23/suffix_stop`
Baseline: `origin/main` at `0a01a4ce`
Reviewed implementation: `03dd06c1`

## 中文

### 结论

- 已覆盖 `suffix`、`stop`、`capture`、`lazy`、`max_tokens` 的单独、组合、嵌套、动态 grammar、跨 token 和大 grammar 场景。
- 修复了 3 个正确性问题：序列化元数据缺失、动态 `TagDispatch` 的 marker/capture 语义、嵌套预算在 marker 前截断时的状态处理。
- 普通 grammar 不再为 `suffix` / `stop` 元数据付出固定内存成本；`Rule` 大小恢复为与 `origin/main` 相同的 88 bytes。
- 修复了 `max_tokens` 配合尚未闭合的 suffix/stop body 时的二次扫描；代表性 4096-byte 用例从 447.1 ms 降到 4.28 ms，恢复近线性增长。
- 已按 `origin/main` 重新审查差异并移除无关行为变化。实现复用了现有 lazy、FSM 和 `TagDispatch` 路径，只在确实使用该功能时启用额外元数据、capture event 和 byte history。
- 所有修复均已本地提交，未 push。

### 主要修复

1. 将 grammar 序列化版本从 15 升至 16，并序列化 sparse suffix/stop 元数据，保证 round trip 后语义不丢失。
2. 为动态 `TagDispatch` marker 建立零宽 capture event，正确支持变长 marker、重复 marker、`stop_capture` 和父级 capture。
3. 在 `max_tokens` 到达边界时，仅保留仍合法的派生；允许已经完整匹配 body 的 suffix/stop occurrence 在不消费 marker 的情况下完成。
4. 将原先放在每个 `Rule` 上的 feature 字段移入按 rule id 排序的 sparse `SuffixStopInfo`，消除普通 grammar 的固定内存回归。
5. 仅在 grammar 同时含有预算和 marker rule 时维护对应 byte history 和扫描逻辑，避免普通 capture grammar 进入功能专用路径。
6. 对预算化 marker body 的 FSM 匹配状态做按 occurrence 的增量缓存，并在 reset/rollback 时清空，消除每个 token 从头重扫 body 的 O(n²) 行为。
7. 新增一个 token 同时跨越 body、marker 和后继 rule 的 fixed/regex × suffix/stop 回归测试。

### 实现审查

单纯把 `suffix` / `stop` 展开成普通 grammar rule 不能完整表达当前语义：`stop` 必须从外层 capture 中隐藏 marker，marker 可以是变长正则，一个 tokenizer token 可以跨越 marker 边界，而且 `max_tokens` 可能在 marker 出现前结束 occurrence。因此保留运行时 capture event 是必要的。

最终实现仍尽量沿用仓库已有结构：

- 语言识别继续使用现有 FSM、lazy rule 和 `TagDispatch`。
- 功能元数据集中在 sparse side table，不扩充所有 `Rule`。
- 普通 grammar 不维护 marker 专用 byte history 或 body-progress cache。
- 移除了 repetition expansion、capture preservation 和负 ID 检查中的无关行为变化。

### 正确性与质量验证

- 聚焦测试：`350 passed, 1 skipped`（Lark 与 serialization）。
- 完整 Python 非外部-tokenizer测试：`2933 passed, 1 skipped, 622 deselected`。
- C++ 测试：`61/61 passed`，并在 `-Wall -Wextra -Werror` 下完成构建。
- 对短输入进行穷举，并检查 token mask 与实际 accept 结果一致。
- 覆盖 reset、rollback、fork、序列化 round trip、嵌套预算、fixed/regex marker、动态 named grammar 和跨边界 token。
- 变更文件通过 Black、isort、clang-format、Ruff 及适用的 pre-commit 文件检查。

`hf_token_required` 测试因依赖外部 tokenizer 而未运行。探索中发现的一个失败分支 capture-history 现象在 `origin/main` 上同样存在，不属于本分支回归，因此未在本次变更中修改。

### 性能结果

普通 grammar：

- `sizeof(Rule)`：本分支与 `origin/main` 均为 88 bytes。
- plain、capture、lazy、max_tokens、mixed 和 large grammar 的 compiled-memory 测量与基线一致。
- 硬件计数器中普通解析路径的 instruction 增幅约为 1.1%，capture 路径约为 1.7%；wall-clock 未观察到稳定回归。

预算化 suffix body 的最差路径微基准：

| 输入 bytes | 基线 | 修复前 suffix | 修复后 suffix |
| ---: | ---: | ---: | ---: |
| 64 | 0.063 ms | 0.203 ms | 0.078 ms |
| 256 | 0.228 ms | 2.088 ms | 0.292 ms |
| 1024 | 0.924 ms | 31.086 ms | 1.042 ms |
| 4096 | 3.658 ms | 447.107 ms | 4.277 ms |

修复前后数据来自独立运行；基线列取自修复后的同轮测量。

含组合功能的大 grammar：

| Rules | Parse | Compile | Accept step | Compiled memory |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 136 µs | 996 µs | 14 µs | 53.6 KiB |
| 128 | 524 µs | 4225 µs | 44 µs | 213.7 KiB |
| 512 | 2106 µs | 22764 µs | 175 µs | 854.2 KiB |

这些本地微基准用于判断增长趋势，不代表跨机器的绝对性能承诺；结果显示解析、匹配和内存均保持近线性增长。

### 提交

- `18289406 support suffix and stop.`
- `62e44b3e fix: preserve suffix and stop interactions`
- `baeada94 perf: store suffix and stop metadata sparsely`
- `5563693a refactor: keep suffix and stop handling scoped`
- `03dd06c1 perf: cache budgeted marker body progress`

## English

### Conclusion

- Validation covers `suffix`, `stop`, `capture`, `lazy`, and `max_tokens` individually and in mixed, nested, dynamic-grammar, cross-token, and large-grammar cases.
- Three correctness defects were fixed: missing serialization metadata, incorrect marker/capture semantics through dynamic `TagDispatch`, and invalid nested-budget state handling when a rule ends before its marker.
- Ordinary grammars no longer pay a fixed memory cost for suffix/stop metadata. `Rule` is again 88 bytes, matching `origin/main`.
- A quadratic rescan with `max_tokens` and an unclosed suffix/stop body was replaced with incremental FSM progress. A representative 4096-byte case improved from 447.1 ms to 4.28 ms and now scales approximately linearly.
- The complete diff was reviewed against `origin/main`, and unrelated behavior changes were removed. Existing lazy, FSM, and `TagDispatch` paths remain in use; extra metadata, capture events, and byte history are enabled only when required.
- All fixes are committed locally and have not been pushed.

### Main fixes

1. Bumped the grammar serialization version from 15 to 16 and serialized sparse suffix/stop metadata so round trips preserve semantics.
2. Added zero-width capture events for dynamic `TagDispatch` markers, covering variable-length and repeated markers, `stop_capture`, and parent captures.
3. At a `max_tokens` boundary, invalid derivations are removed while suffix/stop occurrences whose bodies already match may complete without consuming a marker.
4. Moved feature-only fields out of every `Rule` into a rule-id-sorted sparse `SuffixStopInfo` table, removing the fixed memory regression for ordinary grammars.
5. Gated marker-specific byte history and scanning on grammars that actually combine budgets with marker rules.
6. Cached body-FSM progress per occurrence and cleared it on reset/rollback, removing the O(n²) full-body rescan on every token.
7. Added regression coverage for one token spanning the body, marker, and following rule across fixed/regex × suffix/stop combinations.

### Implementation review

Pure grammar desugaring cannot express all required behavior: `stop` must hide its marker from outer captures, markers may be variable-length regular expressions, one tokenizer token may cross the marker boundary, and `max_tokens` may end an occurrence before its marker appears. Runtime capture events are therefore necessary.

The final design remains aligned with existing repository structures:

- Language recognition continues to use the existing FSM, lazy-rule, and `TagDispatch` machinery.
- Feature metadata lives in a sparse side table instead of expanding every `Rule`.
- Ordinary grammars do not maintain marker-specific byte history or body-progress caches.
- Unrelated changes to repetition expansion, capture preservation, and negative-ID checking were removed.

### Correctness and quality validation

- Focused Lark and serialization tests: `350 passed, 1 skipped`.
- Full Python suite excluding external-tokenizer tests: `2933 passed, 1 skipped, 622 deselected`.
- C++ suite: `61/61 passed`, with a successful `-Wall -Wextra -Werror` build.
- Exhaustive short-input checks verified agreement between token masks and actual token acceptance.
- Reset, rollback, fork, serialization round trips, nested budgets, fixed/regex markers, dynamic named grammars, and boundary-crossing tokens were covered.
- Changed files passed Black, isort, clang-format, Ruff, and all applicable pre-commit file checks.

Tests marked `hf_token_required` were not run because they require external tokenizers. An exploratory failed-alternative capture-history behavior also reproduces on `origin/main`; it is not a branch regression and was intentionally left unchanged.

### Performance results

Ordinary grammars:

- `sizeof(Rule)` is 88 bytes on both this branch and `origin/main`.
- Compiled-memory measurements match the baseline for plain, capture, lazy, max-token, mixed, and large grammars.
- Hardware counters showed approximately 1.1% more instructions on the plain parsing path and 1.7% on the capture path; no stable wall-clock regression was observed.

Worst-case benchmark for a budgeted suffix body:

| Input bytes | Baseline | Suffix before fix | Suffix after fix |
| ---: | ---: | ---: | ---: |
| 64 | 0.063 ms | 0.203 ms | 0.078 ms |
| 256 | 0.228 ms | 2.088 ms | 0.292 ms |
| 1024 | 0.924 ms | 31.086 ms | 1.042 ms |
| 4096 | 3.658 ms | 447.107 ms | 4.277 ms |

The before/after figures come from separate runs; the baseline column is from the post-fix run.

Large grammars combining the new features:

| Rules | Parse | Compile | Accept step | Compiled memory |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 136 µs | 996 µs | 14 µs | 53.6 KiB |
| 128 | 524 µs | 4225 µs | 44 µs | 213.7 KiB |
| 512 | 2106 µs | 22764 µs | 175 µs | 854.2 KiB |

These local microbenchmarks are intended to characterize scaling rather than promise absolute performance across machines. Parsing, matching, and memory usage all remain approximately linear.

### Commits

- `18289406 support suffix and stop.`
- `62e44b3e fix: preserve suffix and stop interactions`
- `baeada94 perf: store suffix and stop metadata sparsely`
- `5563693a refactor: keep suffix and stop handling scoped`
- `03dd06c1 perf: cache budgeted marker body progress`
