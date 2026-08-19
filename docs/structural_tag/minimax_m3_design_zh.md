# MiniMax M3 Structural Tag 设计文档

## 1. 目标

MiniMax M3 会把工具参数递归序列化成带 namespace 的类 XML 元素。JSON 属性名会同时
成为开始标签和结束标签的名字：

```text
]<]minimax[>[<shipping>
]<]minimax[>[<city>Singapore]<]minimax[>[</city>
]<]minimax[>[</shipping>
```

实现需要同时满足以下要求：

- 运行时生成的开始标签名和结束标签名逐字节完全相同；
- 值、对象、数组继续遵守 JSON Schema；
- 嵌套、并列的每一个动态元素都有独立语义；
- grammar union、concat、优化、token mask、rollback、fork、batch、jump-forward 和 EOS
  全部正确；
- `Grammar` 和 `CompiledGrammar` 的 JSON dump/load 不丢失语义；
- 编译结果不可变，可以被多线程共享；
- 生成热路径开销足够低。

公开使用路径为：

```python
structural_tag = xgr.get_model_structural_tag("minimax_m3", ...)
compiled = xgr.GrammarCompiler(tokenizer_info).compile_structural_tag(structural_tag)
matcher = xgr.GrammarMatcher(compiled)
```

## 2. 为什么纯 CFG 或一个栈都不够

对于长度不受限的运行时名字，需要表达的子语言是：

```text
open_prefix name open_suffix content close_prefix name close_suffix
```

CFG 可以让两个 `name` 都符合相同的正则语言，但不能要求两个任意字符串按相同顺序
完全相等。复制语言 `{w#w}` 不是上下文无关语言。

单独一个栈也不能解决这个等值问题。读取开始标签 `abc` 时依次压栈，弹栈比较得到的是
`cba`，不是 `abc`。栈适合维护已经解析出的元素嵌套关系，但同序比较仍然需要“捕获的
字节区间 + 正向游标”，或者等价的队列/持久化字符串表示。

因此最终方案只给 grammar runtime 增加一个作用域严格受限的 capture/backreference
能力，而不是在 grammar 旁边再运行一套 MiniMax 协议 parser。

## 3. 被否决的方案：grammar-global sidecar

最早的实现给整个 grammar 挂一个 `DynamicTagMatcher` 配置，再把 CFG 接受的每个字节与
第二套协议状态机做交集。这个设计从根上存在三个问题。

第一，它的状态是 grammar-global，而不是 occurrence-local。XGrammar 本身支持 grammar
组合，例如：

```text
P = "plain"
M = DynamicTag("<", name, ">", content, "</", ">")
U = P | M
```

当 CFG 已经通过 `P` 分支接受 `plain` 时，该分支已经完成，理应允许结束。全局 MiniMax
sidecar 不知道当前 derivation 根本没有进入 `M`；普通文本末尾只要长得像半个协议前缀，
sidecar 就可能仍处于“未完成”状态并错误屏蔽 EOS。concat、optional format、嵌套
subgrammar 和多个动态 tag dialect 都存在同类作用域问题。

第二，grammar dump 天然不包含完整语义。每个 mutator、cache key、serializer、Web binding
和组合操作都必须额外搬运一份 metadata，任何遗漏都会静默放宽约束。

第三，token mask 需要重复 grammar matcher 已经做过的工作，还要维护另一套协议专用的
索引和 rollback checkpoint。

最终方案已经完全删除 grammar-global dynamic matcher；`GrammarMatcher` 中也没有一套
MiniMax 专用状态机。

## 4. Grammar-local `DynamicTag` IR

Grammar IR 新增一个可以序列化的宏：

```text
DynamicTag(
  open_prefix,
  name_rule,
  open_suffix,
  content_rule,
  close_prefix,
  close_suffix
)
```

其精确定义是：

```text
open_prefix + captured(name_rule) + open_suffix
+ content_rule
+ close_prefix + backreference(captured_name) + close_suffix
```

`kDynamicTag` 是普通 `GrammarExprType`。EBNF parser/printer、grammar builder、visitor、
mutator、dead-code elimination、rule-reference analysis、union、concat 和 JSON serializer
都会识别它。Structure normalization 遇到嵌在 sequence/choice 内部的 `DynamicTag` 时，
会生成一个 helper rule，因此每个编译后的 occurrence 都拥有独立的 rule-local capture
作用域。

这样从构造上解决了组合问题：没有进入 dynamic rule 的 union 分支根本不会拥有未完成的
动态状态。

## 5. 编译到现有 FSM / Earley runtime

每个 `DynamicTag` occurrence 会被降成以下 rule FSM：

```text
固定 open prefix
  -> CaptureStart               （零宽边）
  -> delimiter-safe name DFA
  -> CaptureEnd                 （零宽边）
  -> 固定 open suffix
  -> RuleRef(content_rule)
  -> 固定 close prefix
  -> BackReference              （每次 scan 一个已捕获字节）
  -> 固定 close suffix
```

`CaptureStart`、`CaptureEnd` 和 `BackReference` 都是可序列化的 `FSMEdge` 类型，直接属于
compiled grammar，没有需要同步的带外 metadata。

### 5.1 name rule 的要求

`name_rule` 会被编译并内联成 byte-regular automaton。它可以包含 byte string、character
class、正则的 choice/sequence、有界或无界 repeat、无环 rule reference，以及直接尾递归。
以下情况会在编译时明确拒绝：

- 依赖 tokenizer 的边：`Token`、`ExcludeToken`、EOS；
- name language 内再次出现 `DynamicTag`；
- 相互递归或非尾递归；
- 内联后会丢失运行时含义的 rule metadata：token/char budget、capture、lazy、temperature、
  suffix/stop metadata。

Rule lookahead 可以保留，因为它只是 token-mask 的边界提示，不属于接受的字节语言；内联
以后，固定 open suffix 本身就提供了边界。

name determinization、repeat expansion、delimiter automaton 和 FSM intersection 都限制在
100,000 个状态以内。超限会明确编译失败，不会无界消耗内存。

### 5.2 delimiter-safe name language

调用方给出的 name language 会和一个确定性的 delimiter-safety automaton 取交集。名字必须
非空，并至少包含一个非 ASCII whitespace 字节。名字不能：

- 包含完整 open suffix 或 close suffix；
- 与后续 suffix 跨越 `name + suffix` 边界形成更早的完整 delimiter；
- 当 open/close prefix 一方是另一方的严格前缀时，以剩余 extension 开头。

该 automaton 使用 KMP prefix state，因此复杂度与 delimiter 长度线性相关。这个限制让开始
标签只有唯一 capture 边界，也避免宽泛 name rule 在输入中每出现一次 delimiter 就永久累积
一个 Earley state。它同时符合 M3 wire format：属性元素名不能以 `/` 开始，也不能包含 `>`。

## 6. Matcher 内部表示

Earley parser 本来就通过 `rule_id` 和 `rule_start_pos` 区分 rule occurrence。动态 tag 进度
复用现有热路径 `ParserState` 字段，不增加 state 大小：

- 在 dynamic FSM 中，`sub_element_id` 和 `partial_codepoint` 分别保存经过 `CaptureStart`、
  `CaptureEnd` 时的 parser row；
- 扫描 `BackReference` 时，`repeat_count` 保存下一个待比较字节的 offset。

这些用途不会与 `DynamicTag` rule 中的 UTF-8 character-class 解码或 grammar repeat 重叠。
现有 state equality/hash 已包含这三个字段，因此有歧义的 derivation 也不会错误合并。

每个 `GrammarMatcher` 独立持有：

- `accepted_bytes_`：已经接受的字节历史；
- `row_byte_end_`：Earley input row 到字节 offset 的映射，也覆盖 atomic-token row；
- 仅在候选 token/string 检查期间使用的临时字节。

capture 区间就是两个 marker row 之间的准确字节区间。这样不需要从 `rule_start_pos` 推导
字节边界，后者继续只负责指向 Earley parent node。`BackReference` 按正向顺序读取这个区间。
嵌套元素天然使用各自的 rule occurrence 和 parent completion state，因此不需要另一个全局
tag stack。

该历史表示也让已有状态操作保持精确：

- `Rollback` 同时截断 parser row 和 byte history；
- matcher copy/fork 复制请求私有历史，但共享 compiled grammar；
- speculative token check 和 jump-forward 使用临时历史，返回前完整恢复；
- atomic-token path 与 byte path 合并前会重映射 after-token row；
- batch 中每个元素使用独立 matcher，不共享 capture 状态。

只有当前 Earley derivation 完成时才允许 EOS。未结束的 backreference 本身是 scanable grammar
state，不可能误判成完成。

## 7. Token mask 算法

可复用 token mask 不可能预先知道请求运行时捕获的名字。因此 compiler 只针对“能够到达
`DynamicTag`”的 rule 计算两层 mask：

1. baseline matcher 把尚无 capture 的 backreference 视为拒绝；
2. wildcard matcher 把它视为一个或多个任意字节。

两者的结果有严格单调含义：

- baseline 接受的 token 一定接受；
- wildcard 拒绝的 token 一定拒绝；
- 中间差集依赖运行时 capture，必须用真实 matcher state 精确检查。

不可能到达 dynamic tag 的 rule 继续使用原有 rule-level cache。反向 rule dependency analysis
还会标记父 rule；这是必要的，因为单个 tokenizer token 可能跨过普通父 rule 边界后再进入或
离开 dynamic tag。

当某个 state 唯一可 scan 的边是 `BackReference` 时，走更快的专用路径：

1. 取得下一个必须匹配的 capture 字节；
2. 在已排序 vocabulary 中二分出以该字节开头的 token 区间；
3. 只验证这个区间，并复用 LCP 与 tokenizer trie subtree range。

除非普通 grammar completion 明确允许，否则 stop/special token 会被清除。使用本地 MiniMax
tokenizer（`vocab_size=200,054`）测得结束标签 backreference mask 时间为：

| 捕获名字 | mask 中位时间 |
| --- | ---: |
| ASCII | 1.958 us |
| UTF-8 中文 | 5.208 us |
| 200 字节 key | 2.750 us |

这里是每次 mask fill 的微秒，不是一次 accept 需要几十毫秒。

## 8. 序列化与兼容性

两条持久化路径都不丢失语义：

- `Grammar.serialize_json()` 保存 `kDynamicTag` IR 以及引用的 rule；
- `CompiledGrammar.serialize_json()` 保存降级后的 FSM，包括 `CaptureStart`、`CaptureEnd` 和
  `BackReference` 边。

load 以后不需要在普通 compiled grammar 之外重建协议 metadata 或 tokenizer index。
Grammar print/parse round trip 也会保留 `DynamicTag(...)`。

当前分支使用 serialization version `v18`。由于 grammar IR 和 compiled FSM 格式都发生了
变化，`v17` dump 在新旧 runtime 两个方向都会被明确拒绝，避免被误判为兼容格式。JSON dump
完整支持；如果某个消费者主动把该宏展开成纯 CFG production，就无法保留同序运行时等值约束。

## 9. 多线程安全与所有权

并发边界与普通 XGrammar 一致：

- 优化后的 `Grammar`、compiled FSM 和 adaptive token-mask cache 在编译结束后不可变，可以
  被多个线程共享；
- capture、byte history、rollback state 和 mask scratch buffer 都属于单个
  `GrammarMatcher`；
- 从同一个 compiled grammar 创建或复制出的不同 matcher 可以并行工作；
- 不支持多个线程同时修改同一个 `GrammarMatcher`。

多线程 compile 时，adaptive mask 仍通过 compiler 已有 mutex 写入。运行阶段没有进程级可变
状态，也没有 lazy shared dynamic-tag cache。

ThreadSanitizer 用例覆盖了共享 compiled grammar 和携带不同运行时名字的 batch matcher。

## 10. MiniMax M3 JSON Schema 转换

M3 是现有 `XMLToolCallingConverter` 的 recursive dialect，不是另一套 schema converter。
这样可以复用 `$ref` 解析、schema 遍历、cache、strict mode、property order、cardinality 和
已有 XML extension point。

| JSON Schema 值 | M3 表示 |
| --- | --- |
| 固定对象属性 `k` | 固定 `<k>...</k>` literal |
| 动态/additional property | 使用生成 key rule 的 `DynamicTag` |
| 嵌套 object | 嵌套 property element |
| array | 重复的固定 `<item>...</item>` |
| string | 不带引号的 scalar text，并排除完整 M3 namespace marker |
| integer / number | JSON number lexical form |
| boolean / null | `true`、`false`、`null` |
| `const` / `enum` object 或 array | 递归展开的固定 element |
| unconstrained value | scalar text，或一个及以上递归命名 child element |

已知固定属性名直接使用 literal，不付出运行时等值开销。只有真正运行时生成的名字才使用
`DynamicTag`。与已知 property 同名的 dynamic key 会通过 codepoint trie 排除。

`propertyNames` 会约束运行时 key；显式 `additionalProperties` schema 决定对应 value，显式
`false` 则禁止动态属性。Converter 保留已有 strict-mode 行为：只写 `propertyNames` 而没有
显式 additional/unevaluated policy 时，会允许满足该 schema 的动态属性。

工具参数 schema 缺失或完全 unconstrained 时，会标准化成
`{"type":"object","additionalProperties":true}`，因为 M3 invocation 的参数必须用具名
element 表示。

### 10.1 Structural-tag 外层

内置 `minimax_m3` builder 支持：

- `reasoning_mode="enabled"`：继续 prompt 已经输出的 `<mm:think>` opener；
- `reasoning_mode="disabled"`：不生成 reasoning prefix；
- `reasoning_mode="auto"`：允许一个可选的完整 `<mm:think>...</mm:think>` prefix；
- auto、required、具名/forced function tool choice；
- effective tool choice 允许时，一次生成一个或多个 invocation。

文本 excludes 使用完整协议 marker。裸 namespace marker 刻意不排除，因为它是所有合法 M3
element 共用的多 token 前缀；排除它会导致更长的 tool-call trigger 永远无法生成。

## 11. 性能设计

实现优先保证生成热路径性能：

- 固定名字仍是普通 literal；
- dynamic name 只在编译期 determinize 一次；
- delimiter safety 编入 name DFA，不在运行时执行第二套 parser；
- `ParserState` 大小不变；
- 只有 grammar 含 dynamic tag 的 matcher 才保留 byte history；
- 纯 backreference mask 只检查一个 first-byte vocabulary range；
- 普通 rule 继续使用已有共享 token-mask cache。

使用本地 MiniMax tokenizer 和强制动态属性 schema：

- compile 中位数：`50.631 ms`；
- compiled heap accounting：`0.148 MB`；
- 请求运行时不会按 vocabulary token 分配 dynamic state。

在 10,000 字节普通文本、其余结构等价的 auto tool-call grammar 对比中：

| 指标 | 只有固定结构 | 可包含 dynamic tag |
| --- | ---: | ---: |
| Compile | 5.883 ms | 57.346 ms |
| Compiled memory | 0.080 MB | 0.190 MB |
| Accept | 28.83 ns/byte | 31.92 ns/byte |

普通文本 accept 增量约 10.7%，来源是 matcher-local byte-history 维护；它是精确 rollback 和未来
dynamic occurrence 所必需的。请求内存中占主导的仍然是 parser history。

以上数据均为当前 checkout 在本地开发机上的中位数；使用 200,054-token MiniMax tokenizer、
16 线程 compiler、关闭 grammar cache，并用 10,000 个 ASCII 字节对比 accept。它们用于本分支的
性能回归基线，不代表跨机器绝对性能。

## 12. 验证情况

测试矩阵覆盖：

- 固定、动态、嵌套、并列、UTF-8、长名字和 delimiter 邻接名字；
- close-name mismatch 拒绝，以及通用 IR 层全部 256 种首字节；
- grammar union、concat、nested helper rule 和跨越所有边界的 token；
- name-language repeat、直接尾递归，以及不安全 grammar 的拒绝；
- tokenizer mask、stop/special token、rollback、fork、reset、jump-forward、batch；
- grammar 和 compiled-grammar dump/load round trip；
- recursive object、array、ref、literal、`propertyNames`、显式 additional-property policy、
  reasoning mode 和 tool choice；
- 多个独立 matcher 并发共享同一个 compiled grammar。

当前 checkout 已验证：

- C++ test binary：`101 passed`；
- 完整 Python suite：`3290 passed, 678 skipped`；
- 定向 ThreadSanitizer 并发用例：`4 passed`，无 TSAN 报告。

Web/Wasm 源码使用同一套 IR 和 FSM 实现；当前环境没有 `emcc`，因此尚未实际验证该构建。

## 13. 明确的支持边界

1. 运行时 name rule 必须符合前述 byte-regular 要求。超出范围会明确失败，不会静默放宽。
2. 当 recursive XML string 的 `pattern`、已识别 `format` 或 length constraint 无法与保留的
   namespace-marker exclusion 取交集时，当前 converter 会拒绝；未知 format 仍只作为 annotation。
3. M3 映射生成 typed scalar-or-children value，不支持任意 mixed XML content。
4. Converter 继承 XGrammar 更广泛的 JSON Schema 边界。例如多个重叠
   `patternProperties` 不会自动构造任意 schema intersection，`any_order` 也不是新的完整 key-set
   automaton。
5. 无法安全表示成 M3 element name 的 key 会被拒绝，而不是擅自定义模型和下游 parser 都不
   认识的转义 wire format。
6. 并发执行流之间共享 `CompiledGrammar`，不要共享一个正在修改的 `GrammarMatcher`。

在这些明确边界内，运行时名字等值、grammar 组合、持久化、rollback、token mask 和并发都属于
一等 grammar 语义，而不是 MiniMax 专用的旁路副作用。
