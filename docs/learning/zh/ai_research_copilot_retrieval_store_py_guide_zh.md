# retrieval/store.py 代码阅读指南

对应源码：

```text
D:\kn\projects\agentic-research-copilot\src\agentic_research_copilot\retrieval\store.py
```

这份文档专门解释 `DocumentStore`。它是这个项目本地 RAG 的核心实现，负责把文档变成可检索的知识片段，再把检索结果变成可以交给 reporter 的 `EvidenceItem`。

## 1. 一句话定位

`store.py` 不是一个简单的“向量数据库包装器”，而是本地 RAG 的检索编排层：

```text
文档入库
-> 分块
-> 生成上下文前缀
-> 生成 embedding
-> 写入 Qdrant / SQLite BM25 / 轻量实体关系索引

查询
-> dense 语义召回
-> BM25 关键词召回
-> graph 实体关系扩展
-> 融合候选
-> rerank 重排序
-> parent context 扩展
-> 返回 EvidenceItem
```

几个英文词先记住：

| 术语 | 含义 |
| --- | --- |
| `embedding` | 把文本转换成数字向量，用于语义相似度检索 |
| `dense retrieval` | 基于向量的语义检索，寻找意思相近的内容 |
| `BM25` | 基于词频和逆文档频率的关键词相关性算法，擅长找精确术语 |
| `rerank` | 对初步召回的候选重新排序 |
| `parent-child retrieval` | 用较小的 child chunk 负责命中，用父文档邻近片段补充上下文 |
| `contextual retrieval` | 在 chunk 前面补充文档背景，减少小片段脱离原文语境的问题 |
| `graph signal` | 用实体共现关系帮助检索相关片段 |

## 2. 它在整条主链路中的位置

```text
ResearchCopilot
  -> DocumentStore
      -> index documents
      -> search local evidence
  -> ResearchAgent / pipeline helper
  -> final_evidence
  -> ReporterAgent
  -> cited report
```

`pipeline.py` 负责创建和装配 `DocumentStore`，`graph_runtime.py` 负责把研究任务推进到研究节点，真正的本地文档检索在这里发生。

当研究路线包含 `vector_retrieval` 时，研究执行逻辑最终会调用：

```python
DocumentStore.search(
    query=...,
    context=...,
    purpose=...,
    limit=...,
)
```

返回值不是原始字符串，而是 `EvidenceItem` 列表。这样后面的 reporter、verifier、evaluation 和 trace 都可以继续使用统一的证据结构。

## 3. 先看三个数据结构

### 3.1 `GraphEntity`

```python
class GraphEntity:
    key: str
    label: str
```

它代表从文本中抽出来的一个实体或重要术语。

- `key`：归一化后的内部键。
- `label`：展示给人看的实体名称。

这里的 graph 不是完整的 GraphRAG 知识图谱。当前实现是轻量实体共现图：如果两个实体出现在同一个 chunk 里，就认为它们存在弱关系。

### 3.2 `DocumentChunk`

它是检索系统真正存储和排序的最小单位，关键字段包括：

| 字段 | 作用 |
| --- | --- |
| `document_id` | 父文档 id |
| `chunk_id` | 当前 child chunk 的唯一 id |
| `title` / `source` / `url` | 来源信息 |
| `text` | 原始 chunk 文本 |
| `contextual_text` | 加入上下文前缀后用于检索的文本 |
| `chunk_index` | 当前 chunk 在文档中的序号 |
| `total_chunks` | 该文档总 chunk 数 |
| `metadata` | 来源、上下文前缀、置信度等元数据 |
| `tokens` | 关键词检索使用的 token |
| `embedding` | 向量检索使用的向量 |

最重要的区别是：

```text
text            -> 保留原始证据
contextual_text -> 用于更稳定地做检索
```

最终返回给报告的内容会以原始 chunk 和父级邻居上下文为主，而不是把模型生成的上下文前缀当成事实证据。

### 3.3 `DocumentStore`

它管理四类状态：

1. 原始文档：`self._docs`
2. 文档 chunk：`self._chunks`
3. SQLite FTS5/BM25 索引：`self._keyword_index`
4. 轻量实体关系索引：`self._entity_chunks`、`self._entity_neighbors`

如果配置了 Qdrant，还会把 dense 向量和 chunk payload 写到 Qdrant collection。

## 4. 构造函数：它装配了什么

重点看 `DocumentStore.__init__()`：

```python
DocumentStore(
    embedding_provider=...,
    collection_name="arc_documents",
    qdrant_url=...,
    qdrant_api_key=...,
    qdrant_location=...,
    chunk_size=...,
    chunk_overlap=...,
    parent_context_window=...,
    graph_enabled=True,
    hybrid_fusion="rrf",
    reranker=...,
    contextualizer_provider=...,
    allow_local_fallback=...,
)
```

### 4.1 `embedding_provider`

负责两个动作：

1. 文档入库时，为 `contextual_text` 生成向量。
2. 查询时，为 query 生成向量。

查询向量和 chunk 向量必须使用同一个 embedding 空间，否则余弦相似度没有意义。

### 4.2 `contextualizer_provider`

负责给每个 chunk 生成索引前缀，例如：

```text
这个片段属于支付对账章节，讨论 PayPal callback repair 和 settlement audit。
```

它不替代原文，而是帮助检索器理解这个片段属于哪个文档、章节和主题。

项目通过 `ResearchModelProvider.contextualize_chunk()` 统一这个接口。真实运行时可以使用真实模型 provider；测试和离线场景可以注入 deterministic test double。

### 4.3 Qdrant 配置

Qdrant 是向量数据库，当前用名为 `dense` 的向量字段保存 embedding。

- `qdrant_url` 有值：连接远程 Qdrant。
- `qdrant_location` 是本地路径：使用本地 Qdrant。
- `qdrant_location=":memory:"`：使用进程内存中的 Qdrant。
- `qdrant_prefer_local=True`：没有远程 URL 时优先使用本地模式。

如果 `allow_local_fallback=True`，Qdrant 不可用时会切到本地 dense 搜索；如果为 `False`，初始化、写入或查询失败会直接抛错。这就是严格真实 provider/demo 模式和测试模式的边界。

### 4.4 检索参数

- `chunk_size`：单个 child chunk 的目标长度。
- `chunk_overlap`：相邻 chunk 的重叠长度，减少边界信息丢失。
- `parent_context_window`：命中 child 后，向前后补几个相邻 child。
- `parent_context_max_chars`：最终补充上下文的最大字符数。
- `hybrid_fusion`：选择 `rrf` 或 `dbsf` 融合 dense 与 BM25。
- `reranker`：最终候选排序器。
- `graph_enabled`：是否启用轻量实体关系信号。

## 5. 文档入库链路

从 `add()` 开始读：

```text
add(...)
-> _with_document_id(...)
-> _index_document(...)
```

### 5.1 `add()` 和 `extend()`

`add()` 把标题、来源、URL、摘要、正文和 metadata 组装为 `EvidenceItem`，然后生成稳定的 `document_id`。

`extend()` 用于批量加入已有的 `EvidenceItem`，并根据文档身份去重。

这一步的设计意义是：上游不需要知道 Qdrant、BM25 或 graph 的细节，只需要提供统一的文档对象。

### 5.2 `_index_document()`

这个函数是入库主链路：

```text
snippet + content
-> _chunk_text
-> 每个 chunk 调 _contextualize_chunk
-> _build_contextual_text
-> _embed
-> 保存到内存
-> 写入 SQLite BM25
-> 写入 graph index
-> 写入 Qdrant
```

一个文档会同时进入多个检索结构：

| 结构 | 用途 |
| --- | --- |
| `self._chunks` | 本地 dense 搜索、父级上下文扩展 |
| `SQLiteBM25Index` | 精确术语和关键词检索 |
| graph index | 实体命中和关系扩展 |
| Qdrant | 远程或本地向量检索 |

### 5.3 `_chunk_text()`

分块不是简单地每隔固定字符数切一刀：

1. 先按段落拆分。
2. 过长段落再按句子边界拆分。
3. 单句仍然太长时才按字符切分。
4. 相邻 chunk 可以保留 overlap。

这种顺序的目的，是尽量让一个 chunk 保留完整论述，而不是把一句话从中间截断。

### 5.4 `_contextualize_chunk()`

这个函数负责调用模型生成 chunk 上下文：

```python
self.contextualizer_provider.contextualize_chunk(...)
```

输入包括：

- 文档标题和来源。
- 文档 excerpt。
- 当前 chunk。
- 当前 chunk 序号和总 chunk 数。
- 文档 metadata。

结果是 `ChunkContextContract`，主要包含：

- `context`：上下文前缀。
- `key_terms`：关键术语。
- `provenance_hint`：来源提示。
- `confidence`：上下文生成置信度。

它还有两项工程设计：

1. 使用 `sha256(chunk_text)` 和 provider 名称构造缓存键，避免重复 contextualization。
2. 允许失败时使用本地 deterministic fallback；严格模式下直接抛出异常。

注意：fallback 是测试和离线可复现能力，不是产品真实模型路径。真实 demo 应该使用真实 provider，并通过严格配置尽早暴露错误。

### 5.5 `_build_contextual_text()`

它把以下信息拼成真正用于 embedding 和 BM25 的文本：

```text
Retrieval context
Document title
Source
Chunk position
Context key terms
URL
Metadata
Original excerpt
```

因此一个脱离正文的短片段，也能带着文档标题、来源和上下文参与检索。

## 6. 查询链路

从 `search()` 开始读：

```text
search(query, context, purpose)
-> query_text
-> query embedding
-> dense prefetch
-> BM25 merge
-> graph merge
-> rerank
-> parent context expansion
-> EvidenceItem
```

### 6.1 query、context、purpose 会合并

```python
query_text = " ".join(
    part for part in [query, context or "", purpose or ""]
    if part
)
```

这说明检索不是只看用户的一句关键词：

- `query`：真正要查的问题。
- `context`：当前研究任务的背景。
- `purpose`：这次检索想解决什么证据缺口。

合并后的 `query_text` 同时用于 token 化和 embedding，最后也写入返回结果的 metadata，方便 trace 和调试。

### 6.2 dense 语义召回

如果 Qdrant 可用，调用 `_search_qdrant()`：

1. 查询 embedding。
2. 从 Qdrant 取较大的候选集，而不是只取最终 `limit`。
3. 读取 chunk payload。
4. 结合 Qdrant 语义分数、关键词覆盖率和短语命中分数。

当前 Qdrant 预取阶段的大致权重是：

```text
semantic_score  * 0.68
coverage_score  * 0.24
phrase_score    * 0.08
```

如果 Qdrant 不可用，调用 `_search_local()`，遍历内存中的 chunk，用余弦相似度计算 dense 分数。

这不是两个完全不同的产品路径，而是同一个检索接口的远程 backend 和本地 backend。

### 6.3 BM25 关键词召回

`_merge_keyword_candidates()` 调用：

```python
self._keyword_index.search(query_tokens, limit=...)
```

`SQLiteBM25Index` 在 `retrieval/fulltext.py` 中使用 SQLite FTS5 的 `bm25()` 排名。

BM25 对这些内容尤其有用：

- 精确类名。
- 配置项。
- 错误码。
- 版本号。
- 协议名。
- 罕见的项目术语。

例如用户搜索 `ZKMERKLE-481` 时，纯 embedding 可能不稳定，但 BM25 可以直接命中包含该精确 token 的 chunk。

### 6.4 dense 与 BM25 的融合

`_fuse_dense_keyword_score()` 支持两种策略：

#### `rrf`

`RRF` 是 `Reciprocal Rank Fusion`，中文可以理解为“倒数排名融合”。它不强行比较两个 backend 的原始分数，而是根据各自排名计算贡献：

```text
排名越靠前，贡献越大
排名越靠后，贡献越小
```

它适合 dense 分数和 BM25 分数不在同一尺度的情况。

#### `dbsf`

`DBSF` 是 `Distribution-Based Score Fusion`，根据分数分布做融合。当前代码保留了这个可选策略，默认使用 `rrf`。

最终 metadata 会记录：

- `bm25_score`
- `bm25_rank`
- `fusion_score`
- `fusion_algorithm`
- `retrieval_stage`

所以一次召回为什么排在前面，是可以解释和调试的。

### 6.5 轻量 graph signal

`_index_graph_chunk()` 会为每个 chunk 抽取实体和重要术语：

```text
实体 -> 出现在哪些 chunk
实体 <-> 实体 -> 在同一 chunk 中共现了多少次
```

查询时 `_search_graph()`：

1. 从 query 中抽取实体。
2. 找到直接包含这些实体的 chunk。
3. 沿实体邻居扩展少量候选。
4. 计算 `graph_score`。
5. 在 `_merge_graph_candidates()` 中给候选加权。

当前实现是 LightRAG-inspired，也就是借鉴了轻量图增强思路，但不是完整 LightRAG 或完整 GraphRAG：

- 没有构建完整知识图谱。
- 没有社区发现和多级图摘要。
- 没有图数据库。
- 主要是实体共现和邻居扩展。

面试时应该诚实地说成“轻量实体关系信号”，不要说成“完整 GraphRAG”。

### 6.6 rerank 重排序

`_rerank()` 把候选交给注入的 `BaseReranker`。

默认是 `RuleBasedReranker`，它会：

- 先按融合分数排序。
- 给文档第一个 chunk 一个小 bonus。
- 对同一来源重复命中做轻微多样性惩罚。

也可以注入 `DashScopeReranker`，让真实 rerank 模型对候选重新排序。`DashScopeReranker` 支持：

- 只对前若干候选请求模型。
- 把基础检索分数和 rerank relevance score 合并。
- 请求失败时 fallback 到规则重排。
- `allow_fallback=False` 时严格报错。

这个设计把“候选召回”和“最终排序”分成了两个阶段，后续替换 reranker 不需要重写 DocumentStore。

### 6.7 parent context 扩展

最终排序后，`_parent_context_for_child()` 会找到同一文档中相邻的 child chunk：

```text
命中的 child chunk
+ 前后 parent_context_window 个邻近 chunk
```

返回的内容包含：

- Parent document。
- Source。
- 命中 chunk 的位置。
- URL 和部分 metadata。
- matched child。
- neighbor child。

这就是 parent-child retrieval 的核心：

```text
child chunk 负责精确命中
parent context 负责让 reporter 看懂上下文
```

如果只返回命中的小 chunk，报告模型容易得到半句话或缺少前提；如果一开始就把整篇文档塞进 prompt，又会造成上下文过长和噪声增加。

## 7. 返回给下游的 `EvidenceItem`

`search()` 最后把每个结果转成：

```python
EvidenceItem(
    title=...,
    source=...,
    kind="document-chunk",
    url=...,
    snippet=...,
    content=parent_context,
    score=...,
    metadata=...,
)
```

重点 metadata 包括：

| 字段 | 说明 |
| --- | --- |
| `document_id` | 父文档 id |
| `chunk_id` / `chunk_index` | 命中 child 的身份和位置 |
| `parent_id` / `parent_title` | 父文档身份 |
| `retrieval_strategy` | 当前完整检索策略名 |
| `retrieval_backend` | Qdrant 或 local dense backend |
| `hybrid_fusion` | `rrf` 或 `dbsf` |
| `keyword_backend` | 当前为 `sqlite_fts5_bm25` |
| `context_prefix` | contextual retrieval 前缀 |
| `graph_score` | graph 信号分数 |
| `reranker` / `rerank_provider` | 最终重排器 |
| `grounding_query` | 合并后的 query、context、purpose |

这部分很重要，因为系统不是只返回“找到了一段文字”，而是返回“这段文字从哪里来、为什么被召回、经过了哪些阶段”。

## 8. 文档管理函数

### `list()`

返回当前文档列表，按 source 和 title 排序。

### `profile()`

生成 `CorpusProfile`，包括：

- 文档数。
- 来源数。
- 来源名称。
- 文档类型。
- 高频关键词。
- 是否有参考文档。
- vector backend。
- keyword backend。
- embedding 维度。
- collection 名称。

Planner 和 clarification 阶段可以使用这个 profile 判断本地资料库是否可用。

### `delete()`

删除一个文档时要同时清理：

1. `_docs`
2. `_chunks`
3. contextualization cache
4. BM25 文档
5. graph index
6. Qdrant 中属于该 document id 的 points

这说明删除不是只删一个 Python 列表元素，而是要保持多个索引的一致性。

### `clear()`

清空本地文档、chunk、BM25、graph 和 Qdrant collection，然后重新初始化 collection。

### `close()`

关闭 SQLite BM25 连接和 Qdrant client。

## 9. 这个设计为什么不是简单玩具 RAG

你可以从工程问题出发理解每个设计：

| 实际问题 | `store.py` 的解决方式 |
| --- | --- |
| 小 chunk 脱离文档背景 | contextual retrieval prefix |
| 语义相似但术语不精确 | dense + BM25 hybrid |
| 一个 child 命中但上下文不够 | parent/neighbor context expansion |
| 相关内容通过实体关联出现 | lightweight graph signal |
| 初排结果不够准确 | pluggable reranker |
| 外部 Qdrant 暂时不可用 | local dense fallback 或 strict error |
| 后续无法解释为什么命中 | retrieval metadata 和 score breakdown |

但它也有明确边界：

1. `SQLiteBM25Index` 当前是进程内内存索引，服务重启后需要重新从文档重建。
2. graph 是轻量实体共现图，不是完整 GraphRAG。
3. 当前系统是单节点产品形态，不是分布式向量检索平台。
4. graph entity extraction 对英文术语和预设技术词更友好，复杂中文实体识别不是它的强项。

这些边界不是缺点隐藏起来，而是面试时要主动说清楚的工程取舍。

## 10. 推荐阅读顺序

按这个顺序看源码：

1. `DocumentStore.__init__`
2. `DocumentStore.add`
3. `DocumentStore._index_document`
4. `DocumentStore._contextualize_chunk`
5. `_chunk_text` 和 `_build_contextual_text`
6. `DocumentStore.search`
7. `_search_qdrant` 和 `_search_local`
8. `_merge_keyword_candidates` 和 `_fuse_dense_keyword_score`
9. `_merge_graph_candidates` 和 `_search_graph`
10. `_rerank`
11. `_parent_context_for_child`
12. `retrieval/fulltext.py`
13. `retrieval/rerank.py`
14. `tests/test_retrieval.py`

读每个函数时都问四个问题：

```text
输入是什么？
输出是什么？
它把结果交给了哪个索引或下游模块？
失败时是 fallback 还是 raise？
```

## 11. 面试时怎么讲

可以这样回答：

> 本地 RAG 的核心在 `DocumentStore`。文档入库时，我先按段落和句子做 parent-child chunking，再调用 contextualizer 给每个 child 生成索引上下文前缀，使用 embedding provider 生成向量，同时写入 Qdrant dense index、SQLite FTS5/BM25 keyword index 和轻量实体共现图。查询时把 query、任务 context 和 purpose 合并，先做 dense prefetch，再融合 BM25 精确匹配和 graph signal，随后通过可插拔 reranker 重排，最后把命中的 child 扩展为同文档邻近上下文，统一返回带来源、分数和检索阶段 metadata 的 `EvidenceItem`。这样 reporter 获得的是可追踪证据，而不是一段没有来源的字符串。

如果面试官继续追问 graph，可以补充：

> 这里借鉴了 LightRAG 的图增强思想，但当前实现是轻量实体共现和邻居扩展，不是完整 GraphRAG。它解决的是技术术语、系统组件和相关概念之间的补充召回问题，同时保持单节点、可复现和容易调试。

## 12. 自测问题

看完代码后，你应该能回答：

1. 为什么 embedding 时使用 `contextual_text`，返回证据时还要保留原始 `text`？
2. 为什么不能只使用 dense retrieval？
3. BM25 在什么场景下比 embedding 更有价值？
4. `rrf` 解决了 dense 分数和 BM25 分数不在同一尺度的什么问题？
5. graph index 保存了哪些关系？
6. 为什么先用 child chunk 召回，再扩展 parent context？
7. Qdrant 不可用时，哪一层负责 fallback？
8. `allow_local_fallback=False` 会改变什么行为？
9. `DocumentStore.delete()` 为什么要同时清理多个索引？
10. 当前 graph 为什么不能直接称为完整 GraphRAG？
11. reporter 从 `DocumentStore.search()` 得到的是什么结构？
12. 你如何通过 metadata 判断一次命中走了哪些检索阶段？

如果这 12 个问题能讲清楚，你就真正掌握了这个项目的 RAG 主链路，而不只是记住了几个技术名词。
