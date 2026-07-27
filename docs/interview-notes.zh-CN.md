# AI Research Copilot 面试讲解笔记

这个文档记录项目里可以展开讲的设计点、优化点、边界和取舍。它不是 README 的替代品，而是面试前复盘用的讲稿素材。

## 1. 项目定位

不要把项目讲成“我做了一个 RAG 问答系统”。更准确的说法是：

> 面向复杂问题的 AI Research Copilot，通过规划、搜索、阅读、检索增强、记忆召回、引用校验和评估回放，生成可追溯的研究报告。

> 我做的是面向复杂问题的 AI Research Copilot。主流程是规划、搜索、阅读、综合、校验和回放。RAG 在系统里承担上下文 grounding、已读资料缓存和记忆召回，不作为唯一主路径。

主链路：

`plan -> search/read -> synthesize -> verify/evaluate -> replay`

这条链路强调的是复杂问题研究，而不是静态知识库 FAQ。面试官问“为什么不是普通 RAG”时，可以说复杂研究依赖实时信源、精确引用、动态修正和可回放的研究过程，单纯 `question -> top-k chunks -> answer` 很难覆盖这些要求。

## 2. Source Reader 为什么这么设计

深度研究不能只依赖搜索摘要。搜索 API 返回的 snippet 通常太短，容易丢掉关键数字、时间、结论和上下文；但把整篇网页原文直接塞给最终报告模型，又会带来 token 成本、噪声和引用不可控的问题。

所以项目把外部 source reader 设计成：

`search raw_content -> evidence selection/compression -> citation-ready evidence`

当前有三种策略：

- `extract`：离线测试友好的确定性抽取，按 query term 选择相关句子。
- `model_compress`：对 raw content 做结构化模型压缩，产出 `summary/key_excerpts/relevance/limitations`。
- `chunk_rerank_compress`：先把 raw content 切成 overlapping chunks，按 query 选出相关片段，再把命中 chunk 的前后邻居扩展进来，最后交给模型压缩。

这块可以讲成一个具体产出：我没有停留在“搜索结果摘要”，而是把搜索提供商返回的 raw content 读进来，再通过抽取、重排、邻居扩展和压缩变成可引用证据，最后报告只能引用已有 evidence，避免模型自己编 URL 或编来源。

这次额外优化的是一个非常高频的 RAG 失败点：答案被 chunk 边界切开。例如前一个 chunk 只有“根据最新政策”，后一个 chunk 才有“报销上限为 5000 元”，单独看任何一个 chunk 都不够完整。现在外部 web reader 命中高相关 child chunk 后，会默认带上前后各 1 个 neighbor chunk，再按原文顺序拼接后压缩。面试可以说这是在 precision 和 context completeness 之间做折中：child chunk 负责精确召回，neighbor window 负责补足跨 chunk 上下文。

## 3. 和 Open Deep Research 的关系

这个项目现在可以直接讲成“以 Open Deep Research 为主参考的学习和复刻项目”。重点不是照搬代码，而是把 ODR 的核心研究链路拆出来：LangGraph 编排、supervisor/researcher 分工、搜索读取、证据压缩、报告综合、引用和评估。

Open Deep Research 的主版本也不是完整浏览器/PDF reader。它的 Tavily 路径大致是：

`Tavily search(include_raw_content=True) -> summarize_webpage -> researcher loop -> compress_research -> final report`

也就是说，它主要依赖搜索提供商返回 `raw_content`，再进行摘要压缩和最终报告生成。它的 citation 主要是 URL/source 级，不是 PDF 页码级引用系统。

所以本项目当前的外部 web source reader 已经达到 Open Deep Research 风格的 v1 边界：不是浏览器自动化系统，但已经不是只看 snippet 的浅层搜索。

ODR 还会在正式研究前先做 `clarify_with_user`，避免题目太模糊就直接开跑。现在本项目也补了一个结构化 `/v1/research/clarify` 入口：当 topic 过短、过泛或者缺少目标时，先问一句澄清问题，再进入完整研究流程。

把你现在列的那五个边界放到 ODR 里看，结论其实很清楚：

- 外部 reader：ODR 也是 `raw_content` + 压缩 + 引用锁定这条线，并没有把主版本做成完整浏览器自动化。
- PDF reader：ODR 主线也没有把企业级 OCR / 版面理解做成核心 runtime 能力；这不是它的重点边界。
- single-node：ODR 是图编排 + 工具调用 + 评估驱动的研究 agent，不是要证明自己是分布式平台。
- source quality：ODR 主要把 source quality 放在 benchmark / judge / evaluator 层，而不是运行时硬过滤。
- LightRAG / GraphRAG：ODR 主线里没有把完整 GraphRAG runtime 当卖点，更多是研究工作流本身。

所以这些点里，真正值得继续对齐 ODR 的不是“补一个更重的基础设施外壳”，而是：

1. 更像 ODR 的研究循环：规划、搜索、阅读、综合、校验、回放闭环要稳定。
2. 更像 ODR 的评估闭环：保留 LLM judge / Ragas / trace artifact，方便面试和复盘。
3. 更像 ODR 的工具边界：把不必要的运行时复杂度删掉，保留研究助手真正需要的能力。

需要诚实讲清楚的差异：

- ODR 的 supervisor 是 LLM tool loop，会让模型在 `think_tool`、`ConductResearch`、`ResearchComplete` 之间做决策。
- 本项目现在对齐这条边界：Planner 先生成研究计划，`research_supervisor` 再输出 `think_tool` 反思、`ConductResearch` 委派和 `ResearchComplete` 完成条件。
- 每个 `ConductResearch` 会携带 selected tools、web/internal query rewrites、external/internal/hybrid grounding mode、min evidence、min sources 和 sufficiency criteria。确定性 route hints 只作为测试/offline fallback，不作为真实链路的主设计。
- Source quality 和 ODR 一样放在 evaluation / judge artifact 里暴露，不额外做运行时强过滤。

面试可以这样说：

> 我没有一开始就做完整浏览器栈，而是参考 Open Deep Research，把 v1 收敛在 provider raw content、证据压缩、引用锁定和评估回放上。这样链路更稳定，成本也可控。后续如果要增强，可以接入 Tavily extract/crawl、HTML parser 或浏览器工具。

## 4. 本地知识库 Reader

RAG 最容易被忽视的一块其实是解析和分块。向量库、rerank、模型都很重要，但如果文档解析阶段已经把标题、页码、段落结构弄丢了，后面的召回很难救回来。

当前本地 grounding 路径拆成两个阶段：

1. `DocumentReader` 负责文件解析和 segment metadata。
2. `DocumentStore` 负责 contextual retrieval prefixing、paragraph-aware child chunking、LightRAG-inspired entity/relation graph indexing、embedding、dense/BM25 fusion、graph-score fusion、rerank，以及命中后的 parent/neighbor context 扩展。

`POST /v1/documents/ingest` 支持读取本地文件：

- 普通文本、CSV、JSON、XML、YAML 等文本类文件会作为 normalized document segment。
- Markdown 会按标题切成 section segment。
- HTML 会去掉 `script/style/noscript` 噪声，再把 `h1-h6` 标题转换成 section 边界。
- Markdown/HTML section 会保留 `section_heading`、`section_level`、`section_path`、`section_path_parts`、`section_index`、`section_count`。
- PDF 通过可选 PyMuPDF 解析，按页拆成 page segment。
- PDF reader 优先使用 PyMuPDF block extraction 恢复阅读顺序，取不到 block 时 fallback 到 page text。
- PDF segment 会保留 `page_number`、`page_count`、`segment_kind=page`、`reader=pymupdf`、`pdf_text_parse_method`、`text_block_count`、`line_count`、`heading_hints`、页面宽高和旋转等 metadata。
- 如果 PyMuPDF `find_tables()` 能识别表格，会把表格转换成紧凑 Markdown-like 文本拼回 page segment，并记录 `table_count`、`table_cell_count`、`has_tables`。

这比直接把 PDF 或 Markdown 全文拼成一大块更适合面试讲，因为它体现了“先保结构，再做向量化”的工程意识。

## 5. Chunking 策略

内部文档进入 `DocumentStore` 后，会做 paragraph-aware child chunking + indexing-time contextual retrieval prefixing：

- 默认 chunk size 是 900 字符，overlap 是 160 字符。
- 优先按段落组合 chunk，避免随意截断结构。
- 超长段落会按英文和中文句子边界继续拆分。
- 每个 chunk 会注入 title、source、chunk index、URL 和部分 scalar metadata。
- Markdown/HTML 的 `section_path`、PDF 的 `page_number` 会跟随 metadata 进入 chunk。
- 每个 chunk 先生成 contextual retrieval prefix，再把 prefix + chunk 同时写入 dense vector 和 BM25 keyword index。
- Qdrant 检索时先做 dense/BM25 fusion，再通过 reranker 排序。
- 同时维护一个轻量实体/关系共现图，query entity 命中时会补充直接实体 chunk 和关系邻居 chunk，graph score 会和 dense/BM25 候选融合。
- 检索命中的是 child chunk，但返回给 reporter 的 evidence 会扩展成 same-document parent/neighbor context，metadata 里保留 `parent_id`、`parent_context_window`、`child_text_chars` 和 `parent_context_chars`。

这里的关键不是“我切了 chunk”，而是：

> 解析、分块、图索引、向量索引、parent-child 扩展和 rerank 是分层的。解析层保留来源、标题层级和页码；分块层生成更适合精确召回的 child chunks；图索引层维护轻量实体/关系共现信号；检索层做 dense/BM25 + graph score 融合；rerank 层对候选证据排序；最后再回填父级/相邻上下文，避免只给模型一个孤立文本块。这样任何一个环节出问题，都可以在 trace 和 metadata 里定位。

这次优化的可讲产出：

- 新增本地 `DocumentReader`，让解析从检索逻辑里独立出来。
- 新增 `/v1/documents/ingest`，支持从本地文件直接导入 grounding corpus。
- Markdown/HTML 从“整篇文本”升级为 heading-aware section segmentation。
- PDF 从“全文文本”升级为 page/block/table-aware segments，保留页码、文本块、标题 hint、表格和页面布局 metadata。
- 修复长段落按句子切分的边界逻辑，支持英文和中文标点。
- 补了 reader、chunking、API ingestion 的测试和文档。
- 外部 web source reader 增加 selected chunk neighbor expansion，缓解 provider `raw_content` 被固定窗口切断导致的证据不完整问题。

## 6. 为什么不做纯 RAG

纯 RAG 适合静态知识库问答，但复杂研究问题通常需要：

- 把问题拆成多个子问题。
- 实时搜索外部资料。
- 阅读具体来源，而不是只看 snippet。
- 根据已有证据动态修正方向。
- 把引用锁定在已有 evidence 上。
- 对 source quality、citation coverage、unsupported claims 做评估。
- 保留 trace 和 replay，方便复盘为什么生成这个结论。

因此本项目里 RAG 是 grounding layer，不是完整产品主路径。RAG 用在内部资料、已读内容缓存、记忆召回；主流程还是 LangGraph 编排的规划、搜索、阅读、综合、校验和回放。

## 7. 为什么不只用 embedding 相似度

纯 embedding similarity 在研究场景里不一定稳，因为它可能把语义上接近但缺少关键术语的背景段落排到前面。研究问题经常需要精确命中特定年份、指标、方法名、论文名或系统组件。

当前检索链路是：

`contextual retrieval prefix + child chunks -> LightRAG-inspired graph signal + Qdrant dense + SQLite BM25 fusion -> reranker -> parent/neighbor context expansion`

可以这样讲：

> 我没有把 embedding 当成唯一信号。dense vector 提供语义召回，SQLite FTS5/BM25 保留关键词和术语命中，fusion 保证候选集合更稳，最后 reranker 再按 query 相关性重排。这样比单纯 top-k embedding 更适合研究型问答。

## 8. LightRAG 这一块怎么讲

不要说“我完整复现了 LightRAG”，这样容易被追问到图存储、实体关系抽取、dual-level retrieval、增量更新和实验对比。更稳的说法是：

> 我参考 LightRAG 的图增强检索思想，在本地 grounding 层里实现了轻量 entity/relation graph signal。文档进入索引时，系统会从标题、来源和原始 child chunk 中抽取实体，维护 chunk -> entity、entity -> chunk、entity -> neighbor entity 的共现关系。查询时先做 dense/BM25 hybrid retrieval，再用 query entity 命中和邻居关系补充候选，融合 `graph_score` 后交给 reranker。这样可以补足纯 embedding 对短实体名、组件名和跨 chunk 关系不敏感的问题。

当前这块能讲的价值：

- 它不是把 LightRAG 当 buzzword，而是把“图信号 + 向量检索 + rerank”接进了真实检索链路。
- 图索引从原始 chunk 文本抽实体，避免把 `Document`、`Metadata`、`Excerpt` 这种包装字段当成实体。
- graph hit 会体现在 evidence metadata 里，比如 `graph_query_entities`、`graph_matched_entities`、`graph_expanded_entities`、`graph_score`。
- 这层图增强不改变 ODR 主流程；它只是强化内部资料、已读资料缓存和记忆召回的 grounding 能力。

诚实边界也要说清楚：

- 当前是 LightRAG-inspired lightweight graph，不是完整 LightRAG runtime。
- 现在主要是实体共现和邻居扩展，还没有做 LLM 级别的实体/关系类型抽取。
- 图索引随单节点内存和文档索引维护，还不是独立持久化图数据库。
- 后续如果要继续强化，可以加 LLM relation extraction、persistent graph store、relation type、multi-hop traversal 和 retrieval ablation 实验。

## 9. 当前诚实边界

当前已经有：

- LangGraph supervisor/planner/research_supervisor/researcher/reporter/verifier/evaluator workflow
- Tavily raw content source reading
- query-aware extract / model compress / chunk-rerank-compress
- 本地 text/Markdown/HTML/PDF reader
- Markdown/HTML section metadata
- PDF page/block/table metadata
- Qdrant dense + SQLite FTS5/BM25 hybrid grounding
- DashScope/Qwen rerank
- Parent-Child retrieval context expansion
- LightRAG-inspired entity/relation graph augmentation
- session/summary/canonical memory
- citation-locked report synthesis
- RAG/source/citation evaluation
- trace、checkpoint 和 replay

当前不要包装成：

- 完整浏览器级/分布式 deep research 平台
- 完整浏览器自动化 reader
- 企业级 PDF/OCR/复杂版面解析平台
- 分布式 deep research 平台
- 大规模公开 benchmark

Demo artifact 也要注意：`examples/llm-judge-report.json` 里如果出现“分析深度不够、source quality 混杂、部分来源偏弱”，`examples/ragas-report.json` 里如果 faithfulness/context precision 不够漂亮，不一定说明代码差，更可能是 demo 问题和信源选择问题。面试前最好重新跑一套干净 demo，问题要选能搜到论文、官方文档、技术报告、标准或一手资料的方向。

后续增强方向：

- URL extract fallback：搜索结果没有 raw content 时，用 Tavily extract 或 HTML parser 补读。
- HTML paragraph citation：保留标题层级和段落位置，让引用更细。
- PDF page/section/table citation：最终报告 citation 显示 `PDF p.12`、章节标题或表格位置。
- OCR/图片 caption：提升扫描件、图表截图和论文 figure 的解析质量。
- 更强 layout-aware chunking：避免表格、公式、标题和正文被粗暴切开。

## 9.5. 多 Agent 到底有没有必要

不要把项目讲成“所有问题都强行多 Agent”。更稳的说法是：这个系统参考 Open Deep Research 的原则，默认偏向简单路径；只有当问题能拆成多个相对独立的研究方向时，才让 supervisor 并发委派多个 researcher。

可以这样回答：

> 多 Agent 在这个项目里不是为了堆角色，而是为了解决复杂研究问题里的上下文隔离和并行探索。简单事实查询没有必要多 Agent，单 researcher 就够了；但如果问题包含架构对比、论文证据、工程实现、评估指标、风险边界这些独立方向，supervisor 会把它们拆成多个 `ConductResearch`，每个 researcher 独立搜索、阅读、反思和压缩证据，最后 reporter 统一综合，verifier/evaluator 再检查引用覆盖和证据充分性。

这次优化后，每个外部 researcher 不再只是“一次搜索拿结果”，而是有一个受预算约束的 `search/read/reflect` 小循环：

`query -> read/compress raw_content -> check evidence/source sufficiency -> reflect -> next query or stop`

代码里会把每轮的 query、新增证据数、来源数、缺口、reflection、next query 和 completed_reason 写入 `ResearchNote.research_iterations`、checkpoint 和 trace。面试官如果问“你的多 Agent 怎么证明不是假的”，就可以打开 trace 讲：supervisor 负责拆解和委派，researcher 负责局部探索和停止判断，reporter/verifier 负责最后的综合和质量门禁。

诚实边界也要讲清楚：

- 当前不是完全自由的浏览器 Agent，也不是无限自主探索。
- researcher loop 是 bounded loop，默认 `ARC_RESEARCH_MAX_ITERATIONS=3`，避免成本失控。
- 外部阅读仍然是 ODR v1 风格的 provider raw_content reader，不是完整浏览器自动化。
- 这反而更适合个人项目和面试 demo：可控、可回放、可解释。

## 9.6. 你应该按什么顺序学这个项目

如果你的目标是“把这个项目真正学会”，建议按下面的顺序读：

1. 先看 `README.md` 的产品定位和主流程，先知道它到底是做什么的。
2. 再看 `docs/architecture.md`，把 supervisor、researcher、retriever、reporter、verifier 这几层的职责分清。
3. 接着看 `src/agentic_research_copilot/graph_runtime.py` 和 `pipeline.py`，理解一次 run 是怎么串起来的。
4. 然后看 `src/agentic_research_copilot/agents/researcher.py`、`source_reader.py`、`retrieval/store.py`、`document_reader.py`，把“读什么、怎么读、怎么召回、怎么压缩”搞明白。
5. 最后看 `memory/store.py`、`evaluation.py` 和 `tests/test_pipeline.py`，理解记忆、评估和 trace 为什么要这么记录。

如果你只想先抓主线，就先读这 4 个文件：

- `README.md`
- `docs/architecture.md`
- `src/agentic_research_copilot/graph_runtime.py`
- `src/agentic_research_copilot/agents/researcher.py`

这几个文件读通了，基本就能讲清楚“为什么不是纯 RAG、为什么是 ODR 风格、多 Agent 为什么只在复杂问题上出现、researcher loop 为什么要受预算约束”。

## 10. 推荐面试说法

> 我做这个项目时没有把 RAG 当成主路径，而是把它放在研究工作流里的 grounding 层。主流程用 LangGraph 编排：Planner 先拆解子问题，research supervisor 再按 ODR 风格输出 `think_tool`、`ConductResearch` 和 `ResearchComplete`；每个 `ConductResearch` 决定外部搜索、本地知识库和记忆召回的工具组合与 query rewrite。外部搜索会读取 provider raw content 并压缩成 citation-ready evidence，本地文档会先解析成带 metadata 的 document/section/page segments，再做 contextual retrieval prefixing、paragraph-aware child chunking、LightRAG-inspired graph augmentation、Qdrant dense + SQLite BM25 fusion、rerank 和 parent/neighbor context expansion。最后 reporter 只能引用已有 evidence，verifier/evaluator 会检查引用覆盖、source quality 和 faithfulness proxy，整条链路可以通过 trace 和 replay 回看。
## 11. MCP 这里怎么讲

ODR 不是固定配了 filesystem、GitHub 或浏览器这类 MCP server。它的主线设计是 `mcp_config.url + mcp_config.tools + auth_required`，再通过 `MultiServerMCPClient` 把 allowlist 里的工具挂到 researcher tool loop 里。

所以这个项目不要讲成“我照着 ODR 用了某几个 server”，而是讲：

> 我参考 ODR 的 MCP 工具注册机制：运行时通过 `ARC_MCP_SERVER_URL` 和 `ARC_MCP_TOOLS` 注入工具，researcher 可以在 `think_tool`、`web_search`、`mcp_tool` 和 `ResearchComplete` 之间选择。为了让 demo 可复现，我做了一个本地 streamable HTTP MCP workbench，把运行中的研究系统能力暴露成工具：`search_grounding_corpus` 查已入库资料，`recall_project_memory` 召回 session/summary/canonical memory，`inspect_research_runs` 复盘历史 run 的 trace/evaluation，`check_demo_readiness` 检查模型、搜索、embedding、rerank、MCP、资料库、记忆和历史实验是否准备好。这样 MCP 不只是“能调用”，而是能服务真实实验和面试演示。

这三个工具的面试价值：

- `search_grounding_corpus`：说明 MCP 可以复用项目真实 RAG 链路，返回 Qdrant dense、SQLite BM25、parent-child、graph signal、rerank 等检索元数据。
- `recall_project_memory`：说明 memory 不是 prompt 里写死的上下文，而是有 session/summary/canonical 分层和 governance 信息的可召回状态。
- `inspect_research_runs`：说明系统有 trace/evaluation/replay，不是只输出一段最终答案。
- `check_demo_readiness`：说明 demo 前可以自检 provider、工具、资料库、记忆和历史 run，避免现场才发现链路没跑通。
- `search_reference_corpus`、`inspect_runtime_config`、`recommend_demo_questions`：作为可选工具，用于 ODR/PraisonAI 架构学习、运行配置检查和 demo playbook 准备。

诚实边界也要讲清楚：这个本地 MCP server 是 demo/学习用的 controlled server，不是 ODR 官方内置 server，也不是企业 MCP 网关。真正对齐 ODR 的是配置式工具注册、工具 allowlist、researcher tool loop 和 trace-visible evidence。
