# Resume Demo Summary

## Runtime

- API base: `http://127.0.0.1:8010`
- Strict provider ready: `True`
- Runtime: `langgraph`
- Queue: `celery`
- Retrieval: `qdrant_dense + sqlite_fts5_bm25`
- Reranker: `dashscope_qwen3_rerank`
- MCP loaded: `True`

## Corpus

- Documents: 5
- Sources: 5
- Vector backend: `qdrant_dense`
- Keyword backend: `sqlite_fts5_bm25`

## MCP Readiness

- Score: 1.0
- Passed: 10 / 10
- Failed checks: none

## Runs

- `fl-heterogeneity-comparison`: status `completed`, sources 6, evaluation `True`, trace events 44
- `fl-personalization-design-memo`: status `completed`, sources 6, evaluation `True`, trace events 53

## Boundary

This script seeds controlled excerpts from PDFs for a reliable resume demo. Full PDF ingestion remains available through /v1/documents/ingest, but large PDFs can be slow because strict mode uses real contextualization and embeddings.
