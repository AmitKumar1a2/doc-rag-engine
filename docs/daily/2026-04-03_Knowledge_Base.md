# 2026-04-03 Knowledge Base

## Q&A

**Q: How can end-to-end latency be improved in the current RAG pipeline?**

**A:** The main latency driver is the number of sequential LLM calls per request. For `FACT_LOOKUP`, the current flow is classifier LLM -> retriever -> reranker LLM -> answer LLM, so one query can wait on three model invocations. The most practical optimizations are to reduce that call chain, shrink prompt/context size, and only use heavier models where they add clear quality.

**Q: Would lightweight LLMs help?**

**A:** Yes, but model size is only one part of latency. This repo already uses `phi3:mini` for classification/reranking and `llama3.2:3b` for final answers, so further gains may come from replacing one or both helper LLM steps with cheaper non-generative logic, for example rule-first routing and embedding/score-based reranking, then reserving the answer LLM for final generation.

**Q: What is the most likely first experiment?**

**A:** Benchmark route-level latency first, then compare these options: skip LLM reranking for fact lookup, use a smaller answer model for short factual answers, reduce retrieved context size (`k`, `fetch_k`, chunk length), and stream the final response so perceived latency drops even if total generation time stays similar.
