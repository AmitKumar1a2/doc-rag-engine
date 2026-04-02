# 2026-04-03 Knowledge Base

## Q&A

**Q: How can end-to-end latency be improved in the current RAG pipeline?**

**A:** The main latency driver is the number of sequential LLM calls per request. For `FACT_LOOKUP`, the current flow is classifier LLM -> retriever -> reranker LLM -> answer LLM, so one query can wait on three model invocations. The most practical optimizations are to reduce that call chain, shrink prompt/context size, and only use heavier models where they add clear quality.

**Q: Would lightweight LLMs help?**

**A:** Yes, but model size is only one part of latency. This repo already uses `phi3:mini` for classification/reranking and `llama3.2:3b` for final answers, so further gains may come from replacing one or both helper LLM steps with cheaper non-generative logic, for example rule-first routing and embedding/score-based reranking, then reserving the answer LLM for final generation.

**Q: What is the most likely first experiment?**

**A:** Benchmark route-level latency first, then compare these options: skip LLM reranking for fact lookup, use a smaller answer model for short factual answers, reduce retrieved context size (`k`, `fetch_k`, chunk length), and stream the final response so perceived latency drops even if total generation time stays similar.


**Q: Reranking is still slow even when classifier and reranker use the same model. What should we change?**

**A:** Model identity alone does not remove rerank latency because reranking is still a separate LLM call with a large candidate prompt. A practical optimization is adaptive rerank fast-mode: skip reranking when retrieval is already confident (for example, small candidate set or strong score gap between top hits), and only call reranker for ambiguous cases.

**Q: Which tuning knobs were added for rerank latency control?**

**A:** `RAG_FACT_FAST_MODE` (default `true`) toggles adaptive rerank skipping, and `RAG_RERANK_GAP_THRESHOLD` (default `0.12`) controls how large the top-score confidence gap must be to bypass reranking.

**Q: What does a full latency log tell us about bottlenecks for one query?**

**A:** If routing, reranking, and answer generation are logged separately, the bottleneck is whichever stage dominates wall time. In this case, reranking is the largest contributor by far (`368.640s`), followed by answer generation (`122.371s`), while retrieval itself is fast (`0.715s`). That means optimization should prioritize skipping or simplifying reranking first.

**Q: Why can reranking still be extremely slow even with the same model as classifier?**

**A:** Reranking is a separate LLM call with a much larger prompt (multiple candidate chunks), so it can be much slower than classifier routing even if both use the same model. Prompt size and token generation dominate latency more than model identity alone.

**Q: What is the practical tuning order from this log?**

**A:** First, force rerank skip (`RAG_FACT_FAST_MODE=true`, `RAG_RERANK_GAP_THRESHOLD=0`) and re-measure. Second, reduce answer latency by using a lighter answer model or tighter response length constraints. Third, keep warm-up behavior in mind because first-call model load can inflate routing time.

**Q: How can we speed up reranking without removing it?**

**A:** Keep reranking, but reduce reranker workload. The biggest levers are: fewer candidates into rerank, shorter query-focused snippets per candidate, and a tighter rerank output format (top-N selection instead of full long reasoning). If quality still holds, this preserves rerank benefits with much lower latency.

**Q: What is the recommended experiment order for rerank-speed improvements?**

**A:** 1) Reduce rerank candidate count (for example 12 -> 6) using similarity pre-filter. 2) Shorten per-candidate text (for example 900 chars -> 250-400 query-focused chars). 3) Change rerank objective to return only top IDs (`top_k`) with strict token limits. 4) If needed, move to a dedicated non-generative reranker model (cross-encoder style) for much faster scoring.

**Q: What changed after reducing fact lookup candidates to 8?**

**A:** Total latency dropped from ~520s to ~323s, and rerank dropped from ~369s to ~165s. This confirms rerank cost scales strongly with candidate set size. Retrieval remains fast (<1s), so the primary bottlenecks are still rerank and answer generation.

**Q: What does this imply for next optimization moves (without removing rerank)?**

**A:** Keep reducing reranker token load: fewer candidate passages, shorter per-candidate snippets, and stricter reranker output/token limits. Routing latency (~33s) is also high and likely includes model warm-up overhead, so startup warm-up and lightweight routing fallback should be evaluated next.
