# Knowledge Base for 2026-03-26

## Q1. What is MCP in GenAI?

**Answer:**  
MCP usually stands for **Model Context Protocol**.

It is a standard way for AI assistants and tools to connect to external resources in a structured way.

You can think of it as a common interface that lets an LLM-powered assistant talk to things like:
- files
- databases
- APIs
- design tools
- documentation systems
- other developer tools

Instead of every tool integration being custom and different, MCP provides a more consistent protocol for exposing tools and context to the model.


## Q2. Why is MCP useful?

**Answer:**  
MCP is useful because it makes tool access more standardized and reusable.

Benefits:
- easier integration between models and tools
- more consistent way to expose context
- less one-off glue code
- better portability across assistants and environments


## Q3. How should I think about MCP simply?

**Answer:**  
A simple mental model is:

- the **LLM** is the reasoning engine
- **MCP** is the bridge or protocol used to connect that reasoning engine to external tools and context

So MCP is not the model itself. It is the mechanism that helps the model work with outside systems in a structured way.


## Q4. How can Streamlit be added to this RAG project?

**Answer:**  
The cleanest approach is to keep the existing RAG logic in `src/rag_pipeline.py` and add Streamlit as a separate UI layer.

That means:
- `rag_pipeline.py` stays the backend logic
- Streamlit becomes the frontend entry point
- the UI simply calls `ask_question(...)` and displays the result

This is useful because it avoids rewriting the retrieval or routing logic and keeps the architecture simple.


## Q5. What would the Streamlit UI need to show?

**Answer:**  
A practical first version would include:
- a text input for the user question
- a submit button
- the answer text
- route metadata such as query type, route source, confidence, and reason
- citations and sources

Optional additions later:
- chat history
- upload PDFs
- rebuild index button
- route/debug panel


## Q6. What is reranking in RAG?

**Answer:**  
Reranking is a second-stage selection step used after initial retrieval.

The usual pattern is:
1. retrieve a larger set of candidate chunks
2. score or reorder them again using a stronger method
3. keep the best final chunks for the prompt

Its purpose is to improve the quality of the final context that goes to the LLM.


## Q7. How should I think about reranking simply?

**Answer:**  
A simple mental model is:

- retrieval finds likely relevant chunks quickly
- reranking reviews those candidates more carefully and puts the best ones on top

So reranking is not the same as retrieval. It is a refinement step after retrieval.


## Q8. If MMR is already doing a good job for summary, what value would reranking add?

**Answer:**  
If MMR is already giving good summary coverage, reranking would add value mainly by improving the final quality of which chunks are kept, not by replacing MMR.

In this situation:
- MMR helps reduce redundancy and improve diversity
- reranking could help choose the most useful chunks among those candidates more precisely

So the value of reranking would usually be:
- better relevance ordering
- better final chunk quality
- cleaner context for difficult or ambiguous summary questions

If the current summary output already looks strong, reranking is more of an optimization step than an immediate necessity.


## Q9. Can reranking improve fact lookup when used with similarity search?

**Answer:**  
Yes, it can.

For fact lookup, a common pattern is:
1. use similarity search to retrieve a candidate set
2. apply reranking to reorder those candidates
3. keep the best few chunks for the final prompt

This can help because fact lookup usually depends on choosing the single most precise chunk, and reranking can improve that final selection.

The value is usually:
- better precision
- better ordering of top chunks
- fewer slightly relevant but less exact chunks in the final context

So reranking is often a stronger fit for fact lookup than for summary, while MMR is often a stronger fit for summary than for fact lookup.


## Q10. How can reranking be implemented for fact lookup in this repo?

**Answer:**  
The cleanest design is:

1. keep similarity search as the first retrieval step
2. fetch a larger candidate set
3. rerank those candidates with a stronger scoring method
4. keep the top final chunks for prompting

In this repo, that would fit naturally into the fact lookup flow:
- similarity search retrieves candidates
- reranking refines their order
- the top reranked chunks are passed to the fact-answer prompt

This keeps the current architecture mostly intact while improving final chunk precision for factual questions.
