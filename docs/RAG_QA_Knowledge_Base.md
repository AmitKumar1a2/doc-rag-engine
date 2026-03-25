# RAG Knowledge Base (Q&A)

## Q1. What is RAG?

**Answer:**  
RAG stands for **Retrieval-Augmented Generation**.

It is a pattern where:
1. We **retrieve** relevant chunks from a knowledge source.
2. We give those chunks to an LLM as context.
3. The LLM **generates** an answer using that retrieved context.

So, RAG is not just retrieval. It is the full pipeline of:
- retrieval
- context building
- answer generation


## Q2. What is the retrieval step inside RAG?

**Answer:**  
The retrieval step is the part where the system selects the most useful document chunks for the user query.

In a RAG pipeline, retrieval happens before generation.

Example flow:
1. User asks a question.
2. System finds relevant chunks.
3. Those chunks are passed to the LLM.
4. The LLM answers using the chunks.


## Q3. Is similarity search a type of RAG?

**Answer:**  
No.

Similarity search is **not** a type of RAG. It is one **retrieval method** used inside a RAG system.

So the relationship is:
- **RAG** = the full system
- **Similarity search** = one way to retrieve chunks inside that system


## Q4. What is similarity search?

**Answer:**  
Similarity search retrieves chunks that are most similar to the user query in embedding space.

In simple terms, it asks:

**"Which chunks are closest to the question?"**

This works very well when the user is asking a focused question and we want the most directly relevant passages.


## Q5. When is similarity search most useful?

**Answer:**  
Similarity search is usually best for **fact lookup** style questions.

Examples:
- "What chunk size is used?"
- "Which embedding model is configured?"
- "What does the document say about X?"

Why it works well:
- it prioritizes the most relevant chunks
- it is strong for narrow, specific questions
- it is simple and efficient


## Q6. What is MMR?

**Answer:**  
MMR stands for **Maximal Marginal Relevance**.

It is another retrieval strategy used inside RAG.

MMR tries to retrieve chunks that are:
- relevant to the query
- different from each other

In simple terms, MMR asks:

**"Which chunks are relevant, but not too repetitive?"**


## Q7. Is MMR an alternative to RAG?

**Answer:**  
No.

MMR is not an alternative to RAG. It is a retrieval strategy **within** a RAG pipeline.

So this is the correct comparison:
- not **MMR vs RAG**
- but **MMR vs similarity search inside RAG**


## Q8. When is MMR most useful?

**Answer:**  
MMR is especially useful for **summary** or **broad** questions.

Examples:
- "Summarize the document."
- "What are the main challenges discussed in the paper?"
- "Give me the big picture."

Why it helps:
- summary questions need coverage across multiple parts of the document
- nearby chunks are often very similar to each other
- MMR reduces redundancy and improves diversity


## Q9. Why can similarity search be weak for summary questions?

**Answer:**  
Plain similarity search may return several chunks that are all highly relevant but very similar to one another.

That means the final context can become repetitive.

For summary questions, this is not ideal because:
- the model sees less variety
- important points from other sections may be missed
- the summary can become narrow or repetitive


## Q10. Why can similarity search be better for fact lookup?

**Answer:**  
Fact lookup usually needs the **single most relevant passage**, not a diverse set of passages.

For narrow questions:
- diversity is less important
- precision is more important

That is why similarity search is often the better default for fact lookup.


## Q11. Are MMR and similarity search two types of RAG?

**Answer:**  
No.

They are better described as **two retrieval strategies used within RAG**.

So:
- **RAG** = retrieve + generate
- **Similarity search** = one way to retrieve chunks
- **MMR** = another way to retrieve chunks


## Q12. Can similarity search and MMR be used together?

**Answer:**  
Yes, very often.

A common pattern is:
1. Use similarity search to get a larger candidate set.
2. Use MMR to pick a smaller final set from those candidates.

This gives both:
- relevance
- diversity


## Q13. What does "used together" look like in practice?

**Answer:**  
Suppose we want **5 final chunks** for the prompt.

We can do:
1. Retrieve **20 candidate chunks** using similarity search.
2. Apply MMR on those 20 chunks.
3. Keep the best **5 diversified chunks**.

So:
- `fetch_k = 20`
- `k = 5`

This is a very common setup.


## Q14. What is `fetch_k` in MMR?

**Answer:**  
`fetch_k` is the size of the **initial candidate pool** retrieved before MMR chooses the final results.

In most implementations, it works like this:
1. Run similarity-style retrieval to get `fetch_k` chunks.
2. Apply MMR to that pool.
3. Return the final `k` chunks.

So:
- `fetch_k` = how many candidates to gather first
- `k` = how many final chunks to return


## Q15. Does `fetch_k` behave like similarity search?

**Answer:**  
Yes, conceptually it does.

You can think of `fetch_k` as:

**"How many similarity-retrieved chunks should I collect before MMR starts selecting?"**

So if:
- `fetch_k = 20`
- `k = 5`

Then the system:
1. gets 20 relevant candidates
2. uses MMR to reduce them to 5


## Q16. What is a good rule of thumb for choosing between similarity search and MMR?

**Answer:**  
Use this simple rule:

- Use **similarity search** for narrow, specific, fact-based questions.
- Use **MMR** for broad, summary-style, or multi-aspect questions.

Examples:
- Fact lookup: similarity search
- Summary: MMR
- Broad analytical question: often MMR


## Q17. What is `lambda_mult` in MMR?

**Answer:**  
`lambda_mult` controls the tradeoff between:
- relevance to the query
- diversity among the selected chunks

A simple way to think about it:
- higher `lambda_mult` means MMR behaves more like similarity search
- lower `lambda_mult` means MMR favors more diversity

Common intuition:
- `lambda_mult = 1.0` -> mostly relevance
- `lambda_mult = 0.5` -> balanced relevance and diversity
- `lambda_mult = 0.0` -> mostly diversity

For summary-style questions, a middle value like `0.5` is often a good starting point.


## Q18. What is the cleanest way to think about all of this?

**Answer:**  
The cleanest mental model is:

- **RAG** is the full pipeline.
- **Retrieval** is one step inside that pipeline.
- **Similarity search** and **MMR** are two different ways to perform that retrieval step.

Short version:
- **RAG** = retrieve + generate
- **Similarity search** = get the most similar chunks
- **MMR** = get relevant chunks with more diversity
