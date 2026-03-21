# RAG System Challenges & Solutions

This document summarizes the key challenges encountered while building a local Retrieval-Augmented Generation (RAG) system and the solutions implemented to improve reliability, usability, and system design.

---

## 1. Hallucination / Unsafe Answers

### Problem
The model generated answers even when information was not present in the document.

Example:
> Who is Nemo?

### Root Cause
- LLM fallback to pre-trained knowledge
- No strict grounding enforcement

### Solution
- Enforced **context-only prompting**
- Added **retrieval score filtering**
- Introduced fallback response:
  > "I could not find relevant information..."

### Insight
> RAG systems must prefer **no answer over incorrect answer**

---

## 2. Lack of Traceability (Black-box Responses)

### Problem
Answers had no clear linkage to source documents.

### Root Cause
- No citation system
- Retrieval layer not exposed to user

### Solution
- Introduced **citation mapping ([1], [2], …)**
- Displayed **source file + page**
- Updated prompt to enforce citation usage

### Insight
> Trust in RAG systems comes from **traceability, not just accuracy**

---

## 3. Citation Reliability Issues

### Problem
- Inconsistent citation formatting
- Strict validation caused answer rejection

### Root Cause
- Prompt instructions are not guaranteed enforcement
- Smaller local models (e.g., 3B) have weaker formatting consistency

### Solution
- Avoided strict blocking validation
- Introduced **soft validation approach**
- Added grounding status concept:
  - Grounded
  - Partially Verified

### Insight
> Validation should not degrade UX in early-stage systems

---

## 4. Over-Strict Grounding Breaks UX

### Problem
System failed to respond to simple conversational inputs.

Example:
> hi / hello

### Root Cause
- All queries treated as document-based queries
- No intent differentiation

### Solution
- Introduced **Query Routing**
- Classified queries into:
  - Conversational
  - Local (fact-based)
  - Global (summary)
  - Out-of-scope

### Insight
> Not all queries should go through the RAG pipeline

---

## 5. Global Queries Failure (Major Issue)

### Problem
Summary-type queries failed or returned no answer.

Example:
> Give a 2-line summary of the document

### Root Cause
- RAG retrieves top-k chunks (local retrieval)
- Summaries require global understanding
- Citation validation fails for aggregated answers

### Solution
- Identified **query type mismatch**
- Introduced:
  - Query classification as "global"
  - Increased retrieval depth (higher k)
  - Relaxed citation constraints
  - Planned future: Map-Reduce summarization

### Insight
> RAG is optimized for **local retrieval, not global reasoning**

---

## 6. Retrieval Quality Limitations

### Problem
- Retrieved chunks were repetitive
- Limited coverage across document sections

### Root Cause
- Pure similarity search
- No diversity mechanism

### Solution (Planned)
- Introduce **MMR (Max Marginal Relevance)**

### Insight
> Answer quality depends more on **retrieval quality than LLM capability**

---

## 7. System Rigidity (Single Pipeline Limitation)

### Problem
One pipeline used for all query types resulted in inconsistent behavior.

### Root Cause
- Static pipeline design
- No adaptability

### Solution
- Introduced **Query Routing Layer**
