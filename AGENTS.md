# Repository Instructions

## Daily Knowledge Base Rule

Create or update a daily knowledge base file under `docs/daily/` only when the conversation includes conceptual questions or substantive teaching-style discussion that would be useful to preserve.

Use this workflow:

1. Use the file naming format `YYYY-MM-DD_Knowledge_Base.md`.
2. Capture the important discussion from the current chat session in **Q&A style**.
3. Focus on:
   - concepts discussed
   - explanations given
   - design or implementation decisions
   - practical notes relevant to the repo
   - open questions or next steps
4. Keep the content concise, clear, and useful for later review.
5. If the file for that date already exists, append new Q&A entries instead of replacing existing notes.
6. Mention the created or updated knowledge base file in the final response when the workflow is triggered.


## Trigger Guidance

Treat the knowledge base update as triggered when the conversation includes things like:
- "what is X?"
- "explain X"
- "when should I use X?"
- "what is the difference between X and Y?"
- step-by-step conceptual clarification
- implementation reasoning that teaches how or why something works

Do not trigger the knowledge base update when:
- the user asks for code changes or implementation work
- the user asks to modify files, fix bugs, or add features
- the user asks to push code to git
- the user asks to explain existing repo code or walk through a specific file
- the discussion is mainly about code execution rather than conceptual understanding


## Scope Notes

- Prefer repo-specific discussion over generic theory, unless the theory was part of the session and helpful to preserve.
- If there was little or no substantive discussion before the trigger event, create a short note with the decisions taken during the implementation itself.
- Do not create a knowledge base entry for trivial one-line exchanges that do not add lasting value.
