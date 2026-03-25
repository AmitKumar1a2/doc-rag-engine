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
