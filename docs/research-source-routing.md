# Deep Research Source Routing (Historical)

> Superseded by [LLM Research Orchestration](llm-research-orchestration.md).

The former runtime inferred current-news, market, academic, company, and technical intent with regular expressions. It then enabled source classes, expanded company-specific queries, and ranked results with fixed domain tiers. NVIDIA and TSMC domains were embedded directly in Python.

That architecture was removed because it duplicated the model planner, confused Research expertise with research depth, failed on unknown entities, and silently encoded topic and company policy in execution code.

The compatibility `ResearchSourcePlan` now returns non-semantic fallback metadata only. It does not select tools, providers, queries, or evidence classes. The canonical `ResearchPlan` is produced by Qwen and passed through each iterative decision. Python validates and executes those decisions under deterministic security, health, availability, cost, and budget controls.

Do not add semantic regexes, company-domain maps, topic allowlists, or fixed evidence bundles to the compatibility path. Extend the structured planner schema or tool catalog when the model needs another decision surface.