# Analytical Research Pipeline

## Objective

The Research Agent produces source-grounded expert analysis rather than repeating source sentences. Its epistemic contract is:

- Facts are sourced.
- Inferences are reasoned.
- Forecasts are conditional.
- Unknowns are acknowledged.

A missing article that states the final conclusion verbatim does not prevent analysis when the material premises and causal mechanism are supported.

## Claim Taxonomy

### FACT

A statement directly supported by fetched source text or structured evidence. Numbers, dates, events, supplier relationships, and current market observations must be treated as facts and cited.

### INFERENCE

An analytical conclusion derived from identified facts and a defensible structural relationship. An inference records premises, causal steps, confidence, assumptions, counterarguments, and evidence IDs. It must not be labeled `NOT VERIFIED` merely because no source states the conclusion verbatim.

### FORECAST

A conditional future outcome tied to explicit triggers and a scenario. Forecasts are not factual promises. Confidence falls when important assumptions or intermediate evidence are missing.

### UNKNOWN

A factual point or material premise for which the evidence is insufficient. `UNKNOWN` replaces the prior tendency to label all non-direct conclusions `NOT VERIFIED`.

`NOT VERIFIED` is reserved for an unconfirmed factual claim that would otherwise be presented as fact.

## Evidence Roles

Fetched web evidence is normalized into a compact package with stable IDs such as `S1` and one role:

- `DIRECT`: official investor relations, SEC, newsroom, or equivalent primary evidence
- `STRUCTURAL`: value-chain, supplier/customer, capacity, pricing, utilization, or margin relationships
- `SUPPORTING`: independent context that supports a premise
- `CONTRADICTORY`: evidence that weakens a directional thesis or exposes a risk

The selector preserves at least one high-relevance item from each available role before filling remaining capacity by relevance. This prevents a long run of official result pages from crowding out structural or contradictory evidence.

## Query Planning

Qwen chooses each next action from the user goal, evidence so far, live tool status, unresolved uncertainty, and remaining budget. Intent labels are non-binding metadata and keywords do not select source classes. For company-to-sector questions, the model may search for evidence supporting causal premises, not only an article containing the desired conclusion. Relevant considerations include:

1. The focal company's current performance drivers and official release
2. The supply-chain or value-chain relationship to the target sector
3. Current sector demand, supply, pricing, capacity, and inventory fundamentals
4. Volume, price, mix, margin, earnings, and valuation transmission
5. Exposed beneficiaries and losers, including company-specific differences
6. Market expectations, counterarguments, and already-priced-in risk

Generated follow-up queries are not constrained by an original source-intent category. Before selecting a specialized source, the model asks whether it can materially reduce uncertainty about the actual question.

## Evidence Gap Policy

The gap evaluator distinguishes missing factual premises from missing direct wording for an analytical conclusion.

- Missing material premise: search again; if unresolved, classify it as `UNKNOWN` and lower confidence.
- Supported premises and mechanism: allow `INFERENCE`, even if no source states the final conclusion.
- Identity ambiguity or missing critical current fact: keep `ready_to_answer=false` and perform bounded follow-up.
- Missing non-critical detail: synthesize with an assumption, limitation, or lower confidence.

Evidence gaps are evaluated inside each next-action decision rather than by a mandatory standalone stage. The planner may search, fetch, compare, calculate, analyze, or finalize.

## Analysis Contract

The compact Evidence Package contains a machine-readable `analysis_contract`. It defines:

- `claim_taxonomy`
- `evidence_roles`
- causal stages from facts through first- and second-order effects
- an inference object schema
- a bull/base/bear scenario schema for market or company analysis
- optional analytical sub-questions when causal analysis is material

The internal inference object includes:

```json
{
  "claim": "string",
  "type": "INFERENCE",
  "premises": ["string"],
  "causal_chain": ["string"],
  "confidence": "HIGH|MEDIUM|LOW",
  "assumptions": ["string"],
  "counterarguments": ["string"],
  "evidence_ids": ["S1"]
}
```

This is a structured analytical artifact supplied to later stages, not hidden chain-of-thought.

## Dynamic Synthesis

Evidence normalization always runs. Later calls are selected from Qwen's complexity assessment:

1. Evidence normalization removes duplicate sources, bounds text, adds IDs and roles, and preserves role diversity.
2. Simple and moderate questions can proceed directly to final synthesis.
3. Complex causal, market, conflict, identity, or scenario questions can invoke the Causal Analyst and Research Critic before final revision.

The critic improves inference quality; it does not delete valid analysis solely because the conclusion lacks a direct quotation. If synthesis requests more research, the executor rejects finalization and returns control to the next-action loop.

## Response Shape

For analytical questions, the final answer may use:

- Executive View
- Verified Facts
- What It Means
- Causal Chain
- Sector / Company Impact
- Bull / Base / Bear Scenarios
- Key Risks / Counterarguments
- What to Watch Next
- Confidence / Unknowns

The model adapts this structure to the question rather than forcing every section into every response.

## Regression

The canonical production case is defined in `agents/research/evaluation-tasks.md`. Automated tests verify:

- role assignment and stable evidence IDs
- role diversity in compact evidence selection
- inference-aware evidence gap behavior
- analytical query decomposition instructions
- analysis contract delivery to the Analyst
- critic treatment of valid inference
- final `FACT / INFERENCE / FORECAST / UNKNOWN` policy
- LLM-selected next actions and bounded executor guardrails
- rejection of premature finalization and progress-only answers
- dynamic direct versus Analyst/Critic synthesis

## Production Benchmark: NVIDIA to Memory Sector

Question:

> NVIDIA 실적이 발표됐어. 발표 내용을 분석해주고, 앞으로 메모리 주식 섹터에 미칠 영향에 대한 전문적인 분석을 부탁해.

### Before

The prior production behavior retrieved some earnings facts but treated the absence of a source stating the final sector conclusion as a verification failure. It stopped most impact analysis with `NOT VERIFIED` and did not provide a useful causal chain, second-order effects, scenarios, or counterarguments.

### After

The final production run used separate search/gap, Analyst, Critic, and Final Synthesis calls. It returned a complete 9,393-character report without reaching the output limit.

| Criterion | Before | After |
| --- | --- | --- |
| Factual grounding | Partial earnings facts | Dedicated Verified Facts with citations and arithmetic identified as calculated |
| Analytical depth | Analysis stopped | Company signal connected to sector transmission |
| Causal reasoning | Missing | Explicit premises and causal chain |
| Second-order reasoning | Missing | Pricing, margin, earnings, and valuation transmission |
| Counterarguments | Minimal or missing | Included for directional inferences |
| Scenario quality | Missing | Bull/base/bear with triggers, mechanisms, risks, and confidence |
| Actionable insight | Low | What-to-watch indicators and material unknowns |
| Epistemic labels | `NOT VERIFIED` applied to analysis | FACT / INFERENCE / FORECAST / UNKNOWN; `NOT VERIFIED` count 0 |
| Completion | Conservative refusal | Complete response; no truncation |

An intermediate benchmark revealed a new failure mode: analytical freedom produced unsupported precise ASP changes, margins, valuation multiples, and named-company relationships. The Critic and Final Synthesis prompts were then strengthened to audit every number and company relationship against the Evidence Package, keep scenarios qualitative when no sourced number exists, and omit unsupported company comparisons. This calibration is part of the final pipeline, not an accepted benchmark result.

The final structural checks passed for factual sections, causal reasoning, second-order effects, counterarguments, bull/base/bear scenarios, confidence, unknowns, zero `NOT VERIFIED` misuse, and response completion. Human source review remains necessary for investment decisions; the benchmark validates pipeline behavior, not investment correctness.
