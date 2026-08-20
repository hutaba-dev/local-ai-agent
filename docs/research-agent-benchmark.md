# Research Agent Benchmark

## Purpose

Measure whether adding independent academic evidence sources improves a
researcher-evaluation response, rather than treating a higher tool count as a
success metric.

## Benchmark Prompt

> 인천대학교 기계공학과 안호선 교수의 연구 역량을 논문, 인용, 대표 연구와 최근 성과를 근거로 평가해줘

## Evaluation Criteria

1. The requested researcher and institution are retained in the answer.
2. The answer distinguishes observed evidence from interpretation and limits.
3. Representative-paper and citation claims are supported by an independent
   academic source or explicitly marked unavailable.
4. URLs or DOI records are adjacent to factual claims.
5. A provider outage does not prevent a response based on the remaining sources.

## Before v2.1

The existing Deep Research pipeline used Brave/Naver discovery, public HTML
source fetches, and OpenAlex work-search metadata. The original prompt is now
preserved as an entity anchor, but OpenAlex work search alone can still leave a
citation cross-check or representative-work evidence gap.

Observed baseline for the benchmark after entity-preserving retrieval:

| Measure | Result |
| --- | --- |
| Route | `DEEP_RESEARCH` |
| Successful evidence tools | `web_search`, `web_sources`, `academic_papers` |
| Entity retained | Yes |
| Answer length | 2,380 characters |
| URL markers | 5 |
| Independent citation cross-check | No |
| Legal OA lookup for representative DOI | No |

## v2.1 Changes

- Semantic Scholar adapters: author search, author detail, author papers, and
  paper detail.
- Semantic Scholar is selected only for an academic evidence gap: insufficient
  OpenAlex citation-bearing works or an explicit citation/h-index request.
- Unpaywall is selected only when public source text is sparse and a candidate
  representative work has a DOI.
- Unpaywall returns legal OA locations only; it does not fetch or bypass
  paywalled content.
- Both providers degrade gracefully: OpenAlex and web evidence remain usable
  when Semantic Scholar is rate-limited or Unpaywall is unconfigured.

## v2.1 Benchmark Run

The benchmark was run with the currently configured public providers. Results:

| Measure | Result |
| --- | --- |
| Route | `DEEP_RESEARCH` |
| Successful evidence tools | `web_search`, `web_sources`, `academic_papers` |
| Semantic Scholar | Unavailable without configured key/public capacity; graceful failure |
| Unpaywall | Skipped because the public-source evidence threshold was already met |
| Entity retained | Yes |
| Answer length | 2,920 characters |
| URL markers | 3 |
| Response available after provider failure | Yes |

## Assessment

This run **does not demonstrate a quality improvement from a higher source
count**. Semantic Scholar produced no usable independent evidence under the
current unauthenticated public capacity, and Unpaywall was correctly not called
because its evidence-gap condition was not met. The result proves graceful
degradation and conditional tool selection, not an academic-quality gain.

To measure a genuine improvement, configure `S2_API_KEY` and
`UNPAYWALL_EMAIL`, rerun this prompt, and compare:

- confirmed author identity and affiliation across OpenAlex and Semantic Scholar;
- representative work overlap and citation disagreement, if any;
- legal OA location availability for DOI-confirmed representative papers;
- evidence-backed claims and citations, not response length.

A source is counted as an improvement only if it contributes a claim that is
both relevant to the requested researcher and independently verifiable.

## v3 Synthesis Pipeline

Deep Research now converts successful tool observations into an `Evidence
Package` with identity, metrics, representative works, source records, and
explicit limitations. The same Qwen3.8-27B model then performs three bounded
passes: Analyst / Synthesizer, evidence-bound Critic, and Final Revision. The
Critic may identify only unsupported claims, overinterpretation, identity risk,
missing limitations, repetition, or weak evaluation; it may not introduce facts.

## v2 and v3 Comparison

The same benchmark prompt was run after Semantic Scholar author resolution was
available. Both runs used the same model and the same configured evidence
providers. The v2 comparator used one final model call with raw tool
observations; v3 used the Evidence Package, Analyst, Critic, and Final Revision
passes.

| Criterion | v2 Answer | v3 Answer |
| --- | --- | --- |
| Depth | Identified research areas and practical outputs, but largely listed findings by source. | Connected the Top 2% recognition, field direction, representative-work ambiguity, and evidence gaps into a conditional overall judgment. |
| Evidence use | Raw tool observations were cited directly. | Source text and structured academic records were separated into an Evidence Package before interpretation. |
| Interpretation | Explained individual sources, with limited cross-source weighing. | Distinguished a strong institutional recognition signal from weaker bibliographic identity evidence. |
| Nuance | Noted missing exact metrics. | Explicitly treated the mismatched `Sun Ho Ahn` academic record as a probable different researcher and limited quantitative claims. |
| Judgment quality | Positive overall assessment with a caveat. | Positive but conditional assessment: international recognition is supported, while paper-level and citation-level evaluation remains unconfirmed. |
| Hallucination | Prompt-constrained. | Critic was constrained to package evidence and checked unsupported facts, metric overinterpretation, identity error, and missing limitations. |
| Readability | About 4,100 Korean characters; concise source-by-source format. | 6,711 Korean characters with 18 URL markers; more analytical, but can be more verbose than needed. |
| Usefulness | Gives a practical starting view and follow-up sources. | Better explains what can be concluded now and which missing identifiers block a defensible bibliometric assessment. |

### v2 Answer Record

The v2 answer identified the researcher as an Incheon National University
mechanical-engineering professor working around heat transfer, critical heat
flux, condensation, graphene, thermal management, hydrogen storage, and
advanced mobility. It cited the university's Top 2% researcher notice,
ResearchGate, the AHN Lab page, and reports on graphene-speaker and portable
hydrogen-storage work. Its final assessment was positive, while stating that
paper count, citation totals, h-index, representative papers, and recent
publication volume could not be confirmed. It also correctly rejected the
Semantic Scholar `Sun Ho Ahn` record as likely unrelated because its papers were
in interventional radiology.

### v3 Answer Record

The v3 answer reached the same core factual conclusion but organized it as an
evidence-bound evaluation. It treated the university's 2025 career-long Top 2%
recognition in Advanced Mobility as a strong signal of accumulated impact, then
separated that signal from unverified paper-level metrics. It explicitly stated
that the available Semantic Scholar/OpenAlex author record did not establish
identity with the mechanical-engineering professor and therefore could not be
used for the requested paper-count, citation, or h-index judgment. The final
assessment was that the researcher appears to have substantial international
impact, but that the confidence of a detailed bibliometric ranking is limited
until an authoritative author identifier or publication list is obtained.

The v3 run returned `web_search`, `web_sources`, `academic_papers`, and
`semantic_scholar` successfully. It produced 6,711 Korean characters and 18
URL markers. This is a material improvement in evidence use, interpretation,
and identity-risk handling. Readability remains the limiting tradeoff: the
three-pass output should be monitored for over-length and repeated caveats.
