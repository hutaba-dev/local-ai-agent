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
