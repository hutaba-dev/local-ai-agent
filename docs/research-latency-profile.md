# Deep Research Latency Profile

## Scope

This profile instruments the existing Deep Research implementation without
changing its prompts, model, tool count, research round count, vLLM settings,
or hardware configuration. It separates user-visible latency from the final
model call's decode throughput.

The current code has one bounded research tool round followed by three Research
v3 synthesis calls. It does not currently implement a follow-up/gap-analysis
research loop, so the measured round count is `1 / 1`.

## Primary Benchmark

Question:

```text
안호선교수에 대해서 찾아보고, 연구자로서의 역량을 평가해줘
```

Measured on 2026-08-21 against the production local service:

| Metric | Measured value |
| --- | ---: |
| Total user-visible latency | 254.341 s |
| Final visible output tokens | 2,042 |
| End-to-end rate | 8.03 tok/s |
| LLM calls | 4 |
| Whole-request LLM input tokens | 22,255 |
| Whole-request LLM output tokens | 6,897 |
| Total LLM latency | 241.293 s |
| Tool/network wall time | 13.046 s |
| Other measured overhead | approximately 0.002 s |

`8.03 tok/s` is the end-to-end rate: final visible output tokens divided by the
entire request wall-clock. It is not the model decode rate.

## LLM Call Breakdown

The runtime records streaming timestamps for each vLLM request. TTFT is the
time from request start to the first emitted content token; generation time is
the interval from that token to the end of the streamed response.

| # | Purpose | Input | Output | TTFT | Generation | Total LLM latency | Decode rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `research_mode_decision` | 217 | 111 | 0.105 s | 3.713 s | 3.817 s | 29.90 tok/s |
| 2 | `analyst_synthesis` | 4,597 | 3,544 | 0.735 s | 121.741 s | 122.475 s | 29.11 tok/s |
| 3 | `critic` | 8,111 | 1,200 | 1.466 s | 41.310 s | 42.777 s | 29.05 tok/s |
| 4 | `final_revision` | 9,330 | 2,042 | 1.702 s | 70.522 s | 72.224 s | 28.96 tok/s |

The final synthesis call, which is the source of the final answer usage shown
in the UI, had:

```text
Input tokens:  9,330
Output tokens: 2,042
TTFT:          1.702 s
Decode:        70.522 s
Decode speed:  28.96 tok/s
```

## Stage Breakdown

| Stage | Wall-clock time | Notes |
| --- | ---: | --- |
| Research mode decision | 3.818 s | One Qwen call; selected `DEEP_RESEARCH` |
| Research Round 1 tools | 13.046 s | Sequential tool execution |
| Analyst synthesis | 122.475 s | Qwen call #2 |
| Critic | 42.777 s | Qwen call #3 |
| Final revision | 72.224 s | Qwen call #4 |

Tool work is sequential in the current implementation. Therefore the Round 1
wall-clock is approximately the sum of its tool durations; it is not a
parallel-work aggregate.

## Round 1 Tool Breakdown

| Tool | Wall-clock time | Result |
| --- | ---: | --- |
| `web_search` | 4.667 s | Success |
| `web_sources` | 2.432 s | Success; five URL fetches, sequential |
| `academic_papers` | 0.407 s | Failed: no papers returned after request failures |
| `semantic_scholar` | 5.541 s | Success after rate-limit retries |

### Web Page Fetches

The high-level synchronous `httpx` client used by the current fetcher does not
expose a standalone TCP-connect timestamp. The profiler records `connect=N/A`
rather than inventing a number, alongside actual total fetch time, result, byte
count, and extracted text length.

| URL | Result | Total fetch | Bytes | Extracted text |
| --- | --- | ---: | ---: | ---: |
| `hibrain.net/.../303294` | Success | 0.559 s | 114,169 | 620 chars |
| `s-space.snu.ac.kr/.../000000182351.pdf` | Success | 0.101 s | 1,017 | 10 chars |
| `trend-m.com/.../57250549` | Success | 1.300 s | 737,046 | 35 chars |
| `ko.wikipedia.org/...` | Success | 0.337 s | 111,923 | 157 chars |
| `m.blog.naver.com/...` | Success | 0.135 s | 60,052 | 661 chars |

### Failure and Retry Evidence

`academic_papers` made four OpenAlex requests. Every request returned
`HTTP 429`; the tool did not retry, so it consumed 0.407 seconds in total.

`semantic_scholar` succeeded but each of its three API operations first received
`HTTP 429` and then succeeded on attempt 2:

| Operation | Attempt 1 | Attempt 2 |
| --- | ---: | ---: |
| Author search | 0.454 s, `HTTP 429` | 0.488 s, success |
| Author metadata | 0.218 s, `HTTP 429` | 0.463 s, success |
| Author papers | 0.433 s, `HTTP 429` | 0.483 s, success |

The request durations above sum to 2.539 seconds. The tool took 5.541 seconds,
so the remaining approximately 3.002 seconds is the configured retry backoff.
This establishes that retries added latency in this benchmark.

## UI Metrics

The Agent Activity panel now exposes, without prompt text or private
chain-of-thought:

- End-to-end rate: final output tokens divided by full request time.
- Final synthesis input/output tokens, TTFT, decode duration, and decode rate.
- Whole-request LLM input/output token totals and call count.
- Per-call purpose and total LLM latency.
- Actual timed stages, research rounds, tool wall-clock, per-request retry
  category, and sequential URL fetch measurements.

## Largest Measured Latency Sources

1. `analyst_synthesis`: 122.475 s.
2. `final_revision`: 72.224 s.
3. `critic`: 42.777 s.
4. `semantic_scholar`: 5.541 s, including approximately 3.002 s retry backoff.
5. `web_search`: 4.667 s.

The evidence supports a follow-up optimization discussion focused first on the
three sequential synthesis calls, not on a presumed low model decode rate. No
optimization has been applied in this change.