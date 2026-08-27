import os
import unittest
from unittest.mock import patch

import httpx

from runtime.search_providers import (
    BraveProvider,
    NormalizedSearchResult,
    ProviderResponse,
    ProviderStatus,
    SearchProvider,
    SearchRequest,
    SearchRouter,
    SerperProvider,
    SearXNGProvider,
    canonicalize_url,
    reset_search_state,
)


def search_result(provider: str, rank: int, url: str) -> NormalizedSearchResult:
    return NormalizedSearchResult(
        "NVIDIA official earnings market outlook", url,
        "NVIDIA earnings revenue EPS guidance consensus official current details",
        "2026-08-27", url.split("/")[2], provider, "news", rank,
        providers_seen=(provider,),
    )


class FakeProvider(SearchProvider):
    def __init__(self, name: str, results=(), status=ProviderStatus.AVAILABLE, configured=True, error=None):
        self.name = name
        self.results = tuple(results)
        self.status = status
        self.is_configured = configured
        self.error = error
        self.calls = 0

    def configured(self) -> bool:
        return self.is_configured

    def search(self, request: SearchRequest) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            self.name, self.status, self.results, 5, False,
            429 if self.status == ProviderStatus.RATE_LIMITED else None, self.error,
        )


class JsonResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://provider.example")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("provider error", request=request, response=response)

    def json(self):
        return self.payload


class SearchProviderTests(unittest.TestCase):
    def setUp(self):
        reset_search_state()

    @staticmethod
    def sufficient(provider, title="NVIDIA official earnings market outlook"):
        urls = (
            "https://nvidianews.nvidia.com/news/results",
            "https://www.reuters.com/technology/nvidia-results",
            "https://www.cnbc.com/nvidia-results",
            "https://www.nasdaq.com/market-activity/nvda",
            "https://finance.yahoo.com/quote/NVDA",
        )
        return tuple(
            NormalizedSearchResult(
                title, url, title, "2026-08-27", url.split("/")[2], provider, "news", rank,
                providers_seen=(provider,),
            )
            for rank, url in enumerate(urls, 1)
        )

    def test_a_sufficient_searxng_uses_no_paid_provider(self):
        searxng = FakeProvider("searxng", self.sufficient("searxng"))
        serper = FakeProvider("serper", self.sufficient("serper"))
        brave = FakeProvider("brave", self.sufficient("brave"))
        router = SearchRouter({"searxng": searxng, "serper": serper, "brave": brave})
        context = {"intents": ["CURRENT_NEWS", "MARKET_FINANCE"], "freshness": "VERY_HIGH", "requires_primary": True}

        batch = router.search(SearchRequest("today NVIDIA earnings market outlook", "news", "VERY_HIGH"), context)

        self.assertEqual((searxng.calls, serper.calls, brave.calls), (1, 0, 0))
        self.assertEqual(batch.metrics["estimated_paid_requests"], 0)

    def test_primary_rich_searxng_stops_despite_noisy_results(self):
        results = (
            search_result("searxng", 1, "https://investor.nvidia.com/results"),
            search_result("searxng", 2, "https://investor.nvidia.com/earnings"),
            search_result("searxng", 3, "https://example.com/unrelated"),
            search_result("searxng", 4, "https://other.example.org/noise"),
            search_result("searxng", 5, "https://third.example.net/noise"),
        )
        searxng = FakeProvider("searxng", results)
        serper = FakeProvider("serper", self.sufficient("serper"))
        brave = FakeProvider("brave", self.sufficient("brave"))
        router = SearchRouter({"searxng": searxng, "serper": serper, "brave": brave})

        batch = router.search(SearchRequest("NVIDIA investor earnings"), {"requires_primary": True})

        self.assertEqual((searxng.calls, serper.calls, brave.calls), (1, 0, 0))
        self.assertEqual(batch.metrics["estimated_paid_requests"], 0)

    def test_b_current_news_falls_back_only_until_quality_is_sufficient(self):
        searxng = FakeProvider("searxng", (search_result("searxng", 1, "https://example.com/openai"),))
        serper = FakeProvider("serper", self.sufficient("serper", "OpenAI latest news current update"))
        brave = FakeProvider("brave", self.sufficient("brave"))
        router = SearchRouter({"searxng": searxng, "serper": serper, "brave": brave})

        router.search(SearchRequest("OpenAI latest news", "news", "VERY_HIGH"), {"intents": ["CURRENT_NEWS"], "freshness": "VERY_HIGH"})

        self.assertEqual((searxng.calls, serper.calls, brave.calls), (1, 1, 0))

    def test_d_unavailable_searxng_falls_back_to_serper(self):
        searxng = FakeProvider("searxng", status=ProviderStatus.TEMPORARILY_UNAVAILABLE, error="timeout")
        serper = FakeProvider("serper", self.sufficient("serper", "technical product documentation official guide"))
        brave = FakeProvider("brave", self.sufficient("brave"))
        router = SearchRouter({"searxng": searxng, "serper": serper, "brave": brave})

        router.search(SearchRequest("technical product documentation"), {})

        self.assertEqual((searxng.calls, serper.calls, brave.calls), (1, 1, 0))

    def test_e_unavailable_searxng_and_unconfigured_serper_use_brave(self):
        searxng = FakeProvider("searxng", status=ProviderStatus.TEMPORARILY_UNAVAILABLE, error="http_503")
        serper = FakeProvider("serper", configured=False)
        brave = FakeProvider("brave", self.sufficient("brave"))
        router = SearchRouter({"searxng": searxng, "serper": serper, "brave": brave})

        batch = router.search(SearchRequest("company research"), {})

        self.assertEqual((searxng.calls, serper.calls, brave.calls), (1, 0, 1))
        self.assertEqual(batch.metrics["providers"]["serper"]["status"], "UNCONFIGURED")

    def test_f_serper_429_falls_back_to_brave(self):
        searxng = FakeProvider("searxng", (search_result("searxng", 1, "https://example.com/thin"),))
        serper = FakeProvider("serper", status=ProviderStatus.RATE_LIMITED, error="http_429")
        brave = FakeProvider("brave", self.sufficient("brave"))
        router = SearchRouter({"searxng": searxng, "serper": serper, "brave": brave})

        batch = router.search(SearchRequest("company research"), {})

        self.assertEqual(brave.calls, 1)
        self.assertEqual(batch.metrics["providers"]["serper"]["rate_limited_count"], 1)

    def test_g_repeated_query_uses_cache(self):
        response = JsonResponse({"results": [
            {"title": f"Result {index}", "url": f"https://domain{index}.example/item", "content": "cached evidence"}
            for index in range(5)
        ]})
        provider = SearXNGProvider("http://127.0.0.1:8088")
        with patch("runtime.search_providers.httpx.request", return_value=response) as call:
            first = provider.search(SearchRequest("repeat query"))
            second = provider.search(SearchRequest("repeat query"))

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(call.call_count, 1)

    def test_h_duplicate_url_keeps_cross_provider_provenance(self):
        first = search_result("searxng", 1, "https://Example.com/report/?utm_source=search")
        second = search_result("serper", 1, "https://www.example.com/report")
        searxng = FakeProvider("searxng", (first,))
        serper = FakeProvider("serper", (second, *self.sufficient("serper")))
        router = SearchRouter({"searxng": searxng, "serper": serper, "brave": FakeProvider("brave", configured=False)})

        batch = router.search(SearchRequest("NVIDIA earnings report"), {})

        matches = [item for item in batch.results if canonicalize_url(str(item["url"])) == "https://example.com/report"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["providers_seen"], ["searxng", "serper"])

    def test_adapters_normalize_provider_specific_responses(self):
        searx = JsonResponse({"results": [{"title": "SearX", "url": "https://example.com/a", "content": "Snippet", "engine": "duckduckgo", "score": 1.5}]})
        serper = JsonResponse({"organic": [{"title": "Serper", "link": "https://example.com/b", "snippet": "Snippet"}]})
        brave = JsonResponse({"web": {"results": [{"title": "Brave", "url": "https://example.com/c", "description": "Snippet"}]}})
        with patch.dict(os.environ, {"SERPER_API_KEY": "test", "BRAVE_SEARCH_API_KEY": "test"}), patch(
            "runtime.search_providers.httpx.request", return_value=searx
        ), patch("runtime.search_providers.httpx.post", return_value=serper), patch(
            "runtime.search_providers.httpx.get", return_value=brave
        ):
            responses = (
                SearXNGProvider("http://127.0.0.1:8088").search(SearchRequest("a")),
                SerperProvider().search(SearchRequest("b")),
                BraveProvider().search(SearchRequest("c")),
            )

        self.assertEqual([response.results[0].provider for response in responses], ["searxng", "serper", "brave"])
        self.assertEqual(responses[0].results[0].engine, "duckduckgo")


if __name__ == "__main__":
    unittest.main()