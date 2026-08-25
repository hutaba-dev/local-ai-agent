import os
import unittest
from unittest.mock import Mock, patch

import httpx

from runtime.academic_intelligence import (
    AcademicSourceResult,
    SourceStatus,
    _aggregate_intelligence,
    _orcid_provider,
    _semantic_scholar_provider,
    academic_intelligence,
    academic_source_status,
    scopus_get_abstract,
    scopus_get_author,
    scopus_get_author_documents,
    scopus_get_citation_overview,
    scopus_search_authors,
    scopus_search_documents,
    wos_get_document,
    wos_get_researcher,
    wos_search_documents,
    wos_search_researchers,
)


class AcademicIntelligenceTests(unittest.TestCase):
    def test_missing_credentials_are_explicit_degraded_states(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            status = academic_source_status()

        self.assertEqual(status["scopus"], "UNAVAILABLE")
        self.assertEqual(status["web_of_science"], "UNAVAILABLE")
        self.assertEqual(status["google_scholar"], "UNAVAILABLE")
        self.assertEqual(status["openalex"], "AVAILABLE_FULL")

    def test_scopus_adapters_use_official_endpoints_and_credentials(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        with patch.dict(os.environ, {"SCOPUS_API_KEY": "key", "SCOPUS_INST_TOKEN": "token"}), patch(
            "runtime.academic_intelligence.httpx.get", return_value=response
        ) as get:
            scopus_search_authors("AUTHLASTNAME(Hinton)")
            scopus_get_author("123")
            scopus_get_author_documents("123")
            scopus_search_documents("TITLE(deep learning)")
            scopus_get_abstract(scopus_id="456")
            scopus_get_citation_overview(("456", "789"), "2020-2025")

        urls = [call.args[0] for call in get.call_args_list]
        self.assertIn("https://api.elsevier.com/content/search/author", urls)
        self.assertIn("https://api.elsevier.com/content/author/author_id/123", urls)
        self.assertIn("https://api.elsevier.com/content/search/scopus", urls)
        self.assertIn("https://api.elsevier.com/content/abstract/scopus_id/456", urls)
        self.assertIn("https://api.elsevier.com/content/abstract/citations", urls)
        self.assertTrue(all(call.kwargs["headers"]["X-ELS-APIKey"] == "key" for call in get.call_args_list))
        self.assertTrue(all(call.kwargs["headers"]["X-ELS-Insttoken"] == "token" for call in get.call_args_list))

    def test_wos_adapters_use_starter_and_researcher_endpoints(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        with patch.dict(os.environ, {"WOS_API_KEY": "key"}), patch(
            "runtime.academic_intelligence.httpx.get", return_value=response
        ) as get:
            wos_search_researchers("Geoffrey Hinton")
            wos_get_researcher("A-1234-2000")
            wos_search_documents('AU=("Geoffrey Hinton")')
            wos_get_document("WOS:123")

        urls = [call.args[0] for call in get.call_args_list]
        self.assertIn("https://api.clarivate.com/apis/wos-researcher/researchers", urls)
        self.assertIn("https://api.clarivate.com/apis/wos-researcher/researchers/A-1234-2000", urls)
        self.assertIn("https://api.clarivate.com/apis/wos-starter/v1/documents", urls)
        self.assertIn("https://api.clarivate.com/apis/wos-starter/v1/documents/WOS:123", urls)
        self.assertTrue(all(call.kwargs["headers"]["X-ApiKey"] == "key" for call in get.call_args_list))

    def test_provider_http_states_distinguish_entitlement_and_rate_limit(self) -> None:
        from runtime.academic_intelligence import _provider_failure

        request = httpx.Request("GET", "https://example.test")
        forbidden = httpx.Response(403, request=request)
        limited = httpx.Response(429, request=request)

        self.assertEqual(
            _provider_failure("scopus", httpx.HTTPStatusError("forbidden", request=request, response=forbidden)).status,
            SourceStatus.NO_ENTITLEMENT,
        )
        self.assertEqual(
            _provider_failure("wos", httpx.HTTPStatusError("limited", request=request, response=limited)).status,
            SourceStatus.RATE_LIMITED,
        )

    def test_orcid_provider_resolves_public_identity(self) -> None:
        payload = {
            "expanded-result": [{
                "orcid-id": "0000-0001-2345-6789",
                "given-names": "Ada",
                "family-names": "Researcher",
                "institution-name": ["Example University"],
            }]
        }
        with patch("runtime.academic_intelligence._request_json", return_value=payload) as request:
            result = _orcid_provider("Ada Researcher professor")

        self.assertEqual(result.status, SourceStatus.AVAILABLE_FULL)
        self.assertEqual(result.identities[0]["identifiers"]["orcid"], "0000-0001-2345-6789")
        self.assertEqual(request.call_args.args[0], "https://pub.orcid.org/v3.0/expanded-search/")

    def test_semantic_scholar_fetches_papers_for_one_exact_identity(self) -> None:
        author_payload = {"data": [{
            "authorId": "S2-AUTHOR",
            "name": "Ada Researcher",
            "paperCount": 12,
            "citationCount": 345,
            "hIndex": 7,
            "affiliations": ["Example University"],
        }]}
        papers_payload = {"data": [{
            "paperId": "S2-PAPER",
            "title": "Verified Work",
            "year": 2024,
            "citationCount": 9,
            "externalIds": {"DOI": "10.1000/verified"},
            "venue": "Example Journal",
            "authors": [{"name": "Ada Researcher"}],
            "abstract": "Evidence.",
        }]}
        with patch(
            "runtime.academic_intelligence._request_json",
            side_effect=[author_payload, papers_payload],
        ) as request:
            result = _semantic_scholar_provider("Ada Researcher professor")

        self.assertEqual(result.status, SourceStatus.AVAILABLE_FULL)
        self.assertEqual(result.metrics["document_count"], 12)
        self.assertEqual(result.publications[0]["doi"], "10.1000/verified")
        self.assertEqual(
            request.call_args_list[1].args[0],
            "https://api.semanticscholar.org/graph/v1/author/S2-AUTHOR/papers",
        )

    def test_aggregator_deduplicates_publications_and_preserves_source_metrics(self) -> None:
        identity_scopus = {
            "name": "Ada Researcher", "source": "scopus",
            "identifiers": {"scopus_author_id": "1"}, "affiliations": ["Example University"],
        }
        identity_wos = {
            "name": "Ada Researcher", "source": "web_of_science",
            "identifiers": {"wos_researcher_id": "A-1"}, "affiliations": ["Example University"],
        }
        same_scopus = {
            "title": "A Shared Paper", "doi": "10.1000/shared", "year": 2024,
            "authors": ["Ada Researcher"], "citation_count": 20, "sources": ["scopus"],
            "source_ids": {"scopus_id": "S1"},
        }
        same_wos = {
            "title": "A shared paper", "doi": "https://doi.org/10.1000/shared", "year": 2024,
            "authors": ["Ada Researcher"], "citation_count": 18, "sources": ["web_of_science"],
            "source_ids": {"wos_uid": "W1"},
        }
        results = [
            AcademicSourceResult(
                "scopus", SourceStatus.AVAILABLE_FULL, (identity_scopus,), (same_scopus,),
                {"document_count": 118, "citation_count": 5000, "h_index": 35},
            ),
            AcademicSourceResult(
                "web_of_science", SourceStatus.AVAILABLE_FULL, (identity_wos,), (same_wos,),
                {"document_count": 112, "citation_count": 4300, "h_index": 32},
            ),
            AcademicSourceResult(
                "openalex", SourceStatus.AVAILABLE_FULL, publications=tuple(
                    {"title": f"OpenAlex Paper {index}", "year": 2020, "authors": ["Ada Researcher"], "sources": ["openalex"]}
                    for index in range(18)
                ), metrics={"document_count": 18, "citation_count": 200, "h_index": 8},
            ),
        ]

        intelligence = _aggregate_intelligence("Ada Researcher professor", results)

        shared = next(paper for paper in intelligence["merged_verified_corpus"] if paper.get("doi"))
        self.assertEqual(shared["sources"], ["scopus", "web_of_science"])
        self.assertEqual(shared["authorship_confidence"], "HIGH")
        self.assertEqual(intelligence["coverage"]["scopus"]["h_index"], 35)
        self.assertEqual(intelligence["coverage"]["web_of_science"]["h_index"], 32)
        self.assertTrue(any(conflict.get("source") == "openalex" for conflict in intelligence["conflicts"]))

    def test_orchestrator_runs_independent_sources_and_degrades_without_paid_credentials(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=True), patch(
            "runtime.academic_intelligence._openalex_provider",
            return_value=AcademicSourceResult("openalex", SourceStatus.AVAILABLE_FULL),
        ), patch(
            "runtime.academic_intelligence._semantic_scholar_provider",
            return_value=AcademicSourceResult("semantic_scholar", SourceStatus.AVAILABLE_LIMITED),
        ), patch(
            "runtime.academic_intelligence._orcid_provider",
            return_value=AcademicSourceResult("orcid", SourceStatus.AVAILABLE_LIMITED),
        ), patch(
            "runtime.academic_intelligence._crossref_provider",
            return_value=AcademicSourceResult("crossref", SourceStatus.AVAILABLE_FULL),
        ):
            result = academic_intelligence("Unique Missing Credential Researcher professor")

        self.assertEqual(result["source_status"]["scopus"], "UNAVAILABLE")
        self.assertEqual(result["source_status"]["web_of_science"], "UNAVAILABLE")
        self.assertIn("openalex", result["source_status"])
        self.assertIn("COVERAGE_CHECK", result["pipeline"])

    def test_matching_curated_sources_skip_public_fallback(self) -> None:
        scopus = AcademicSourceResult(
            "scopus", SourceStatus.AVAILABLE_FULL,
            ({"name": "Ada Researcher", "source": "scopus", "identifiers": {"scopus_author_id": "1"}},),
            metrics={"document_count": 100},
        )
        wos = AcademicSourceResult(
            "web_of_science", SourceStatus.AVAILABLE_FULL,
            ({"name": "Ada Researcher", "source": "web_of_science", "identifiers": {"wos_researcher_id": "A-1"}},),
            metrics={"document_count": 95},
        )
        with patch.dict(
            os.environ,
            {"SCOPUS_API_KEY": "key", "WOS_API_KEY": "key", "BRAVE_SEARCH_API_KEY": ""},
            clear=True,
        ), patch("runtime.academic_intelligence._scopus_provider", return_value=scopus), patch(
            "runtime.academic_intelligence._wos_provider", return_value=wos
        ), patch("runtime.academic_intelligence._openalex_provider") as openalex, patch(
            "runtime.academic_intelligence._semantic_scholar_provider"
        ) as semantic, patch(
            "runtime.academic_intelligence._orcid_provider"
        ) as orcid, patch(
            "runtime.academic_intelligence._crossref_provider"
        ) as crossref:
            result = academic_intelligence("Unique Curated Agreement Researcher professor")

        self.assertFalse(result["selection_policy"]["public_fallback_triggered"])
        self.assertEqual(set(result["selection_policy"]["providers_called"]), {"scopus", "web_of_science"})
        self.assertNotIn("openalex", result["source_status"])
        openalex.assert_not_called()
        semantic.assert_not_called()
        orcid.assert_not_called()
        crossref.assert_not_called()

    def test_curated_coverage_conflict_triggers_public_fallback(self) -> None:
        scopus = AcademicSourceResult(
            "scopus", SourceStatus.AVAILABLE_FULL,
            ({"name": "Conflict Researcher", "source": "scopus"},),
            metrics={"document_count": 120},
        )
        wos = AcademicSourceResult(
            "web_of_science", SourceStatus.AVAILABLE_FULL,
            ({"name": "Conflict Researcher", "source": "web_of_science"},),
            metrics={"document_count": 20},
        )
        fallback_result = AcademicSourceResult("openalex", SourceStatus.AVAILABLE_FULL)
        with patch.dict(
            os.environ,
            {"SCOPUS_API_KEY": "key", "WOS_API_KEY": "key", "BRAVE_SEARCH_API_KEY": ""},
            clear=True,
        ), patch("runtime.academic_intelligence._scopus_provider", return_value=scopus), patch(
            "runtime.academic_intelligence._wos_provider", return_value=wos
        ), patch(
            "runtime.academic_intelligence._openalex_provider", return_value=fallback_result
        ) as openalex, patch(
            "runtime.academic_intelligence._semantic_scholar_provider",
            return_value=AcademicSourceResult("semantic_scholar", SourceStatus.AVAILABLE_LIMITED),
        ), patch(
            "runtime.academic_intelligence._orcid_provider",
            return_value=AcademicSourceResult("orcid", SourceStatus.AVAILABLE_LIMITED),
        ), patch(
            "runtime.academic_intelligence._crossref_provider",
            return_value=AcademicSourceResult("crossref", SourceStatus.AVAILABLE_FULL),
        ):
            result = academic_intelligence("Unique Coverage Conflict Researcher professor")

        self.assertTrue(result["selection_policy"]["public_fallback_triggered"])
        self.assertIn("orcid", result["selection_policy"]["providers_called"])
        openalex.assert_called_once()

    def test_normalized_researcher_name_cache_avoids_repeated_provider_calls(self) -> None:
        fallback = AcademicSourceResult("openalex", SourceStatus.AVAILABLE_FULL)
        with patch.dict(os.environ, {}, clear=True), patch(
            "runtime.academic_intelligence._openalex_provider", return_value=fallback
        ) as openalex, patch(
            "runtime.academic_intelligence._semantic_scholar_provider",
            return_value=AcademicSourceResult("semantic_scholar", SourceStatus.AVAILABLE_LIMITED),
        ), patch(
            "runtime.academic_intelligence._orcid_provider",
            return_value=AcademicSourceResult("orcid", SourceStatus.AVAILABLE_LIMITED),
        ), patch(
            "runtime.academic_intelligence._crossref_provider",
            return_value=AcademicSourceResult("crossref", SourceStatus.AVAILABLE_FULL),
        ):
            first = academic_intelligence("Unique Cache Researcher professor")
            second = academic_intelligence("unique cache researcher PROFESSOR")

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        openalex.assert_called_once()

    def test_researcher_benchmark_fixtures_expose_identity_and_coverage_risks(self) -> None:
        def identity(source: str, name: str, affiliation: str) -> dict[str, object]:
            return {"name": name, "source": source, "affiliations": [affiliation], "identifiers": {}}

        cases = {
            "long_career_split": (
                "Ada Researcher",
                [
                    AcademicSourceResult("scopus", SourceStatus.AVAILABLE_FULL, (identity("scopus", "Ada Researcher", "A University"),), metrics={"document_count": 140}),
                    AcademicSourceResult("openalex", SourceStatus.AVAILABLE_FULL, (identity("openalex", "Ada Researcher", "A University"),), metrics={"document_count": 30}),
                ],
                "publication_count_discrepancy",
            ),
            "namesake_split_profile": (
                "Alex Kim",
                [AcademicSourceResult("openalex", SourceStatus.AVAILABLE_FULL, (
                    identity("openalex", "Alex Kim", "A University"),
                    identity("openalex", "Alex Kim", "B University"),
                ))],
                "identity_unresolved",
            ),
            "institution_change": (
                "Mina Lee",
                [
                    AcademicSourceResult("scopus", SourceStatus.AVAILABLE_FULL, (identity("scopus", "Mina Lee", "Old University"),)),
                    AcademicSourceResult("orcid", SourceStatus.AVAILABLE_FULL, (identity("orcid", "Mina Lee", "New Institute"),)),
                ],
                "affiliation_mismatch",
            ),
        }
        for label, (name, results, expected_conflict) in cases.items():
            with self.subTest(label=label):
                intelligence = _aggregate_intelligence(name, results)
                self.assertIn(expected_conflict, {item["type"] for item in intelligence["conflicts"]})

        early_career = _aggregate_intelligence("Early Researcher", [
            AcademicSourceResult("scopus", SourceStatus.AVAILABLE_FULL, metrics={"document_count": 4}),
            AcademicSourceResult("openalex", SourceStatus.AVAILABLE_FULL, metrics={"document_count": 3}),
        ])
        self.assertNotIn("publication_count_discrepancy", {item["type"] for item in early_career["conflicts"]})
        korean = _aggregate_intelligence("김민수", [
            AcademicSourceResult("orcid", SourceStatus.AVAILABLE_FULL, (identity("orcid", "김민수", "한국대학교"),)),
        ])
        self.assertEqual(korean["researcher"]["native_name"], "김민수")

    def test_doi_work_with_matching_author_is_verified_only_after_identity_resolution(self) -> None:
        work = {
            "title": "Single Source DOI Work",
            "doi": "10.1000/single",
            "year": 2024,
            "authors": ["Ada Researcher"],
            "sources": ["crossref"],
        }
        resolved = _aggregate_intelligence("Ada Researcher professor", [
            AcademicSourceResult(
                "orcid", SourceStatus.AVAILABLE_FULL,
                ({"name": "Ada Researcher", "source": "orcid", "identifiers": {"orcid": "0000-1"}},),
            ),
            AcademicSourceResult(
                "google_scholar", SourceStatus.AVAILABLE_LIMITED,
                ({"name": "Ada Researcher", "source": "google_scholar", "identifiers": {"google_scholar_profile": "https://example.test"}},),
            ),
            AcademicSourceResult("crossref", SourceStatus.AVAILABLE_FULL, publications=(work,)),
        ])
        unresolved = _aggregate_intelligence(
            "Ada Researcher professor",
            [AcademicSourceResult("crossref", SourceStatus.AVAILABLE_FULL, publications=(work,))],
        )

        self.assertEqual(resolved["publication_candidates"][0]["authorship_confidence"], "MEDIUM")
        self.assertEqual(resolved["merged_publication_count"], 1)
        self.assertEqual(unresolved["publication_candidates"][0]["authorship_confidence"], "LOW")
        self.assertEqual(unresolved["merged_publication_count"], 0)


if __name__ == "__main__":
    unittest.main()