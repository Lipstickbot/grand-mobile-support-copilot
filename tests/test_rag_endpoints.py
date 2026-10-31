"""HTTP adapter tests with in-memory engines, never a model, GPU or external API."""
from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import UUID

from fastapi.testclient import TestClient as _TestClient
from functools import partial
TestClient = partial(_TestClient, base_url="http://127.0.0.1")
import pytest

from support_copilot.api import MAX_REQUEST_BYTES, create_app
from support_copilot.config import Settings
from support_copilot.schemas import RagSearchRequest


PRIVATE = "PRIVATE-RAG-REQUEST-must-not-echo"


def triage_record():
    return {"kind": "report", "messages": [{"role": "player", "text": "hello"}]}


def triage_response():
    return {"lang": "en", "category": "report.help", "priority": "p3",
            "is_spam_or_duplicate": False,
            "extracted": {"nickname": None, "rule_refs": [], "has_evidence": False, "server": None},
            "draft_reply": {"lang": "en", "text": "Please provide details.", "cited": []},
            "needs_human": True, "confidence": .3, "reasons": ["synthetic fixture"]}


def search_response():
    return {"query_lang": "ru", "corpus_version": "synthetic-v1", "elapsed_ms": 1.25,
            "warnings": ["Scores are not calibrated relevance probabilities."],
            "results": [{"canonical_id": "rules:900:2.1", "section_id": 900,
                "section_title": "Synthetic section", "ref": "2.1", "text": "Synthetic rule.",
                "score": .05, "source_type": "rule"}]}


class Engine:
    def __init__(self):
        self.ready = True
        self.loads = self.closes = 0
        self.searches = []
        self.triages = []
        self.failure = None

    def load(self):
        self.loads += 1

    def close(self):
        self.closes += 1

    def health(self):
        return {"ready": self.ready}

    def metadata(self):
        if self.failure:
            raise self.failure
        return {"contract_version": "v1", "categories": [{"code": "report.help"}]}

    def search_knowledge(self, payload):
        assert isinstance(payload, RagSearchRequest)
        self.searches.append(payload)
        if self.failure:
            raise self.failure
        return search_response()

    def get_rule(self, section_id, ref):
        if self.failure:
            raise self.failure
        return search_response()["results"][0] if (section_id, ref) == (900, "2.1") else None

    def triage(self, payload):
        self.triages.append(payload)
        return triage_response()


def app_for(engine):
    return create_app(Settings(), lambda settings: engine)


def assert_request_id(response):
    value = response.headers["X-Request-ID"]
    assert str(UUID(value, version=4)) == value
    return value


def test_live_without_engine_loading_or_health_calls():
    engine = Engine()
    engine.health = lambda: (_ for _ in ()).throw(AssertionError("Liveness touched engine"))
    with_factory = []
    app = create_app(Settings(), lambda settings: with_factory.append(settings) or engine)
    client = TestClient(app)
    response = client.get("/live", headers={"X-Request-ID": PRIVATE})
    assert response.status_code == 200 and response.json() == {"alive": True}
    assert_request_id(response)
    assert PRIVATE not in response.headers["X-Request-ID"]
    assert not with_factory and engine.loads == 0
    assert client.get("/health").status_code == 503


def test_metadata_search_and_readiness_preserve_independent_contracts():
    engine = Engine()
    with TestClient(app_for(engine)) as client:
        meta = client.get("/v1/meta")
        assert meta.status_code == 200 and meta.json()["contract_version"] == "v1"
        assert_request_id(meta)
        assert client.get("/ready").json() == {"ready": True}
        result = client.post("/v1/rag/search", json={"query": "Оскорбления", "rule_refs": ["2.1"], "limit": 3, "lang": "ru"})
        assert result.status_code == 200 and result.json() == search_response()
        assert_request_id(result)
        assert len(engine.searches) == 1 and engine.searches[0].limit == 3
        assert not engine.triages  # retrieval did not ask the model to classify/generate
        engine.ready = False
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 503
        assert client.get("/live").json() == {"alive": True}
    assert engine.loads == engine.closes == 1


def test_new_routes_before_lifespan_fail_safely_with_identifiers():
    client = TestClient(app_for(Engine()))
    for result in (client.get("/v1/meta"), client.get("/ready"), client.get("/v1/rag/rules/900/2.1"),
                   client.post("/v1/rag/search", json={"query": "valid"})):
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "not_ready"
        assert_request_id(result)


def test_exact_rule_lookup_is_section_scoped_without_search_or_generation():
    engine = Engine()
    with TestClient(app_for(engine)) as client:
        result = client.get("/v1/rag/rules/900/2.1")
        assert result.status_code == 200 and result.json() == search_response()["results"][0]
        assert_request_id(result)
        missing = client.get("/v1/rag/rules/901/2.1")
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "rule_not_found"
        assert_request_id(missing)
        assert not engine.searches and not engine.triages
        engine.failure = RuntimeError(PRIVATE)
        failure = client.get("/v1/rag/rules/900/2.1")
        assert failure.status_code == 500 and PRIVATE not in failure.text
        assert_request_id(failure)


@pytest.mark.parametrize("path", ["0/2.1", "100001/2.1", "900/not-a-ref", "900/2", "900/1.2.3.4.5"])
def test_rule_lookup_validates_reference_and_section(path):
    with TestClient(app_for(Engine())) as client:
        response = client.get("/v1/rag/rules/" + path)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert_request_id(response)


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "   "}, {"query": PRIVATE * 200},
    {"query": PRIVATE, "limit": 0}, {"query": PRIVATE, "limit": 11},
    {"query": PRIVATE, "rule_refs": [PRIVATE] * 21},
    {"query": PRIVATE, "rule_refs": [PRIVATE * 2]},
    {"query": PRIVATE, "lang": "invalid"},
])
def test_retrieval_request_limits_never_echo_invalid_input(payload):
    engine = Engine()
    with TestClient(app_for(engine)) as client:
        result = client.post("/v1/rag/search", json=payload)
        assert result.status_code == 422
        assert result.json()["error"]["code"] == "invalid_request"
        assert PRIVATE not in result.text
        assert_request_id(result)
        assert not engine.searches


@pytest.mark.parametrize("failure,status,code", [
    (RuntimeError(PRIVATE), 500, "retrieval_failed"),
    (TimeoutError(PRIVATE), 504, "retrieval_timeout"),
])
def test_retrieval_errors_are_safe_and_release_gate(failure, status, code):
    engine = Engine()
    engine.failure = failure
    with TestClient(app_for(engine)) as client:
        result = client.post("/v1/rag/search", json={"query": PRIVATE})
        assert result.status_code == status and result.json()["error"]["code"] == code
        assert PRIVATE not in result.text
        assert_request_id(result)
        engine.failure = None
        assert client.post("/v1/rag/search", json={"query": "retry"}).status_code == 200


def test_metadata_and_invalid_retrieval_response_hide_private_provider_values():
    engine = Engine()
    engine.failure = RuntimeError(PRIVATE)
    with TestClient(app_for(engine)) as client:
        metadata = client.get("/v1/meta")
        assert metadata.status_code == 500 and PRIVATE not in metadata.text
        assert_request_id(metadata)
        engine.search_knowledge = lambda request: {"private": PRIVATE}
        response = client.post("/v1/rag/search", json={"query": "valid"})
        assert response.status_code == 500 and PRIVATE not in response.text
        assert_request_id(response)


def test_ids_are_server_generated_unique_and_present_on_413_404_and_uncaught_500():
    app = app_for(Engine())

    @app.get("/test-uncaught")
    async def fail():
        raise RuntimeError(PRIVATE)

    with TestClient(app) as client:
        responses = [client.get("/live", headers={"X-Request-ID": PRIVATE}),
                     client.get("/missing"), client.get("/test-uncaught"),
                     client.post("/v1/rag/search", content=iter([b"a" * MAX_REQUEST_BYTES, b"b"]))]
        assert [r.status_code for r in responses] == [200, 404, 500, 413]
        ids = [assert_request_id(r) for r in responses]
        assert len(set(ids)) == len(ids)
        assert all(PRIVATE not in r.text and PRIVATE not in r.headers["X-Request-ID"] for r in responses)


def test_search_and_inference_have_separate_bounded_admission():
    engine = Engine()
    entered_search, release_search = threading.Event(), threading.Event()
    entered_triage, release_triage = threading.Event(), threading.Event()

    def search(payload):
        entered_search.set()
        assert release_search.wait(8)
        return search_response()

    def triage(payload):
        entered_triage.set()
        assert release_triage.wait(8)
        return triage_response()

    engine.search_knowledge, engine.triage = search, triage
    with TestClient(app_for(engine)) as client, ThreadPoolExecutor() as pool:
        first_search = pool.submit(client.post, "/v1/rag/search", json={"query": "first"})
        first_triage = None
        try:
            assert entered_search.wait(3)
            first_triage = pool.submit(client.post, "/triage", json=triage_record())
            assert entered_triage.wait(3)  # busy retrieval does not queue generation
            search_busy = client.post("/v1/rag/search", json={"query": "second"})
            triage_busy = client.post("/triage", json=triage_record())
            assert search_busy.status_code == triage_busy.status_code == 503
            assert search_busy.json()["error"]["code"] == triage_busy.json()["error"]["code"] == "busy"
            assert_request_id(search_busy)
            assert client.get("/live").status_code == 200
            assert client.get("/health").status_code == 200
            assert client.get("/v1/meta").status_code == 200
        finally:
            release_search.set()
            release_triage.set()
        assert first_search.result(timeout=3).status_code == 200
        assert first_triage.result(timeout=3).status_code == 200


def test_openapi_exposes_operations_metadata_knowledge_and_triage_groups():
    schema = app_for(Engine()).openapi()
    assert schema["paths"]["/live"]["get"]["tags"] == ["operations"]
    assert schema["paths"]["/v1/meta"]["get"]["tags"] == ["metadata"]
    assert schema["paths"]["/v1/rag/search"]["post"]["tags"] == ["knowledge"]
    assert schema["paths"]["/triage"]["post"]["tags"] == ["triage"]
