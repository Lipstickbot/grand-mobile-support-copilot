"""Local HTTP adapter. Importing this module never loads model weights."""

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path as FilePath
from uuid import uuid4
from urllib.parse import urlsplit

from fastapi import FastAPI, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, HTMLResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .config import Settings, ROOT
from .schemas import KnowledgeHit, RagSearchRequest, RagSearchResponse, TriageRequest, TriageResponse, ClassificationResponse

MAX_REQUEST_BYTES = 128 * 1024


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _default_engine_factory(settings):
    from .engine import TriageEngine

    return TriageEngine(settings)


def _is_ready(health: dict) -> bool:
    return health.get("ready", health.get("status") == "ready") is True


class _RequestID:
    """Server-generated IDs on every HTTP response; caller headers are not trusted."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        started = False

        async def identified_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                # Do not allow an inner handler to replace the server's identifier.
                headers = [(name, value) for name, value in message.get("headers", [])
                           if name.lower() != b"x-request-id"]
                message = {**message, "headers": headers + [(b"x-request-id", request_id.encode("ascii"))]}
            await send(message)

        try:
            return await self.app(scope, receive, identified_send)
        except Exception:
            # Catch before the outer server-error middleware so its 500 also has
            # a request ID and never serializes an exception containing input.
            if started:
                raise
            return await _error("internal_error", "Request could not be completed.", 500)(
                scope, receive, identified_send)


class _BoundedBody:
    """Bound bytes before JSON decoding, including requests without Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                return await _error("request_too_large", "Request exceeds 128 KiB.", 413)(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        return await self.app(scope, bounded_receive, send)


class _LocalBoundary:
    """Loopback service: do not accept arbitrary browser hosts or foreign writes."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {key.lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        host = headers.get(b"host", "")
        try:
            parsed = urlsplit("http://" + host)
            allowed = (parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.username is None
                       and not parsed.path and not parsed.query and not parsed.fragment)
            _ = parsed.port
        except ValueError:
            allowed = False
        if not allowed:
            return await _error("untrusted_host", "Only loopback service hosts are allowed.", 400)(scope, receive, send)
        origin = headers.get(b"origin")
        if origin is not None and scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
            try:
                parsed_origin = urlsplit(origin)
                same_origin = (parsed_origin.scheme == scope.get("scheme", "http")
                               and parsed_origin.netloc.casefold() == host.casefold()
                               and not parsed_origin.path and not parsed_origin.query and not parsed_origin.fragment)
            except ValueError:
                same_origin = False
            if not same_origin:
                return await _error("foreign_origin", "Cross-origin writes are not allowed.", 403)(scope, receive, send)
        return await self.app(scope, receive, send)


def create_app(settings=None, engine_factory=None) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    factory = engine_factory if engine_factory is not None else _default_engine_factory

    @asynccontextmanager
    async def lifespan(app):
        engine = factory(settings)
        app.state.engine = engine
        app.state.loaded = False
        app.state.inference_gate = asyncio.Semaphore(1)
        app.state.retrieval_gate = asyncio.Semaphore(1)
        try:
            try:
                await run_in_threadpool(engine.load)
                app.state.loaded = True
            except Exception:
                raise RuntimeError("Engine initialization failed.") from None
            yield
        finally:
            app.state.loaded = False
            try:
                await run_in_threadpool(engine.close)
            except Exception:
                raise RuntimeError("Engine shutdown failed.") from None

    app = FastAPI(title="Local Support Copilot", lifespan=lifespan)
    app.state.loaded = False
    app.add_middleware(_BoundedBody)
    app.add_middleware(_LocalBoundary)
    app.add_middleware(_RequestID)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        from .dashboard import HTML
        return HTMLResponse(HTML, headers={"Cache-Control": "no-store", "Content-Security-Policy":
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})

    @app.get("/assets/dashboard.js", include_in_schema=False)
    async def dashboard_script():
        from .dashboard import JS
        return Response(JS, media_type="application/javascript", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    def read_private_report(path, require_schema=True):
        path = FilePath(path).resolve()
        if not path.is_relative_to((ROOT / ".private").resolve()) or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Invalid report location or size")
        report = json.loads(path.read_text(encoding="utf8"))
        if (require_schema and report.get("schema_version") != 1) or report.get("human_gold") is not False:
            raise ValueError("Unsupported report provenance")
        return report

    @app.get("/v1/evaluation", tags=["evaluation"])
    async def evaluation():
        try:
            report = await run_in_threadpool(read_private_report, settings.evaluation_report_path)
            if report.get("label_kind") != "synthetic_development_not_human_gold":
                raise ValueError("Not a synthetic development report")
            return JSONResponse(report, headers={"Cache-Control": "no-store"})
        except Exception:
            return _error("evaluation_unavailable", "A verified local evaluation report is not available.", 404)

    @app.get("/v1/quality", tags=["evaluation"])
    async def current_quality():
        try:
            from .quality_report import PublicQualityReport
            report = await run_in_threadpool(read_private_report, ROOT / ".private/quality-v5/dashboard-quality.json")
            safe = PublicQualityReport.model_validate(report).model_dump(mode="json")
            return JSONResponse(safe, headers={"Cache-Control": "no-store"})
        except Exception:
            return _error("quality_unavailable", "A completed local quality comparison is not available.", 404)

    @app.get("/v1/robustness", tags=["evaluation"])
    async def input_robustness():
        try:
            report = await run_in_threadpool(read_private_report, ROOT / ".private/input-v4/summary.json")
            if report.get("label_kind") != "authored_synthetic_regression_not_human_gold":
                raise ValueError("Unsupported robustness report")
            return JSONResponse(report, headers={"Cache-Control": "no-store"})
        except Exception:
            return _error("robustness_unavailable", "Input robustness report is not available.", 404)

    @app.get("/v1/classifier/metrics", tags=["evaluation"])
    async def classifier_metrics():
        try:
            report = await run_in_threadpool(read_private_report, settings.classifier_path.parent / "metrics.json")
            if report.get("label_kind") != "weak_and_synthetic_not_human_gold":
                raise ValueError("Unsupported classifier metrics")
            # Emit only numeric aggregate reports, never preparation rows or paths.
            safe = {key: report[key] for key in ("schema_version", "label_kind", "human_gold",
                "weak_label_holdout_agreement", "authored_synthetic_challenge")}
            try:
                integrated = await run_in_threadpool(read_private_report,
                    settings.classifier_path.parent / "integrated-metrics.json", False)
                safe["integrated_challenge"] = {key: integrated[key] for key in (
                    "scope", "n", "models", "learned_abstentions", "learned_confident_categories")}
            except (FileNotFoundError, KeyError, ValueError):
                safe["integrated_challenge"] = None
            return JSONResponse(safe, headers={"Cache-Control": "no-store"})
        except Exception:
            return _error("classifier_metrics_unavailable", "Classifier measurements are not available.", 404)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        return _error("invalid_request", "Request does not match the API schema.", 422)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        return _error("http_error", "HTTP request could not be processed.", exc.status_code)

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, _exc: Exception):
        return _error("internal_error", "Request could not be completed.", 500)

    @app.get("/live", tags=["operations"])
    async def live():
        # Liveness never touches weights, retrieval state or the inference lock.
        return {"alive": True}

    @app.get("/ready", tags=["operations"])
    @app.get("/health", tags=["operations"])
    async def health():
        if not app.state.loaded:
            return _error("not_ready", "Engine is not ready.", 503)
        try:
            result = await run_in_threadpool(app.state.engine.health)
            return JSONResponse(result, status_code=200 if _is_ready(result) else 503)
        except Exception:
            return _error("not_ready", "Engine health check failed.", 503)

    @app.get("/v1/meta", tags=["metadata"])
    async def metadata():
        if not app.state.loaded:
            return _error("not_ready", "Engine is not ready.", 503)
        try:
            return await run_in_threadpool(app.state.engine.metadata)
        except Exception:
            return _error("metadata_failed", "Service metadata could not be read.", 500)

    @app.post("/v1/rag/search", response_model=RagSearchResponse, tags=["knowledge"])
    async def search_knowledge(payload: RagSearchRequest):
        if not app.state.loaded:
            return _error("not_ready", "Engine is not ready.", 503)
        gate = app.state.retrieval_gate
        if gate.locked():
            return _error("busy", "Knowledge search is busy; retry later.", 503)
        async with gate:
            try:
                result = await run_in_threadpool(app.state.engine.search_knowledge, payload)
                return RagSearchResponse.model_validate(result)
            except TimeoutError:
                return _error("retrieval_timeout", "Knowledge search timed out.", 504)
            except Exception:
                return _error("retrieval_failed", "Knowledge search could not be completed.", 500)

    @app.get("/v1/rag/rules/{section_id}/{ref}", response_model=KnowledgeHit, tags=["knowledge"])
    async def get_rule(section_id: int = Path(ge=1, le=100000),
                       ref: str = Path(max_length=32, pattern=r"^\d+(?:\.\d+){1,3}$")):
        if not app.state.loaded:
            return _error("not_ready", "Engine is not ready.", 503)
        try:
            rule = await run_in_threadpool(app.state.engine.get_rule, section_id, ref)
            if rule is None:
                return _error("rule_not_found", "Rule was not found.", 404)
            return KnowledgeHit.model_validate(rule)
        except Exception:
            return _error("rule_lookup_failed", "Rule could not be read.", 500)

    @app.post("/triage", response_model=TriageResponse, tags=["triage"])
    async def triage(payload: TriageRequest):
        if not app.state.loaded:
            return _error("not_ready", "Engine is not ready.", 503)
        gate = app.state.inference_gate
        if gate.locked():
            return _error("busy", "Engine is busy; retry later.", 503)
        async with gate:
            try:
                result = await run_in_threadpool(app.state.engine.triage, payload)
                # Validate inside the protected boundary so response errors cannot echo data.
                return TriageResponse.model_validate(result)
            except TimeoutError:
                return _error("timeout", "Inference timed out.", 504)
            except Exception:
                return _error("inference_failed", "Inference could not be completed.", 500)

    @app.post("/v1/classify", response_model=ClassificationResponse, tags=["classification"])
    async def classify_only(payload: TriageRequest):
        if not app.state.loaded:
            return _error("not_ready", "Engine is not ready.", 503)
        gate = app.state.retrieval_gate
        if gate.locked():
            return _error("busy", "CPU classifier and search are busy; retry later.", 503)
        async with gate:
            try:
                result = await run_in_threadpool(app.state.engine.classify_request, payload)
                return ClassificationResponse.model_validate(result)
            except Exception:
                return _error("classification_failed", "Classification could not be completed.", 500)

    return app
