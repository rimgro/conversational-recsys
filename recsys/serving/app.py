"""CPU-only right-tower search API. Run with uvicorn's factory mode."""
from contextlib import asynccontextmanager
from pathlib import Path
import hmac
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from .contract import SearchRequest, SearchResponse
from .index import CandidateIndex

MAX_BODY_BYTES = 4 * 1024 * 1024


class RequestGuard:
    """Authenticate before decoding JSON; cap even chunked bodies before buffering."""
    def __init__(self, app, api_key, max_body_bytes=MAX_BODY_BYTES):
        self.app, self.api_key, self.limit = app, api_key.encode(), max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        async def reject(code, message):
            await JSONResponse({"detail": message}, status_code=code,
                headers={"Connection": "close"})(scope, receive, send)
        if scope["path"] not in {"/health", "/live"}:
            if not hmac.compare_digest(headers.get(b"x-api-key", b""), self.api_key):
                return await reject(401, "Invalid API key")
        if scope["method"] == "POST":
            size = headers.get(b"content-length")
            if size is not None:
                try:
                    size = int(size)
                except ValueError:
                    return await reject(400, "Invalid content length")
                if size < 0 or size > self.limit:
                    return await reject(413, "Request body too large")
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.limit:
                    return await reject(413, "Request body too large")
                if not message.get("more_body", False):
                    break
            delivered = False
            async def replay():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


def create_app(bundle_dir, api_key, index_factory=CandidateIndex):
    if not api_key or len(api_key) < 24 or "\n" in api_key or "\r" in api_key:
        raise ValueError("Set an API key of at least 24 characters")

    @asynccontextmanager
    async def lifespan(app):
        app.state.index = index_factory(bundle_dir)
        yield
        del app.state.index

    app = FastAPI(title="Qwen right tower", version="1", lifespan=lifespan,
        docs_url=None, redoc_url=None)
    app.add_middleware(RequestGuard, api_key=api_key)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Do not echo vectors/history or nonfinite values into error responses.
        return JSONResponse(status_code=422, content={"detail": "Invalid request",
            "errors": [{k: e[k] for k in ("loc", "type", "msg")} for e in error.errors()]})

    @app.get("/live")
    def live():
        return {"status": "ok"}

    @app.get("/health")
    def health():
        return app.state.index.health()

    @app.post("/v1/search", response_model=SearchResponse)
    def search(body: SearchRequest):
        index = app.state.index
        if (body.bundle_id != index.manifest["bundle_id"]
                or body.release_id != index.manifest["release_id"]):
            raise HTTPException(409, "Query encoder and index releases differ")
        started = time.perf_counter()
        kwargs = body.model_dump(exclude={"bundle_id", "release_id"})
        try:
            result = index.search(**kwargs)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return result
    return app


def create_env_app():
    path = os.environ.get("QWEN_API_KEY_FILE")
    key = os.environ.get("QWEN_API_KEY")
    if path and key:
        raise ValueError("Use QWEN_API_KEY_FILE or QWEN_API_KEY, not both")
    if path:
        key = Path(path).read_text().strip()
    bundle = os.environ.get("QWEN_RIGHT_BUNDLE")
    if not bundle:
        raise ValueError("QWEN_RIGHT_BUNDLE is required")
    return create_app(bundle, key)
