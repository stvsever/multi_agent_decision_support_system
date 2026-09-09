"""FastAPI application for the COMPASS dashboard."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from ..config.settings import COMPASS_FULL_NAME, COMPASS_VERSION
from .paths import WEB_CLIENT_DIR
from .routers import (
    batch_router,
    catalog_router,
    datasets_router,
    hf_router,
    prompts_router,
    reports_router,
    runs_router,
    settings_router,
    system,
)
from .run_manager import get_run_manager

logger = logging.getLogger("compass.api")

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_run_manager()
    yield
    get_run_manager().shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="COMPASS Dashboard",
        description=COMPASS_FULL_NAME,
        version=COMPASS_VERSION,
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )

    # The dev client runs on its own Vite origin; the built client is same-origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (
        system.router,
        settings_router.router,
        catalog_router.router,
        hf_router.router,
        datasets_router.router,
        runs_router.router,
        batch_router.router,
        reports_router.router,
        prompts_router.router,
    ):
        app.include_router(router, prefix=API_PREFIX)

    @app.exception_handler(ValidationError)
    async def invalid_config(request: Request, exc: ValidationError) -> JSONResponse:
        """A constraint declared in schemas.py is the caller's mistake, not ours."""
        details = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', 'invalid value')}".lstrip(": ")
            for e in exc.errors()
        )
        return JSONResponse(status_code=422, content={"detail": details or "Invalid configuration."})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    _mount_client(app)
    return app


def _mount_client(app: FastAPI) -> None:
    """Serve the built web client, falling back to a build hint when absent."""
    index = WEB_CLIENT_DIR / "index.html"
    if not index.exists():
        @app.get("/{full_path:path}", include_in_schema=False)
        async def missing_client(full_path: str) -> JSONResponse:
            if full_path == "api" or full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": f"No such endpoint: /{full_path}"})
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "The web client has not been built.",
                    "fix": "cd src/full_stack/frontend && npm install && npm run build",
                    "dev": "npm run dev serves the client on http://localhost:5173",
                },
            )
        return

    assets = WEB_CLIENT_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    root = WEB_CLIENT_DIR.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        # The catch-all exists for client-side routes. It must never answer for
        # the API namespace, or a renamed endpoint would return the app shell
        # with a 200 instead of a 404.
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"No such endpoint: /{full_path}")
        # Client routes fall through to index.html, but the path is attacker
        # controlled, so a resolved candidate that escapes the bundle directory
        # must never be served: `/../../../.env` would otherwise return secrets.
        if full_path:
            try:
                candidate = (root / full_path).resolve()
            except (OSError, ValueError):
                candidate = None
            if candidate is not None and candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(str(candidate))
        return FileResponse(str(index))
