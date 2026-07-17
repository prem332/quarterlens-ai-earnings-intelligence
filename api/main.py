from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.middleware.guardrails import GuardrailsMiddleware
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.routes import analysis, reports, evidence, export
from observability.azure_monitor_setup import setup_azure_monitor
from observability.phoenix_setup import setup_phoenix
from observability.langfuse_setup import setup_langfuse, flush_langfuse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise observability. Shutdown: flush Langfuse events."""
    setup_azure_monitor()   # Azure Monitor first — sets up OTEL provider
    setup_phoenix()         # Phoenix cloud tracing
    setup_langfuse()        # Langfuse LLM call monitoring
    yield
    flush_langfuse()


app = FastAPI(
    title="QuarterLens AI",
    description="Earnings intelligence platform — SEC filing cross-verification",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Phase 3 middleware — pass-through for now
app.add_middleware(GuardrailsMiddleware)
app.add_middleware(RateLimiterMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(analysis.router)
app.include_router(reports.router)
app.include_router(evidence.router)
app.include_router(export.router)

# Serve React build — must come after API routes
_static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(os.path.join(_static_dir, "index.html"))