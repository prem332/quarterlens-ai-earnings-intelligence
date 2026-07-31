import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.routes import analysis, reports, export
from observability.azure_monitor_setup import setup_azure_monitor
from observability.phoenix_setup import setup_phoenix
from observability.langfuse_setup import setup_langfuse, flush_langfuse
from tools.rerank_documents import warm_up as warm_up_cross_encoder
from tools.run_finbert import warm_up as warm_up_finbert


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise observability + warm the cross-encoder. Shutdown: flush Langfuse events."""
    setup_azure_monitor()   # Azure Monitor first — sets up OTEL provider
    setup_phoenix()         # Phoenix cloud tracing
    setup_langfuse()        # Langfuse LLM call monitoring
    # Load both CPU models now instead of inside whichever user's request
    # happens to be first. to_thread so the event loop can still serve
    # /api/health while they load. Concurrently because they are independent
    # and both are pure CPU/disk work.
    #
    # FinBERT was missing here until measured: sentiment_agent took 7.0s on a
    # fresh server's first request and 0.3-0.4s on every one after, all of it
    # model loading (warm inference is ~0.06s/passage). The cross-encoder was
    # already warmed; FinBERT silently charged that 7s to the first user.
    await asyncio.gather(
        asyncio.to_thread(warm_up_cross_encoder),
        asyncio.to_thread(warm_up_finbert),
    )
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(analysis.router)
app.include_router(reports.router)
app.include_router(export.router)


@app.get("/api/health", include_in_schema=False)
async def health() -> dict:
    """Liveness probe — no Azure calls, so it stays fast even if a dependency
    (Key Vault, AI Search, ...) is degraded. Container orchestrators poll this."""
    return {"status": "ok"}

# Serve React build — must come after API routes
_static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_static_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(os.path.join(_static_dir, "index.html"))