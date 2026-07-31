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
from azure_clients.redis_client import warm_up as warm_up_redis
from azure_clients.sql_client import sql_client


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
    #
    # Redis joins them: it is a lazy singleton whose first call pays the TCP
    # connect + TLS handshake + ping, measured at ~3.3s. That landed on the
    # first cache lookup of the first analysis, i.e. inside retrieval.
    await asyncio.gather(
        asyncio.to_thread(warm_up_cross_encoder),
        asyncio.to_thread(warm_up_finbert),
        asyncio.to_thread(warm_up_redis),
    )

    # Azure SQL is warmed fire-and-forget, NOT awaited, because it is the one
    # dependency that can take a minute: the database is Serverless and
    # auto-pauses when idle, so the first connection triggers a resume --
    # measured at 49.4s, against ~0.95s once awake. Awaiting it would hold the
    # container un-ready for that whole time; skipping it entirely would charge
    # it to numeric_validation_agent inside the first user's analysis.
    #
    # Kicking it off here overlaps the resume with model loading and with the
    # user's own retrieval/sentiment stages, so by the time numeric validation
    # runs the connection is usually live. Failures are swallowed: SQL being
    # unreachable must not stop the API from serving, and calculate_metric
    # already retries on its own.
    async def _warm_sql() -> None:
        try:
            await asyncio.to_thread(sql_client.warm_up)
        except Exception as exc:  # noqa: BLE001 — best-effort warm-up
            print(f"[startup] SQL warm-up failed (non-fatal): {exc}")

    sql_task = asyncio.create_task(_warm_sql())

    yield

    sql_task.cancel()
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