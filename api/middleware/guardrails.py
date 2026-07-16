# Phase 3 placeholder — prompt injection detection, PII scrubbing, off-domain filtering
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class GuardrailsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await call_next(request)