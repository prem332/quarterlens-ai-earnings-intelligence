# Phase 3 placeholder — sliding window rate limiting per client IP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RateLimiterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await call_next(request)