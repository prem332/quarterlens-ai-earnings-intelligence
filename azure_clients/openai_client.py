"""
azure_clients/openai_client.py

Azure OpenAI client for:
  - Chat completions (gpt-5-mini) — streaming and non-streaming
  - Async chat completions (achat) — for async agent nodes
  - Embeddings (text-embedding-3-small)

Secrets sourced from Key Vault via the kv singleton.

gpt-5-mini behaviour notes (from Azure portal sample code):
  - api_version must be "2024-12-01-preview" (not 2024-08-01-preview)
  - max_completion_tokens should be 16384 (Azure recommended value)
  - Does NOT support temperature parameter
  - IS a reasoning model — uses internal reasoning tokens before output
  - Minimum safe max_completion_tokens: 4096 to avoid empty responses
"""

import logging
from typing import Optional

try:
    from langfuse.openai import AzureOpenAI
    from openai import AsyncAzureOpenAI
    _langfuse_instrumented = True
except ImportError:
    from openai import AzureOpenAI, AsyncAzureOpenAI
    _langfuse_instrumented = False

from azure_clients.key_vault_client import kv

logger = logging.getLogger(__name__)

# Minimum token budget — below this, reasoning model produces empty responses
_MIN_SAFE_TOKENS = 4096

# Azure-recommended max for gpt-5-mini
_DEFAULT_MAX_TOKENS = 16384

# Embedding dimensionality — must match the AI Search index schema
EMBEDDING_DIMENSIONS = 1536


class OpenAIClient:
    """
    Thin wrapper around AzureOpenAI for chat and embedding calls.
    One instance shared across all agents via the module-level singleton.
    Exposes both sync (chat) and async (achat) interfaces — same credentials,
    same deployment. Async client used by async agent nodes (Phase 2).
    """

    def __init__(self):
        endpoint = kv.get_secret("AZURE-OPENAI-ENDPOINT")
        api_key = kv.get_secret("AZURE-OPENAI-KEY")
        self._chat_deployment = kv.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME")
        self._standard_deployment = kv.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME-STANDARD")
        self._embedding_deployment = "text-embedding-3-small"

        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-12-01-preview",  # required for gpt-5-mini
        )

        self._async_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-12-01-preview",
        )

        logger.info(
            "OpenAIClient: connected — primary=%s, standard=%s, embedding=%s",
            self._chat_deployment,
            self._standard_deployment,
            self._embedding_deployment,
        )

    # ------------------------------------------------------------------
    # Chat completions (sync)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 1.0,               # kept for API compat; not passed to model
        max_tokens: int = _DEFAULT_MAX_TOKENS,  # kept for API compat
        max_completion_tokens: Optional[int] = None,
    ) -> object:
        """
        Non-streaming chat completion.

        Args:
            messages:               OpenAI message list.
            tools:                  Tool schemas for function calling (optional).
            tool_choice:            "auto" | "none" | specific tool name (optional).
            temperature:            Ignored — gpt-5-mini does not support this parameter.
            max_tokens:             Alias for max_completion_tokens (kept for compat).
            max_completion_tokens:  Max tokens. Takes precedence over max_tokens.
                                    Must be >= 4096 for gpt-5-mini reasoning model.

        Returns:
            The full ChatCompletion response object.
        """
        limit = max_completion_tokens or max_tokens
        limit = max(limit, _MIN_SAFE_TOKENS)

        kwargs = dict(
            model=self._chat_deployment,
            messages=messages,
            max_completion_tokens=limit,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = self._client.chat.completions.create(**kwargs)
        logger.debug(
            "OpenAIClient.chat: %d prompt + %d completion tokens",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return response

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Tiered async chat — model routing (Phase 2)
    # ------------------------------------------------------------------

    async def achat_tiered(
        self,
        messages: list[dict],
        model_tier: str = "primary",
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_completion_tokens: Optional[int] = None,
    ) -> object:
        """
        Async chat completion with model tier routing.

        Routes to:
          "primary"   → gpt-5.4-mini (complex reasoning, comparison, report)
          "standard"  → gpt-5-mini   (simple fact lookups)

        Args:
            messages:    OpenAI message list.
            model_tier:  "primary" | "standard". Defaults to "primary".
            tools:       Tool schemas for function calling (optional).
            tool_choice: "auto" | "none" | specific tool name (optional).
            max_tokens / max_completion_tokens: token cap; floored at 4096.

        Returns:
            The full ChatCompletion response object.
        """
        deployment = (
            self._standard_deployment if model_tier == "standard"
            else self._chat_deployment
        )

        limit = max_completion_tokens or max_tokens
        limit = max(limit, _MIN_SAFE_TOKENS)

        kwargs = dict(
            model=deployment,
            messages=messages,
            max_completion_tokens=limit,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = await self._async_client.chat.completions.create(**kwargs)
        logger.debug(
            "OpenAIClient.achat_tiered [%s/%s]: %d prompt + %d completion tokens",
            model_tier, deployment,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return response

    async def achat_tiered_stream(
        self,
        messages: list[dict],
        model_tier: str = "primary",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_completion_tokens: Optional[int] = None,
    ):
        """
        Same routing/token-floor rules as achat_tiered, but yields text deltas
        as they arrive instead of waiting for the full response.

        Used by report_agent's draft step so a UI can show tokens appearing
        live. Not used anywhere on the eval path — run_baseline_eval.py and
        every other caller keep using the non-streaming achat_tiered.

        Yields: str chunks. Concatenate everything yielded to get the full text.
        """
        deployment = (
            self._standard_deployment if model_tier == "standard"
            else self._chat_deployment
        )
        limit = max_completion_tokens or max_tokens
        limit = max(limit, _MIN_SAFE_TOKENS)

        stream = await self._async_client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=limit,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """
        Embed a single string.
        L1 cache: returns cached embedding if available, skipping API call.

        Returns:
            1536-dim embedding vector.
        """
        from azure_clients.redis_client import get_embedding_cached, set_embedding_cached

        # L1 cache check — returns instantly if seen before
        cached = get_embedding_cached(text)
        if cached is not None:
            return cached

        response = self._client.embeddings.create(
            model=self._embedding_deployment,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        embedding = response.data[0].embedding

        # Store in L1 cache for future calls
        set_embedding_cached(text, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings in one API call.

        L1 cache: only the texts not already cached are sent to the API; the
        rest are spliced back in by index. This matters because the dominant
        caller is mmr_rerank(), which re-embeds the same retrieved chunk texts
        on every single retrieval — measured at 1.3-1.9s per call for ~24
        chunks (~14.5K tokens), paid even when the exact query was just run.
        Chunk text is immutable, so those vectors never need recomputing.

        Returns:
            List of 1536-dim embedding vectors, order-preserving.
        """
        if not texts:
            return []

        from azure_clients.redis_client import (
            get_embeddings_batch_cached,
            set_embeddings_batch_cached,
        )

        cached = get_embeddings_batch_cached(texts)
        missing_idx = [i for i, emb in enumerate(cached) if emb is None]

        if not missing_idx:
            logger.debug("OpenAIClient.embed_batch: %d texts, all cached", len(texts))
            return [emb for emb in cached]  # type: ignore[misc]

        missing_texts = [texts[i] for i in missing_idx]
        response = self._client.embeddings.create(
            model=self._embedding_deployment,
            input=missing_texts,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        sorted_data = sorted(response.data, key=lambda d: d.index)
        fresh = [d.embedding for d in sorted_data]

        for i, embedding in zip(missing_idx, fresh):
            cached[i] = embedding
        set_embeddings_batch_cached(missing_texts, fresh)

        logger.debug(
            "OpenAIClient.embed_batch: %d texts (%d cached, %d embedded)",
            len(texts), len(texts) - len(missing_idx), len(missing_idx),
        )
        return [emb for emb in cached]  # type: ignore[misc]


# Module-level singleton
openai_client = OpenAIClient()