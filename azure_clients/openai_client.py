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
from collections.abc import Iterator
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
    # Chat completions (async) — used by async agent nodes (Phase 2)
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_completion_tokens: Optional[int] = None,
    ) -> object:
        """
        Async non-streaming chat completion.
        Mirrors chat() exactly — use in async agent nodes with await.

        Args:
            messages:               OpenAI message list.
            tools:                  Tool schemas for function calling (optional).
            tool_choice:            "auto" | "none" | specific tool name (optional).
            max_tokens:             Alias for max_completion_tokens (kept for compat).
            max_completion_tokens:  Max tokens. Must be >= 4096 for gpt-5-mini.

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

        response = await self._async_client.chat.completions.create(**kwargs)
        logger.debug(
            "OpenAIClient.achat: %d prompt + %d completion tokens",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return response

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
          "primary"  → gpt-5.4-mini (complex reasoning, comparison, report)
          "standard" → gpt-5-mini   (simple fact lookups)

        Args:
            messages:    OpenAI message list.
            model_tier:  "primary" | "standard". Defaults to "primary".
            All other args mirror achat().

        Returns:
            The full ChatCompletion response object.
        """
        deployment = (
            self._standard_deployment
            if model_tier == "standard"
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

    # ------------------------------------------------------------------
    # Streaming (sync only — streaming stays sync for Phase 1/2)
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> Iterator[str]:
        """
        Streaming chat completion — yields text deltas as they arrive.

        Args:
            messages:    OpenAI message list.
            temperature: Ignored — gpt-5-mini does not support this parameter.
            max_tokens:  Max tokens. Enforced minimum of 4096.

        Yields:
            Text delta strings as they stream from the API.
        """
        limit = max(max_tokens, _MIN_SAFE_TOKENS)
        stream = self._client.chat.completions.create(
            model=self._chat_deployment,
            messages=messages,
            max_completion_tokens=limit,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """
        Embed a single string.

        Returns:
            1536-dim embedding vector.
        """
        response = self._client.embeddings.create(
            model=self._embedding_deployment,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings in one API call.

        Returns:
            List of 1536-dim embedding vectors, order-preserving.
        """
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._embedding_deployment,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        sorted_data = sorted(response.data, key=lambda d: d.index)
        logger.debug(
            "OpenAIClient.embed_batch: %d texts embedded", len(sorted_data)
        )
        return [d.embedding for d in sorted_data]


# Module-level singleton
openai_client = OpenAIClient()