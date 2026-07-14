"""
azure_clients/openai_client.py

Azure OpenAI client for:
  - Chat completions (gpt-5-mini) — streaming and non-streaming
  - Embeddings (text-embedding-3-small)

Secrets sourced from Key Vault via the kv singleton.
"""

import logging
from collections.abc import Iterator
from typing import Optional

from openai import AzureOpenAI

from azure_clients.key_vault_client import kv

logger = logging.getLogger(__name__)

# Embedding dimensionality — must match the AI Search index schema
EMBEDDING_DIMENSIONS = 1536


class OpenAIClient:
    """
    Thin wrapper around AzureOpenAI for chat and embedding calls.
    One instance shared across all agents via the module-level singleton.
    """

    def __init__(self):
        endpoint = kv.get_secret("AZURE-OPENAI-ENDPOINT")
        api_key = kv.get_secret("AZURE-OPENAI-KEY")
        self._chat_deployment = kv.get_secret("AZURE-OPENAI-DEPLOYMENT-NAME")
        self._embedding_deployment = "text-embedding-3-small"

        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-08-01-preview",
        )
        logger.info(
            "OpenAIClient: connected — chat=%s, embedding=%s",
            self._chat_deployment,
            self._embedding_deployment,
        )

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Non-streaming chat completion.

        Args:
            messages:     OpenAI message list [{"role": ..., "content": ...}, ...]
            tools:        Tool schemas for function calling (optional).
            tool_choice:  "auto" | "none" | specific tool name (optional).
            temperature:  Sampling temperature. Default 0.0 for deterministic agent reasoning.
            max_tokens:   Max tokens in the response.

        Returns:
            The full ChatCompletion response object (caller reads .choices[0].message).
        """
        kwargs = dict(
            model=self._chat_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
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

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """
        Streaming chat completion — yields text deltas as they arrive.
        Used by the frontend's live agent progress view.

        Args:
            messages:    OpenAI message list.
            temperature: Sampling temperature.
            max_tokens:  Max tokens in the response.

        Yields:
            Text delta strings as they stream from the API.
        """
        stream = self._client.chat.completions.create(
            model=self._chat_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
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

        Args:
            text: Input text to embed.

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
        Azure OpenAI supports up to 2048 inputs per request.

        Args:
            texts: List of strings to embed.

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
        # API guarantees order is preserved but sort by index to be safe
        sorted_data = sorted(response.data, key=lambda d: d.index)
        logger.debug(
            "OpenAIClient.embed_batch: %d texts embedded", len(sorted_data)
        )
        return [d.embedding for d in sorted_data]


# Module-level singleton
openai_client = OpenAIClient()