"""Resilient LLM caller: retry with exponential backoff + jitter, a fallback
model chain, and a cosmetic token streamer.

The answer LLM is reached through `get_answer_llm()`, which returns a proxy whose
`.invoke(messages)` retries the primary model on transient errors and then falls
through `LLM_FALLBACK_MODELS` before giving up. The proxy is a drop-in for the
previous `ChatOllama` singleton — it exposes the same `.content` /
`.usage_metadata` / `.response_metadata` on its result.

Synchronous on purpose: the whole backend is sync (`def` endpoints running in a
threadpool), so this avoids introducing asyncio.
"""

import logging
import random
import time

from app.config import (
    # LLM_FALLBACK_MODELS,
    # LLM_MAX_RETRIES,
    LLM_MODEL,
    # LLM_RETRY_BASE,
    # LLM_RETRY_MAX_BACKOFF,
    # LLM_TEMPERATURE,
    # OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

LLM_UNAVAILABLE_MESSAGE = (
    "Xin lỗi, hệ thống tạm thời không thể xử lý yêu cầu của bạn. "
    "Vui lòng thử lại sau giây lát.\n"
    "(I apologize, but I am temporarily unable to process your request. "
    "Please try again in a moment.)"
)


class AllModelsFailedError(Exception):
    """Raised when every model in the fallback chain has been exhausted."""


# Per-model client cache (ChatOllama is just an HTTP-client wrapper).
_MODELS: dict = {}


# def _get_chat_model(model_name: str):
#     if model_name not in _MODELS:
#         from langchain_ollama import ChatOllama
#
#         _MODELS[model_name] = ChatOllama(
#             model=model_name,
#             temperature=LLM_TEMPERATURE,
#             base_url=OLLAMA_BASE_URL,
#         )
#     return _MODELS[model_name]
#
#
# def call_llm_with_retry(messages, model_name: str, max_retries: int = LLM_MAX_RETRIES):
#     """Invoke one model, retrying transient failures with capped exponential
#     backoff + jitter. Re-raises the last error once retries are exhausted."""
#     for attempt in range(max_retries + 1):
#         try:
#             return _get_chat_model(model_name).invoke(messages)
#         except Exception as exc:  # noqa: BLE001 - any LLM/transport error is retryable
#             if attempt < max_retries:
#                 backoff = min(
#                     LLM_RETRY_BASE ** attempt + random.uniform(0, 1),
#                     LLM_RETRY_MAX_BACKOFF,
#                 )
#                 logger.warning(
#                     "LLM '%s' lỗi (attempt %d/%d): %s -> retry sau %.2fs",
#                     model_name, attempt + 1, max_retries + 1, exc, backoff,
#                 )
#                 time.sleep(backoff)
#             else:
#                 logger.warning("LLM '%s' hết retry: %s", model_name, exc)
#                 raise
#
#
# def call_with_fallback(messages, preferred_model: str = None):
#     """Try the model chain (primary + fallbacks) in order; return the first
#     success. Raises AllModelsFailedError if every model fails."""
#     chain = [LLM_MODEL] + list(LLM_FALLBACK_MODELS)
#     if preferred_model and preferred_model in chain:
#         chain.remove(preferred_model)
#         chain.insert(0, preferred_model)
#
#     last_error = None
#     for model_name in chain:
#         try:
#             return call_llm_with_retry(messages, model_name)
#         except Exception as exc:  # noqa: BLE001 - move to next model in the chain
#             last_error = exc
#             continue
#
#     raise AllModelsFailedError(str(last_error))
#
#
# class _RetryingChatModel:
#     """Drop-in for the old answer-LLM singleton: `.invoke(messages)` routes
#     through retry + fallback and returns the underlying chat result."""
#
#     def invoke(self, messages):
#         return call_with_fallback(messages)
#
#
# _ANSWER_LLM = None
#
#
# def get_answer_llm() -> _RetryingChatModel:
#     global _ANSWER_LLM
#     if _ANSWER_LLM is None:
#         _ANSWER_LLM = _RetryingChatModel()
#     return _ANSWER_LLM
#
#
# def stream_tokens(text: str, min_delay: float = 0.02, max_delay: float = 0.08):
#     """Yield an already-complete answer word-by-word (cosmetic streaming)."""
#     words = text.split()
#     for i, word in enumerate(words):
#         yield word if i == 0 else " " + word
#         time.sleep(random.uniform(min_delay, max_delay))
