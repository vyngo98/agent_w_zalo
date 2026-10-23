"""Langfuse tracing cho agent (tùy chọn, chịu lỗi).

`get_langfuse_handler()` trả về một LangChain `CallbackHandler` của Langfuse để truyền vào
`agent.invoke(config={"callbacks": [...]})`. Nếu tracing bị tắt, thiếu key, hoặc chưa cài gói
`langfuse` -> trả None (no-op) để KHÔNG bao giờ làm hỏng pipeline chat.

Hỗ trợ cả langfuse v3 (`langfuse.langchain`) và v2 (`langfuse.callback`). Handler được cache
(singleton) và tái dùng cho mọi request; thuộc tính theo từng request (session_id, user_id,
tags) truyền qua `metadata` của config ở `app/api.py`.
"""

import logging

from app.config import (
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

logger = logging.getLogger(__name__)

_HANDLER = None
_INITIALIZED = False


def get_langfuse_handler():
    """CallbackHandler của Langfuse (cache) hoặc None nếu tắt/thiếu cấu hình/chưa cài gói."""
    global _HANDLER, _INITIALIZED
    if _INITIALIZED:
        return _HANDLER
    _INITIALIZED = True

    if not LANGFUSE_ENABLED:
        logger.info("Langfuse: đã tắt (LANGFUSE_ENABLED=0).")
        return None
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        logger.info("Langfuse: thiếu LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY -> bỏ qua tracing.")
        return None

    # langfuse v3: đọc key/host từ biến môi trường LANGFUSE_* (đã nạp qua .env).
    try:
        from langfuse.langchain import CallbackHandler

        _HANDLER = CallbackHandler()
        logger.info("Langfuse: bật tracing (v3), host=%s", LANGFUSE_HOST)
        return _HANDLER
    except Exception as exc_v3:  # noqa: BLE001 - thử v2 rồi mới bỏ qua
        try:
            from langfuse.callback import CallbackHandler  # langfuse v2

            _HANDLER = CallbackHandler(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            logger.info("Langfuse: bật tracing (v2), host=%s", LANGFUSE_HOST)
            return _HANDLER
        except Exception as exc_v2:  # noqa: BLE001 - không chặn pipeline
            logger.warning(
                "Langfuse: không khởi tạo được, bỏ qua tracing (v3: %s | v2: %s). "
                "Cài `pip install langfuse` nếu muốn trace.",
                exc_v3, exc_v2,
            )
            _HANDLER = None
            return None
