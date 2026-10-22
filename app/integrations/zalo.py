import json
import logging
import os
# import sys

import httpx

from app.config import (
    ZALO_API_BASE,
    ZALO_BOT_TOKEN,
    ZALO_CHATS_PATH,
    ZALO_POLL_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Giới hạn ký tự của sendMessage.text / sendPhoto.caption theo docs Zalo.
ZALO_TEXT_LIMIT = 2000

UNSUPPORTED_MESSAGE = (
    "Xin lỗi, hiện mình chỉ trả lời được câu hỏi dạng văn bản. "
    "Bạn vui lòng gõ câu hỏi bằng chữ nhé."
)
NO_DOCUMENT_MESSAGE = (
    "Chưa có báo cáo tài chính nào được nạp. "
    "Vui lòng upload file PDF qua giao diện web trước khi hỏi."
)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class ZaloBotClient:
    """Client tối giản cho Bot API Zalo. Mỗi call bọc lỗi -> trả None/[] + log, KHÔNG ném ra
    ngoài để không làm chết vòng polling."""

    def __init__(self, token: str = None, base: str = None, timeout: float = 60.0):
        self.token = token or ZALO_BOT_TOKEN
        self.base = (base or ZALO_API_BASE).rstrip("/")
        self.timeout = timeout

    def _endpoint(self, method: str) -> str:
        return f"{self.base}/bot{self.token}/{method}"

    def _post(self, method: str, payload: dict = None, timeout: float = None) -> dict | None:
        """POST một method. Trả `result` nếu ok, ngược lại None (đã log)."""
        if not self.token:
            logger.warning("zalo._post(%s): thiếu ZALO_BOT_TOKEN", method)
            return None
        try:
            resp = httpx.post(
                self._endpoint(method),
                json=payload or {},
                timeout=timeout or self.timeout,
            )
            data = resp.json()
            if not data.get("ok"):
                logger.warning("zalo.%s ok=false: %s", method, data)
                return None
            return data.get("result")
        except Exception as exc:  # noqa: BLE001 - không bao giờ ném ra làm chết worker
            logger.warning("zalo.%s lỗi: %s", method, exc)
            return None

    def get_me(self) -> dict | None:
        return self._post("getMe")

    def send_message(self, chat_id: str, text: str) -> dict | None:
        return self._post("sendMessage", {"chat_id": str(chat_id), "text": text})

    def send_photo(self, chat_id: str, photo: str, caption: str = None) -> dict | None:
        payload = {"chat_id": str(chat_id), "photo": photo}
        if caption:
            payload["caption"] = caption[:ZALO_TEXT_LIMIT]
        return self._post("sendPhoto", payload)

    def delete_webhook(self) -> dict | None:
        return self._post("deleteWebhook")

    def get_updates(self, timeout: int = None) -> list[dict]:
        """Long-polling. Trả list update đã chuẩn hoá (docs không nêu offset; một số bản trả
        tối đa 1 update/lần hoặc một object đơn -> chấp nhận cả list lẫn dict)."""
        t = ZALO_POLL_TIMEOUT if timeout is None else timeout
        # timeout HTTP phải lớn hơn timeout long-poll để không ngắt giữa chừng.
        result = self._post("getUpdates", {"timeout": t}, timeout=t + 15)
        if result is None:
            return []
        if isinstance(result, list):
            return [u for u in result if isinstance(u, dict)]
        if isinstance(result, dict):
            # Có thể là {"updates": [...]} hoặc chính một update đơn.
            inner = result.get("updates")
            if isinstance(inner, list):
                return [u for u in inner if isinstance(u, dict)]
            return [result]
        return []


# --------------------------------------------------------------------------- #
# Định dạng trả lời
# --------------------------------------------------------------------------- #

_CONTENT_TYPE_VI = {"table": "bảng", "text": "văn bản"}


def format_reply(message: dict) -> str:
    """Ghép nội dung trả lời + dòng trích dẫn trang từ `message["sources"]`."""
    content = (message or {}).get("content") or ""
    sources = (message or {}).get("sources") or []
    if not sources:
        return content
    parts = []
    for s in sources:
        page = s.get("page", "?")
        kind = _CONTENT_TYPE_VI.get(s.get("content_type"), s.get("content_type") or "")
        parts.append(f"trang {page}" + (f" ({kind})" if kind else ""))
    return content + "\n\n📄 Nguồn: " + ", ".join(parts)


def chunk_text(text: str, limit: int = ZALO_TEXT_LIMIT) -> list[str]:
    """Chia `text` thành các phần ≤ limit ký tự, ưu tiên cắt ở ranh giới dòng."""
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    chunks, buf = [], ""
    for line in text.split("\n"):
        # Dòng đơn dài hơn limit -> cắt cứng theo độ dài.
        while len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = line if not buf else buf + "\n" + line
        if len(candidate) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------- #
# Map chat_id Zalo -> session (data/zalo_chats.json)
# --------------------------------------------------------------------------- #

def _read_chat_map() -> dict:
    if not os.path.exists(ZALO_CHATS_PATH):
        return {}
    try:
        with open(ZALO_CHATS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("zalo: đọc %s lỗi: %s", ZALO_CHATS_PATH, exc)
        return {}


def _write_chat_map(mapping: dict) -> None:
    os.makedirs(os.path.dirname(ZALO_CHATS_PATH), exist_ok=True)
    with open(ZALO_CHATS_PATH, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def get_or_create_zalo_session(chat_id: str, display_name: str = None) -> dict | None:
    """Tìm/khởi tạo session cho một chat Zalo. Trả session dict, hoặc None nếu chưa có tài liệu
    `ready` để gắn. Session gắn **tài liệu mới nhất** tại thời điểm tạo (nhất quán với web UI)."""
    from app import storage

    chat_id = str(chat_id)
    mapping = _read_chat_map()

    session_id = mapping.get(chat_id)
    if session_id:
        session = storage.get_session(session_id)
        if session:
            return session
        # Session đã bị xóa -> bỏ mapping cũ, tạo lại bên dưới.
        mapping.pop(chat_id, None)


    title = f"Zalo • {display_name}" if display_name else "Zalo"
    session = storage.create_session(title=title)

    mapping[chat_id] = session["id"]
    _write_chat_map(mapping)
    return session
