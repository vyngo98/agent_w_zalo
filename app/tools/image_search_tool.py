"""Tìm ẢNH qua DuckDuckGo (gói `ddgs`) cho nhánh `image` của graph v2.

Deterministic, MIỄN PHÍ, không cần API key (khác Tavily). `ddgs` là thư viện unofficial nên
API/kết quả có thể đổi và có rate-limit/ToS — MỌI lời gọi bọc try/except trả [] để không bao
giờ làm hỏng luồng (mirror khả năng chịu lỗi của `search_node` trong app/agent/graph.py).

- `ddg_image_search(query)` — hàm thuần, trả list[dict] {title, image, thumbnail, url}. Node
  trong graph gọi thẳng hàm này.
- `search_images(query)` — bản `@tool` để default tool-calling agent / bước Zalo dùng lại.
"""

import logging

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from app.config import (
    IMAGE_SEARCH_MAX_RESULTS,
    IMAGE_SEARCH_REGION,
    IMAGE_SEARCH_SAFESEARCH,
)

logger = logging.getLogger(__name__)


def ddg_image_search(query: str, max_results: int = None) -> list[dict]:
    """Tìm ảnh trên DuckDuckGo. Trả list[dict] chuẩn hoá {title, image, thumbnail, url}.

    `image` = URL ảnh gốc, `thumbnail` = URL ảnh thu nhỏ, `url` = trang nguồn. Lỗi/định dạng
    lạ -> [] (không bao giờ raise)."""
    n = max_results or IMAGE_SEARCH_MAX_RESULTS
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            # Gọi positional cho `query` vì các phiên bản ddgs đặt tên tham số đầu khác nhau
            # (query/keywords). region/safesearch/max_results giữ ổn định.
            raw = ddgs.images(
                query,
                region=IMAGE_SEARCH_REGION,
                safesearch=IMAGE_SEARCH_SAFESEARCH,
                max_results=n,
            )
        rows = []
        for r in raw or []:
            if not isinstance(r, dict):
                continue
            image = r.get("image") or r.get("url") or ""
            if not image:
                continue
            rows.append({
                "title": r.get("title") or "",
                "image": image,
                "thumbnail": r.get("thumbnail") or image,
                "url": r.get("url") or r.get("source") or image,
            })
        return rows
    except Exception as exc:  # noqa: BLE001 - không bao giờ làm hỏng luồng
        logger.warning("ddg_image_search lỗi: %s", exc)
        return []


@tool
def search_images(query: str, config: RunnableConfig) -> str:
    """Tìm hình ảnh/ảnh trên web (DuckDuckGo) cho một chủ đề/thực thể.

    Dùng khi người dùng muốn XEM hình ảnh, ảnh, logo, biểu đồ, hình minh họa của một thực thể
    (công ty, sản phẩm, địa điểm...). Ảnh được GỬI TRỰC TIẾP cho người dùng (đính kèm), nên
    chỉ cần trả lời ngắn gọn bằng lời — KHÔNG dán lại URL ảnh."""
    rows = ddg_image_search(query)

    # Ghi kết quả vào `images_sink` (do app/api.py truyền qua config.configurable) để tầng
    # kênh (Zalo sendPhoto / UI) gửi ảnh thật. `config` được inject tự động, không lộ cho model.
    sink = (config.get("configurable") or {}).get("images_sink")
    if sink is not None and rows:
        sink.extend(rows)

    if not rows:
        return "Không tìm thấy hình ảnh phù hợp."

    titles = ", ".join(r["title"] or "(ảnh)" for r in rows)
    return (
        f"Đã tìm thấy {len(rows)} hình ảnh và đính kèm trực tiếp cho người dùng: {titles}. "
        "Hãy trả lời ngắn gọn, KHÔNG dán lại URL/đường link ảnh."
    )
