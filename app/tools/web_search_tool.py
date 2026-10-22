import logging

from langchain.tools import tool
from langchain_tavily import TavilySearch

from app.config import (
    TAVILY_API_KEY
)

logger = logging.getLogger(__name__)

def _tavily_results(raw) -> list[dict]:
    """Chuẩn hoá output TavilySearch về list[dict]. langchain_tavily.TavilySearch trả dict
    {"query":..., "results": [{"title","url","content","score"}, ...]}; một số phiên bản trả
    thẳng list. Trả [] nếu không nhận dạng được."""
    if isinstance(raw, dict):
        results = raw.get("results")
        return results if isinstance(results, list) else []
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []

@tool
def search_webs(query: str) -> str:
    """Tìm thông tin văn bản trên web qua Tavily. Dùng cho câu cần dữ liệu ngoài/cập nhật
    (tin tức, giá, số liệu...). Trả về chuỗi: danh sách kết quả đã format kèm References.

    LUÔN trả về một chuỗi tự mô tả — kể cả khi không có kết quả hoặc lỗi — để model biết rõ
    trạng thái và KHÔNG tìm lại cùng một truy vấn."""
    try:
        search_tool = TavilySearch(max_results=5, tavily_api_key=TAVILY_API_KEY)
        results = _tavily_results(search_tool.invoke({"query": query}))
        if not results:
            logger.warning("tool search_webs: không có kết quả / định dạng lạ")
            return "Không tìm thấy kết quả web cho truy vấn này."

        formatted, references = "", []
        for i, result in enumerate(results, start=1):
            title = result.get("title") or "No Title"
            url = result.get("url") or ""
            content = result.get("content") or ""
            formatted += f"{i}. {title}\n{content} [^{i}]\n\n"
            references.append(f"[^{i}]: [{title}]({url})")

        text = formatted + "\n**References:**\n" + "\n".join(references)
        return text
    except Exception as exc:  # noqa: BLE001 - không bao giờ làm hỏng luồng
        logger.warning("graph.search lỗi: %s", exc)
        return "Tìm kiếm web thất bại do lỗi kỹ thuật. Không thể lấy kết quả cho truy vấn này."