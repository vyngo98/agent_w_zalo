"""
4 loại memory cho LangChain v1 agent (create_agent + middleware)

- Working   : trạng thái hội thoại hiện tại  -> InMemorySaver (checkpointer) + SummarizationMiddleware
- Semantic  : sự thật/sở thích về user        -> Store, namespace (user_id, "semantic")
- Episodic  : các "tình huống" đã xảy ra      -> Store, namespace (user_id, "episodic")
- Procedural: quy tắc/cách làm việc tốt nhất  -> Store, namespace (user_id, "procedural")

Long-term memory (3 loại sau) được:
  * ĐỌC  trong wrap_model_call  -> chèn vào system prompt trước mỗi lần gọi model
  * GHI  trong after_agent      -> LLM trích xuất sau khi agent trả lời xong
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------
# Context (truyền user_id mỗi lần invoke)
# ----------------------------------------------------------------------------
@dataclass
class Context:
    user_id: str = "default"


# ----------------------------------------------------------------------------
# Schema để LLM trích xuất memory sau mỗi lượt
# ----------------------------------------------------------------------------
class Episode(BaseModel):
    situation: str = Field(description="Người dùng hỏi/cần gì (ngắn gọn)")
    actions: str = Field(description="Agent đã làm gì (tool nào, cách tiếp cận)")
    outcome: str = Field(description="Kết quả / phản hồi của người dùng")


class MemoryUpdate(BaseModel):
    semantic_facts: list[str] = Field(
        default_factory=list,
        description="Sự thật BỀN VỮNG về user: tên, nghề, sở thích, ngôn ngữ, bối cảnh... "
        "Mỗi mục là 1 câu độc lập. Bỏ qua thông tin tạm thời.",
    )
    episode: Episode | None = Field(
        default=None,
        description="Tóm tắt lượt này nếu đáng nhớ (bài học, kết quả rõ ràng). "
        "None nếu chỉ là chào hỏi/xã giao.",
    )
    procedural_rules: list[str] = Field(
        default_factory=list,
        description="Quy tắc/cách làm việc mà user muốn agent tuân theo từ giờ "
        "(vd: 'Luôn trả lời bằng tiếng Việt', 'Luôn kèm nguồn'). "
        "Chỉ thêm khi user chỉnh sửa/yêu cầu rõ ràng.",
    )


EXTRACT_PROMPT = """Bạn là bộ phận quản lý bộ nhớ của một AI agent.
Đọc đoạn hội thoại gần nhất và trích xuất memory cần lưu.

Memory đã có (KHÔNG lặp lại):
{existing}

Hội thoại:
{conversation}
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return " ".join(
        p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
    )


def _last_turn(messages) -> list:
    """Các message từ HumanMessage cuối cùng trở đi."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return []


# ----------------------------------------------------------------------------
# Middleware chính
# ----------------------------------------------------------------------------
class LongTermMemoryMiddleware(AgentMiddleware):
    """Semantic + Episodic + Procedural memory."""

    def __init__(
        self,
        extractor_model,
        top_k: int = 5,
        dedupe_threshold: float = 0.85,
        seed_rules: list[str] | None = None,
    ):
        super().__init__()
        self.extractor = extractor_model.with_structured_output(MemoryUpdate,  method="function_calling")
        self.top_k = top_k
        self.dedupe_threshold = dedupe_threshold
        self.seed_rules = seed_rules or []

    # ---- namespaces -------------------------------------------------------
    @staticmethod
    def _ns(user_id: str, kind: str) -> tuple[str, str]:
        return (user_id, kind)

    @staticmethod
    def _user_id(runtime) -> str:
        ctx = getattr(runtime, "context", None)
        return getattr(ctx, "user_id", None) or "default"

    # ---- ĐỌC: chèn memory vào system prompt -------------------------------
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        store = request.runtime.store
        if store is None:
            return handler(request)

        uid = self._user_id(request.runtime)
        humans = [m for m in request.messages if isinstance(m, HumanMessage)]
        query = _text(humans[-1]) if humans else ""

        # Semantic + Episodic: tìm theo độ tương đồng với câu hỏi hiện tại
        facts = store.search(self._ns(uid, "semantic"), query=query, limit=self.top_k)
        episodes = store.search(self._ns(uid, "episodic"), query=query, limit=3)
        # Procedural: nạp toàn bộ (tập quy tắc nhỏ, luôn cần áp dụng)
        rules = store.search(self._ns(uid, "procedural"), limit=20)

        rule_texts = self.seed_rules + [r.value["text"] for r in rules]

        blocks = []
        if facts:
            blocks.append(
                "## Điều đã biết về người dùng (semantic memory)\n"
                + "\n".join(f"- {f.value['text']}" for f in facts)
            )
        if episodes:
            blocks.append(
                "## Các tình huống tương tự trước đây (episodic memory)\n"
                + "\n".join(f"- {e.value['text']}" for e in episodes)
            )
        if rule_texts:
            blocks.append(
                "## Quy tắc làm việc cần tuân thủ (procedural memory)\n"
                + "\n".join(f"- {t}" for t in rule_texts)
            )

        if not blocks:
            return handler(request)

        new_prompt = (request.system_prompt or "") + "\n\n" + "\n\n".join(blocks)
        return handler(request.override(system_prompt=new_prompt))

    # ---- GHI: trích xuất & lưu memory sau khi agent xong -------------------
    def after_agent(self, state, runtime) -> dict | None:
        store = runtime.store
        if store is None:
            return None

        turn = _last_turn(state["messages"])
        if not turn:
            return None

        uid = self._user_id(runtime)
        conversation = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Agent' if isinstance(m, AIMessage) else 'Tool'}: "
            f"{_text(m)[:800]}"
            for m in turn
            if _text(m)
        )

        # cho extractor biết memory đã có để tránh trùng
        existing = []
        for kind in ("semantic", "procedural"):
            for it in store.search(self._ns(uid, kind), limit=20):
                existing.append(f"[{kind}] {it.value['text']}")

        update: MemoryUpdate = self.extractor.invoke(
            EXTRACT_PROMPT.format(
                existing="\n".join(existing) or "(trống)", conversation=conversation
            )
        )

        for fact in update.semantic_facts:
            self._put_dedup(store, self._ns(uid, "semantic"), fact)

        for rule in update.procedural_rules:
            self._put_dedup(store, self._ns(uid, "procedural"), rule)

        if update.episode:
            e = update.episode
            text = f"Tình huống: {e.situation} | Hành động: {e.actions} | Kết quả: {e.outcome}"
            store.put(
                self._ns(uid, "episodic"),
                str(uuid.uuid4()),
                {"text": text, **e.model_dump()},
            )
        return None

    def _put_dedup(self, store, namespace, text: str) -> None:
        """Chỉ lưu nếu chưa có mục nào quá giống (dựa trên embedding score)."""
        similar = store.search(namespace, query=text, limit=1)
        if similar and (similar[0].score or 0) >= self.dedupe_threshold:
            return
        store.put(namespace, str(uuid.uuid4()), {"text": text})




# ----------------------------------------------------------------------------
# Cách dùng
# ----------------------------------------------------------------------------
# agent = build_agent(model, [search_webs, search_images], SYSTEM_PROMPT)
# cfg = {"configurable": {"thread_id": "chat-1"}}           # working memory theo thread
# agent.invoke({"messages": [("user", "Mình tên Nam, thích mèo, trả lời tiếng Việt nhé")]},
#              cfg, context=Context(user_id="nam"))          # long-term memory theo user
# # Thread MỚI, cùng user -> vẫn nhớ semantic/episodic/procedural:
# agent.invoke({"messages": [("user", "Gợi ý ảnh con vật mình thích")]},
#              {"configurable": {"thread_id": "chat-2"}}, context=Context(user_id="nam"))