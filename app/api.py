import logging
import re
import time
import uuid

from pydantic import BaseModel

from app import storage
from app.agent.lead_agent import create_custom_agent
from app.config import LLM_MODEL, PROMPT_VERSION
from app.guardrails import (
    INPUT_BLOCKED_MESSAGE,
    OUTPUT_BLOCKED_MESSAGE,
    check_input_guardrails,
    check_output_guardrails,
)

from app.integrations.zalo import (
    NO_DOCUMENT_MESSAGE,
    UNSUPPORTED_MESSAGE,
    ZaloBotClient,
    chunk_text,
    format_reply,
    get_or_create_zalo_session,
)
from app.llm import LLM_UNAVAILABLE_MESSAGE
from app.observability import (
    RequestLog,
    estimate_cost,
    now_iso,
    write_request_log,
)
from app.tracing import get_langfuse_handler
# from app.semantic_cache import get_semantic_cache

logger = logging.getLogger(__name__)

agent = create_custom_agent()

# Zalo Bot client (kênh Zalo). Tái dùng cho cả worker polling và webhook (giai đoạn 2).
zalo_client = ZaloBotClient()

class MessageRequest(BaseModel):
    content: str


def clean_markdown(text):
    # Xóa các ký tự in đậm/in nghiêng như **, *, __, _
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # Xóa các ký tự tiêu đề #, ##, ### ở đầu dòng
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Xóa định dạng khối mã code ```
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    return text


def _handle_message(session_id: str, session: dict, req: MessageRequest) -> dict:
    """Run the full pipeline for one user message (guardrails, cache, agent,
    logging, persistence) and return the stored assistant message dict. Never
    raises on LLM failure — returns a graceful apology turn instead."""
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    # Semantic cache is consulted/populated only for standalone questions (the
    # first turn, no prior history) to avoid wrong hits on context-dependent
    # follow-ups. Captured before any message is appended this turn.
    # document_id = session["document_id"]
    # is_first_turn = len(session["messages"]) == 0

    # Input guardrail runs before anything is stored or the agent is invoked.
    in_res = check_input_guardrails(req.content)
    if not in_res.passed:
        # Store the original message (audit trail) + a canned refusal, log the
        # block, and return without calling the agent.
        storage.append_message(session_id, "user", req.content)
        write_request_log(RequestLog(
            request_id=request_id,
            user_id=session_id,
            timestamp=now_iso(),
            prompt_template="blocked_input",
            prompt_version=PROMPT_VERSION,
            model=LLM_MODEL,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            cache_hit=False,
            guardrail_input_pass=False,
            guardrail_output_pass=True,
            cost_usd=0.0,
            error=None,
        ))
        return storage.append_message(session_id, "assistant", INPUT_BLOCKED_MESSAGE)

    # Passed: use the redacted text (if any PII was found) so PII is neither
    # persisted on disk nor sent to the LLM.
    content = in_res.modified_text or req.content
    storage.append_message(session_id, "user", content)

    # # Semantic cache lookup (standalone questions only): a hit returns the stored
    # # answer without invoking the agent.
    # if is_first_turn and document_id:
    #     hit = get_semantic_cache().get(content, document_id)
    #     if hit:
    #         write_request_log(RequestLog(
    #             request_id=request_id,
    #             user_id=session_id,
    #             timestamp=now_iso(),
    #             prompt_template="cache",
    #             prompt_version=PROMPT_VERSION,
    #             model=LLM_MODEL,
    #             input_tokens=0,
    #             output_tokens=0,
    #             latency_ms=(time.perf_counter() - t0) * 1000.0,
    #             cache_hit=True,
    #             guardrail_input_pass=True,
    #             guardrail_output_pass=True,
    #             cost_usd=0.0,
    #             error=None,
    #         ))
    #         return storage.append_message(
    #             session_id, "assistant", hit["response"],
    #             sources=hit.get("sources") or None,
    #         )

    # Refresh to include the just-added user message; pass the full history so
    # follow-up questions have conversational context.
    session = storage.get_session(session_id)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in session["messages"]
    ]
    # Collect retrieval citations from the RAG tool (via a per-request sink)
    # so the UI can show which pages/tables the answer drew from. `log_sink`
    # carries observability data (e.g. prompt template) back from the graph.
    sources_sink: list = []
    images_sink: list = []
    log_sink: dict = {}

    # Langfuse tracing (tùy chọn). None nếu tắt/chưa cài -> callbacks rỗng, chạy bình thường.
    # metadata: gom trace theo session Zalo và gắn user/tags cho dễ lọc trên Langfuse.
    callbacks = []
    lf_handler = get_langfuse_handler()
    if lf_handler is not None:
        callbacks.append(lf_handler)

    try:
        result = agent.invoke(
            {"messages": history},
            config={
                "callbacks": callbacks,
                "run_name": "zalo_agent",
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": session_id,
                    "langfuse_tags": ["zalo-agent", f"prompt:{PROMPT_VERSION}"],
                },
                "configurable": {
                    "sources_sink": sources_sink,
                    "images_sink": images_sink,
                    "log_sink": log_sink,
                },
            },
        )
    except Exception as e:  # noqa: BLE001 - log the failure, degrade gracefully
        latency_ms = (time.perf_counter() - t0) * 1000.0
        write_request_log(RequestLog(
            request_id=request_id,
            user_id=session_id,
            timestamp=now_iso(),
            prompt_template=log_sink.get("prompt_template", "agent"),
            prompt_version=PROMPT_VERSION,
            model=LLM_MODEL,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cache_hit=False,
            guardrail_input_pass=in_res.passed,
            guardrail_output_pass=True,
            cost_usd=0.0,
            error=str(e),
        ))
        logger.error("Lỗi model agent!")
        # All retries + fallback models exhausted -> apology turn (HTTP 200).
        return storage.append_message(session_id, "assistant", LLM_UNAVAILABLE_MESSAGE)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    final = result["messages"][-1]
    answer = final.content

    # Output guardrail: block unsafe content before it reaches the user.
    out_res = check_output_guardrails(answer)
    if not out_res.passed:
        answer = OUTPUT_BLOCKED_MESSAGE
        sources_sink = []  # drop citations for a blocked answer
        images_sink = []   # drop images for a blocked answer

    answer = clean_markdown(answer)

    usage = getattr(final, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    write_request_log(RequestLog(
        request_id=request_id,
        user_id=session_id,
        timestamp=now_iso(),
        prompt_template=log_sink.get("prompt_template", "agent"),
        prompt_version=PROMPT_VERSION,
        model=LLM_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cache_hit=False,
        guardrail_input_pass=in_res.passed,
        guardrail_output_pass=out_res.passed,
        cost_usd=estimate_cost(LLM_MODEL, input_tokens, output_tokens),
        error=None,
    ))

    # Dedup by page, preserving first-seen order.
    seen = set()
    sources = []
    for s in sources_sink:
        if s["page"] not in seen:
            seen.add(s["page"])
            sources.append(s)

    # Dedup image results by URL, preserving first-seen order.
    seen_img = set()
    images = []
    for im in images_sink:
        if im.get("image") and im["image"] not in seen_img:
            seen_img.add(im["image"])
            images.append(im)

    # Populate the semantic cache for standalone questions with a safe answer.
    # Never cache live web-search / image-search answers (routes search/image) — they go stale
    # and the image side-channel is not stored in the cache.
    # if (
    #     is_first_turn
    #     and document_id
    #     and out_res.passed
    #     and log_sink.get("prompt_template") not in ("generate_search", "generate_image")
    # ):
    #     get_semantic_cache().put(content, document_id, answer, sources or None)

    return storage.append_message(
        session_id, "assistant", answer,
        sources=sources or None, images=images or None,
    )


def _process_zalo_update(update: dict) -> None:
    """Điều phối MỘT update Zalo: map chat -> session, chạy pipeline RAG (`_handle_message`),
    rồi gửi trả lời (text có trích dẫn + ảnh). Dùng chung cho worker polling (giai đoạn 1) và
    webhook (giai đoạn 2). Bọc try/except toàn bộ — không bao giờ ném ra làm chết worker."""
    try:
        logger.info("zalo update raw: %s", update)
        # Chuẩn hoá cả 2 shape: polling (item đã bị _post bóc lớp "result") và webhook (còn
        # nguyên envelope {"ok","result"}). Không có "result" dict -> item chính là result.
        payload = update or {}
        result = payload.get("result")
        if not isinstance(result, dict):
            result = payload
        event = result.get("event_name")
        message = result.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not chat_id:
            logger.warning("zalo update thiếu chat.id: %s", update)
            return

        # Chỉ xử lý tin nhắn văn bản; event khác (ảnh/sticker/voice/unsupported) -> nhắc nhẹ.
        if event != "message.text.received":
            zalo_client.send_message(chat_id, UNSUPPORTED_MESSAGE)
            return

        text = (message.get("text") or "").strip()
        if not text:
            zalo_client.send_message(chat_id, UNSUPPORTED_MESSAGE)
            return

        sender = message.get("from") or {}
        session = get_or_create_zalo_session(chat_id, sender.get("display_name"))
        if session is None:
            zalo_client.send_message(chat_id, NO_DOCUMENT_MESSAGE)
            return

        reply = _handle_message(session["id"], session, MessageRequest(content=text))

        # Văn bản (kèm trích dẫn), chia nhỏ theo giới hạn 2000 ký tự của Zalo.
        for part in chunk_text(format_reply(reply)):
            zalo_client.send_message(chat_id, part)

        # Ảnh (nhánh image) -> gửi từng URL công khai qua sendPhoto.
        for im in reply.get("images") or []:
            if im.get("image"):
                zalo_client.send_photo(chat_id, im["image"], im.get("title"))
    except Exception as exc:  # noqa: BLE001 - không bao giờ làm chết worker/route
        logger.exception("zalo _process_zalo_update lỗi: %s", exc)