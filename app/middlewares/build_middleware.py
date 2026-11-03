from langchain.agents.middleware import AgentMiddleware

from app.llm import create_chat_model
from app.middlewares.title_middleware import TitleMiddleware
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from app.middlewares.memory_middleware import LongTermMemoryMiddleware

def _build_runtime_middlewares(
    *,
    include_dangling_tool_call_patch: bool,
    lazy_init: bool = True,
) -> list[AgentMiddleware]:
    """Build shared base middlewares for agent execution."""
    from app.middlewares.tool_error_handling_middleware import ToolErrorHandlingMiddleware
    from app.middlewares.thread_data_middleware import ThreadDataMiddleware
    from app.middlewares.llm_error_handling_middleware import LLMErrorHandlingMiddleware
    middlewares: list[AgentMiddleware] = [
        ThreadDataMiddleware(lazy_init=lazy_init),
    ]

    if include_dangling_tool_call_patch:
        from app.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware

        middlewares.append(DanglingToolCallMiddleware())

    middlewares.append(LLMErrorHandlingMiddleware())

    middlewares.append(ToolErrorHandlingMiddleware())
    middlewares.append(TitleMiddleware())

    # --- WORKING MEMORY: tự tóm tắt khi hội thoại dài, giữ 20 message gần nhất
    middlewares.append(SummarizationMiddleware(
        model=create_chat_model("ag/gemini-3.8-flash-low"),
        trigger=("tokens", 4000),
        keep=("messages", 20),
    ))
    # --- SEMANTIC + EPISODIC + PROCEDURAL
    middlewares.append(LongTermMemoryMiddleware(
        extractor_model=create_chat_model("ag/gemini-3.8-flash-low"),
        seed_rules=["Always cite the source when using web search results"],
    ))
    middlewares.append(ToolCallLimitMiddleware(
        tool_name="search_webs", run_limit=2, exit_behavior="continue"
    ))
    middlewares.append(ToolCallLimitMiddleware(
        tool_name="search_images", run_limit=2, exit_behavior="continue"
    ))

    return middlewares


def build_lead_runtime_middlewares(*, lazy_init: bool = True) -> list[AgentMiddleware]:
    """Middlewares shared by lead agent runtime before lead-only middlewares."""
    return _build_runtime_middlewares(
        include_dangling_tool_call_patch=True,
        lazy_init=lazy_init,
    )