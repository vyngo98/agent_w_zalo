import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from pydantic import BaseModel, Field
from typing import Any
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.tools.image_search_tool import search_images
from app.tools.web_search_tool import search_webs
from app.prompts.system_prompt import SYSTEM_PROMPT

from langgraph.store.memory import InMemoryStore
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)

from app.middlewares.memory_middleware import LongTermMemoryMiddleware

# from app.middlewares.tool_error_handling_middleware import build_lead_runtime_middlewares
# from app.middlewares.clarification_middleware import ClarificationMiddleware
#
# load_dotenv()
#
# def _build_middlewares(config: RunnableConfig, model_name: str | None, agent_name: str | None = None, custom_middlewares: list[AgentMiddleware] | None = None):
#     """Build middleware chain based on runtime configuration.
#
#     Args:
#         config: Runtime configuration containing configurable options like is_plan_mode.
#         agent_name: If provided, MemoryMiddleware will use per-agent memory storage.
#         custom_middlewares: Optional list of custom middlewares to inject into the chain.
#
#     Returns:
#         List of middleware instances.
#     """
#     middlewares = build_lead_runtime_middlewares(lazy_init=True)
#
#     # Add summarization middleware if enabled
#     summarization_middleware = _create_summarization_middleware()
#     if summarization_middleware is not None:
#         middlewares.append(summarization_middleware)
#
#     # Add MemoryMiddleware (after TitleMiddleware)
#     middlewares.append(MemoryMiddleware(agent_name=agent_name))
#
#     # ClarificationMiddleware should always be last
#     middlewares.append(ClarificationMiddleware())

# ----------------------------------------------------------------------------
# Context (truyền user_id mỗi lần invoke)
# ----------------------------------------------------------------------------
@dataclass
class Context:
    user_id: str = "default"

def create_custom_agent():
    # temperature=0 giữ model ổn định/ít lan man -> hạn chế gọi lại tool một cách vô ích.
    model = ChatOpenAI(
        model="ag/gemini-3.8-flash-high",
        api_key=os.getenv("9ROUTER_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0,
    )
    memory_model = ChatOpenAI(
        model="ag/gemini-3.8-flash-low",
        api_key=os.getenv("9ROUTER_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0,
    )

    # agent = create_agent(
    #     model=model,
    #     tools=[
    #         search_webs,
    #         search_images,
    #     ],
    #     system_prompt=SYSTEM_PROMPT,
    #     # Giới hạn mỗi tool tối đa 2 lần/lượt; exit_behavior="continue" -> chặn lần gọi thừa và BUỘC
    #     # model trả lời bằng kết quả đã thu được (thay vì dừng hẳn/lỗi).
    #     middleware=[
    #         ToolCallLimitMiddleware(
    #             tool_name="search_webs", run_limit=2, exit_behavior="continue",
    #         ),
    #         ToolCallLimitMiddleware(
    #             tool_name="search_images", run_limit=2, exit_behavior="continue",
    #         ),
    #     ],
    #     checkpointer=InMemorySaver()
    # )


    # Store dùng cho long-term memory; index để search ngữ nghĩa (đổi embedding theo provider của bạn)
    store = InMemoryStore(
        index={
            "embed": "ollama:nomic-embed-text",
            "dims": 1536,
            "fields": ["text"],
        }
    )

    agent = create_agent(
        model=model,
        tools=[search_webs,
             search_images],
        system_prompt=SYSTEM_PROMPT,
        context_schema=Context,
        middleware=[
            # --- WORKING MEMORY: tự tóm tắt khi hội thoại dài, giữ 20 message gần nhất
            SummarizationMiddleware(
                model=memory_model,
                trigger=("tokens", 4000),
                keep=("messages", 20),
            ),
            # --- SEMANTIC + EPISODIC + PROCEDURAL
            LongTermMemoryMiddleware(
                extractor_model=memory_model,
                seed_rules=["Always cite the source when using web search results"],
            ),
            ToolCallLimitMiddleware(
                tool_name="search_webs", run_limit=2, exit_behavior="continue"
            ),
            ToolCallLimitMiddleware(
                tool_name="search_images", run_limit=2, exit_behavior="continue"
            ),
         ],
        checkpointer=InMemorySaver(),
        store=store,
        )

    return agent