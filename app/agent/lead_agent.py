import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.tools.image_search_tool import search_images
from app.tools.web_search_tool import search_webs
from app.prompts.system_promp import SYSTEM_PROMPT

load_dotenv()

def _build_middlewares(config: RunnableConfig, model_name: str | None, agent_name: str | None = None, custom_middlewares: list[AgentMiddleware] | None = None):
    """Build middleware chain based on runtime configuration.

    Args:
        config: Runtime configuration containing configurable options like is_plan_mode.
        agent_name: If provided, MemoryMiddleware will use per-agent memory storage.
        custom_middlewares: Optional list of custom middlewares to inject into the chain.

    Returns:
        List of middleware instances.
    """
    middlewares = build_lead_runtime_middlewares(lazy_init=True)

    # Add summarization middleware if enabled
    summarization_middleware = _create_summarization_middleware()
    if summarization_middleware is not None:
        middlewares.append(summarization_middleware)

    # Add MemoryMiddleware (after TitleMiddleware)
    middlewares.append(MemoryMiddleware(agent_name=agent_name))

    # ClarificationMiddleware should always be last
    middlewares.append(ClarificationMiddleware())

def create_custom_agent():
    # temperature=0 giữ model ổn định/ít lan man -> hạn chế gọi lại tool một cách vô ích.
    model = ChatOpenAI(
        model="ag/gemini-3.8-flash-high",
        api_key=os.getenv("9ROUTER_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=[
            search_webs,
            search_images,
        ],
        system_prompt=SYSTEM_PROMPT,
        # Chặn vòng lặp gọi tool: với các câu real-time (vd "giá cổ phiếu FPT hôm nay"),
        # model hay gọi search_webs liên tục đến khi chạm recursion_limit rồi fail. Giới hạn
        # mỗi tool tối đa 2 lần/lượt; exit_behavior="continue" -> chặn lần gọi thừa và BUỘC
        # model trả lời bằng kết quả đã thu được (thay vì dừng hẳn/lỗi).
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="search_webs", run_limit=2, exit_behavior="continue",
            ),
            ToolCallLimitMiddleware(
                tool_name="search_images", run_limit=2, exit_behavior="continue",
            ),
        ],
    )

    return agent