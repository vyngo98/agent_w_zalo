import os
from dotenv import load_dotenv

load_dotenv()

# Web search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Image search — DuckDuckGo qua gói `ddgs`. Không cần API key.
IMAGE_SEARCH_MAX_RESULTS = int(os.getenv("IMAGE_SEARCH_MAX_RESULTS", "2"))
IMAGE_SEARCH_REGION = os.getenv("IMAGE_SEARCH_REGION", "vn-vi")
IMAGE_SEARCH_SAFESEARCH = os.getenv("IMAGE_SEARCH_SAFESEARCH", "moderate")

# Kênh Zalo Bot (app/integrations/). Giai đoạn 1: long-polling (chưa cần webhook secret).
ZALO_BOT_TOKEN = os.getenv("ZALO_BOT_TOKEN", "")
ZALO_API_BASE = os.getenv("ZALO_API_BASE", "https://bot-api.zaloplatforms.com")
ZALO_ENABLE_POLLING = os.getenv("ZALO_ENABLE_POLLING", "0") not in ("0", "false", "False", "")
ZALO_POLL_TIMEOUT = int(os.getenv("ZALO_POLL_TIMEOUT", "30"))

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")

DOCUMENTS_PATH = os.path.join(_DATA_DIR, "documents.json")
SESSIONS_DIR = os.path.join(_DATA_DIR, "sessions")
ZALO_CHATS_PATH = os.path.join(_DATA_DIR, "zalo_chats.json")  # map chat_id Zalo -> session_id

REQUEST_LOG_PATH = os.path.join(_DATA_DIR, "request_logs.jsonl")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.8-flash-high"
)

# Langfuse tracing (tùy chọn). Thiếu key hoặc chưa cài gói -> tracing tự tắt, không ảnh hưởng
# pipeline. Gói langfuse đọc trực tiếp các biến LANGFUSE_* từ môi trường (đã nạp qua .env).
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "1") not in ("0", "false", "False", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")