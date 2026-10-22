# Zalo AI Agent

An AI-powered conversational agent integrated with **Zalo Bot API** using Long Polling. The agent allows users to interact through Zalo messages and supports web search and image search through dedicated tools.

## Features

- **Zalo Bot Integration:** Connects to Zalo through the Bot API.
- **Long Polling:** Receives incoming messages by continuously polling for updates.
- **Web Search:** Searches the web for relevant information using the `search_webs` tool.
- **Image Search:** Searches for images using the `search_images` tool.
- **AI-powered Responses:** Uses an LLM to understand user requests, select tools, and generate responses.
- **Image Handling:** Supports collecting image search results for downstream processing and delivery.
- **Tool Call Limits:** Limits tool executions per request to reduce unnecessary repeated searches.

## Architecture

```text
User (Zalo)
    |
    v
Zalo Bot API
    |
    v
main.py
    |
    v
Long Polling
    |
    v
Incoming Message Handler
    |
    v
AI Agent
    |
    +-------------------+
    |                   |
    v                   v
 Web Search         Image Search
    |                   |
    +---------+---------+
              |
              v
       LLM Response
              |
              v
      Outgoing Message
              |
              v
        Zalo Bot API
              |
              v
        User (Zalo)
```

## Project Structure

The following is an example structure. Adjust the module names to match the actual project.

```text
project/
├── main.py              # Application entry point and server startup
├── .env                 # Environment variables and bot token
├── .env.example         # Example environment configuration
├── requirements.txt     # Python dependencies
├── app/
│   └── agent            # define agent
│   ├── integrations     # code to integrate channels
│   |   └── zalo.py     
│   ├── prompts          # contain prompts
│   |   └── system_prompt.py
│   └── tools            # 
│       └── image_search_tool.py  # duckduckgo
│       └── web_search_tool.py    # tavily
|   ...
└── README.md
```

## Requirements

- Python 3.11 or a compatible Python version
- A Zalo Bot and its access token
- An LLM provider API key and endpoint configuration
- Internet access for Zalo API requests and web/image searches

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:vyngo98/agent_w_zalo.git-url
cd agent_w_zalo
```

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Configure environment variables

Create a `.env` file in the project root:

```dotenv
# Required: Zalo Bot access token
ZALO_BOT_TOKEN=your_zalo_bot_token
ZALO_ENABLE_POLLING=1

# LLM configuration
9ROUTER_API_KEY=your_llm_api_key
BASE_URL=your_llm_api_base_url

# TAVILY configuration
TAVILY_API_KEY=yout_tavily_api_key
```

The LLM variables above are examples based on the current agent configuration. Use the exact variable names expected by your implementation.

**Important:** Never commit `.env` or expose your bot token in source code, logs, screenshots, or public repositories.

Add the following entries to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
```

### 2. Obtain a Zalo Bot token

1. Create or configure a bot through the Zalo Bot platform.
2. Obtain the bot access token.
3. Add the token to `ZALO_BOT_TOKEN` in your `.env` file.
4. Ensure the bot is available to the intended users and that the relevant messaging permissions are configured.

The bot token authenticates requests to the Zalo Bot API. Keep it private.

## Running the Application

### 1. Start the server

Run `main.py` from the project root:

```bash
python main.py
```

The application must be started before users begin interacting with the bot.

On startup, the application should load the environment variables, initialize the AI agent, and start the Long Polling loop.

Keep the process running to continue receiving messages.

### 2. Send a message through Zalo

Once the application is running:

1. Open Zalo.
2. Access the configured bot.
3. Send a text message or a request that requires web or image search.
4. The application receives the message through Long Polling.
5. The agent processes the request and sends a response through the Zalo Bot API.

Example requests:

- "Search the web for the latest AI news."
- "Find information about NVIDIA's latest GPUs."
- "Show me pictures of cute baby cats."
- "Find images of the Eiffel Tower."

Actual results depend on the configured LLM, search tools, and available external services.

## AI Agent and Tools

The agent is configured with two tools.

### `search_webs`

Searches the web to retrieve relevant information for user questions.

Typical use cases:

- Current events and recent information
- Company and product research
- Technical documentation
- General knowledge requiring external sources

### `search_images`

Searches for images relevant to the user's request. The implementation uses DuckDuckGo image search and normalizes results into a consistent structure.

Example image result:

```json
{
  "title": "Example image",
  "image": "https://example.com/image.jpg",
  "thumbnail": "https://example.com/thumbnail.jpg",
  "url": "https://example.com/source"
}
```

The application can collect image results separately from the LLM's natural-language response so that images can be delivered or processed independently.

### Tool Execution Limits

The agent can use middleware to limit repeated tool calls during a single request. For example, each search tool can be limited to two executions per agent run.

This helps reduce redundant requests, unnecessary latency, and excessive API usage.

## Long Polling

The application uses Long Polling to retrieve incoming messages from the Zalo Bot API.

The general process is:

1. Start the application.
2. Request updates from the Zalo Bot API.
3. Wait for new messages or updates.
4. Process incoming messages.
5. Invoke the AI agent.
6. Send the response back to the user.
7. Continue polling for subsequent updates.

Unlike a webhook-based integration, Long Polling does not require a publicly accessible webhook endpoint. However, the application must remain running and maintain network connectivity to receive messages.

For production deployments, implement appropriate request timeouts, retry and backoff policies, update-offset management, error handling, and graceful shutdown.

## Troubleshooting

| Problem | Suggested action |
|---|---|
| `ZALO_BOT_TOKEN` is missing | Verify that `.env` exists and that environment variables are loaded before accessing the token. |
| Bot does not receive messages | Check the token, network connectivity, polling loop, and bot access permissions. |
| Web search fails | Check internet connectivity, search provider availability, and tool error logs. |
| Image search returns no results | Check the search query, DuckDuckGo availability, and the image search error logs. |
| Images are found but not delivered | Verify the image URLs and confirm that the Zalo sending method supports the supplied URL or file input. |
| Agent repeatedly calls a tool | Check the tool-call limit middleware and agent configuration. |
| LLM requests fail | Verify the API key, base URL, model identifier, and provider availability. |

## Security Considerations

- Store credentials in environment variables rather than source code.
- Never commit `.env` to version control.
- Restrict bot access if the application is intended for authorized users only.
- Validate incoming messages and external URLs.
- Apply timeouts and resource limits to external HTTP requests.
- Handle malformed tool outputs and API errors safely.
- Avoid logging bot tokens or sensitive user information.

## Limitations

- Search results depend on external search services and their availability.
- LLM-generated responses may contain inaccuracies and should be validated for critical use cases.
- Long Polling requires a continuously running application process.
- Local image paths are only accessible to the machine or service that stores those files. Sending images to Zalo may require a supported file-upload mechanism or an externally accessible image URL.

## License

Add the applicable license information here.
