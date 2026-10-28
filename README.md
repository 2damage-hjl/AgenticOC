# AgenticOC

AI-powered NPC conversations for Stardew Valley. Uses LLM to generate dynamic, contextual dialogue with any villager.

## Quick Start (For Users)

### Prerequisites

1. **Stardew Valley** with **SMAPI** installed (minimum version 3.18.0)
2. **Content Patcher** (required dependency listed in `manifest.json`)
3. **Python 3.11** with pip
4. An **LLM API key** (DeepSeek, OpenAI, OpenRouter, Qwen, or Gemini)

### Installation

1. Download this repository (**Code → Download ZIP**, or `git clone`) and put the whole folder into your Stardew Valley `Mods` folder, e.g. `Mods\[CP] Damon\`
2. Install Python dependencies from the `ai` folder inside the mod:

```bash
cd "<Stardew Valley>\Mods\[CP] Damon\ai"
python -m pip install -r requirements.txt
```

> NVIDIA GPU users can speed up embeddings by first running
> `python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`

3. Open `ai/config.json` and fill in your API key:

```json
{
    "Provider": "deepseek",
    "ModelName": "deepseek-chat",
    "ServerAddress": "https://api.deepseek.com",
    "ApiKey": "sk-your-api-key-here",
    "Temperature": 0.75
}
```

4. Start the AI server:

```bash
cd "<Stardew Valley>\Mods\[CP] Damon\ai"
python server.py
```

or double-click **`Start_AgenticOC.bat`**. Keep the window open (server at `http://localhost:8000`).

5. Launch Stardew Valley through SMAPI

> **First launch** will take 1-2 minutes to download the AI embedding model (BGE-M3, ~2GB). Subsequent launches are instant.

> 中文部署说明见 [ai/README.md](ai/README.md)。

### How to Use

1. Start the AI server via `Start_AgenticOC.bat` (keep the window open)
2. Launch Stardew Valley
3. Walk near any NPC and press **Q** to start a conversation
4. Type your message and press Enter
5. Choose "Continue" to keep talking or "Stay Silent" to end

---

## Configuration

### config.json Fields

| Field | Description | Example |
|-------|-------------|---------|
| `Provider` | LLM provider | `deepseek`, `openai`, `openrouter`, `qwen`, `gemini` |
| `ModelName` | Model identifier | `deepseek-chat`, `gpt-4o` |
| `ServerAddress` | API endpoint URL | `https://api.deepseek.com` |
| `ApiKey` | Your API key (**required**) | `sk-xxxxxxxx` |
| `Temperature` | Response creativity (0.0-1.0) | `0.75` |

### Supported Providers

| Provider | ServerAddress | Environment Variable |
|----------|---------------|---------------------|
| DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `QWEN_API_KEY` |
| Gemini | (uses Google API) | `GOOGLE_API_KEY` |

> You can also set API keys via environment variables instead of `config.json`.

---

## For Developers (Building from Source)

### Prerequisites

- Python 3.11
- .NET 6.0 SDK (only to rebuild the mod DLL)

### Build the SMAPI Mod DLL (optional)

The prebuilt `AgenticOC.dll` is included. To rebuild it, set `GamePath` in `AgenticOC.csproj` to your Stardew Valley install if needed, then:

```bash
dotnet build -c Release
```

### Run the AI Server from Source

```bash
cd ai
python -m pip install -r requirements.txt
python server.py
```

PyInstaller packaging (`Build.bat` / `AgenticOC.spec`) is still in the repo but no longer the recommended path.

### Project Structure

```
AgenticOC/
├── AgenticOC.dll              # SMAPI mod (prebuilt)
├── manifest.json              # SMAPI manifest
├── Start_AgenticOC.bat        # One-click launcher
├── AgenticOC.csproj / src/    # C# mod source
├── ai/
│   ├── config.json            # User fills in API key here
│   ├── requirements.txt       # Pinned Python dependencies
│   ├── server.py              # FastAPI server entry point
│   ├── graph.py               # Core conversation graph
│   ├── llm.py                 # LLM factory
│   ├── memory/                # Memory systems (ST/MT/LT)
│   ├── gameinfo/              # Game state utilities
│   ├── prompt_construction/   # Prompt templates & few-shot
│   └── ...
└── README.md
```

### Nexus Mod Release Package

The repository itself is the distributable folder — zip it and users drop it into `Mods`:

```
[CP] Damon/
├── AgenticOC.dll
├── manifest.json
├── Start_AgenticOC.bat
└── ai/
    ├── config.json             # user edits this
    ├── requirements.txt
    └── ...（Python 源码）
```

Users still need Python 3.11 + `pip install -r ai/requirements.txt`.

## License

See [ai/LICENSE](ai/LICENSE) for details.
