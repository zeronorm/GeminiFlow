# gemini_flow

> Fork of [JasonHongGG/GeminiFlow](https://github.com/JasonHongGG/GeminiFlow) — 升级支持多轮对话（Multi-turn Chat）

## Install

```bash
cd projects/gemini_flow
python -m pip install -r requirements.txt
```

## Run

### 单轮模式（Single-turn）

```bash
python cli.py chat -c user_cookies "用繁中回覆一句：測試成功"
```

Choose model:

```bash
python cli.py chat -m gemini-3-pro -c user_cookies "用繁中回覆一句：測試成功"
python cli.py chat -m gemini-3-flash -c user_cookies "用繁中回覆一句：測試成功"
```

Debug mode (prints token/response previews):

```bash
python cli.py chat --debug -c user_cookies "hello"
```

### 多轮对话模式 (Multi-turn CLI)

不带初始 prompt 启动，进入交互式 REPL，支持多轮上下文记忆：

```bash
python cli.py chat -c user_cookies
```

启动后在 `You:` 提示符下输入每一轮的提问。输入 `exit` 或 `quit` 结束会话：

```
Starting interactive session. Type 'exit' or 'quit' to close.
You: 周杰伦的老婆是谁？
Gemini: 昆凌（Hannah Quinlivan）...
You: 她的年龄
Gemini: 昆凌出生于1993年8月12日，目前她32岁...
You: quit
```

> 每次重新运行命令即开启一个全新的对话（context 不保留）。

---

## Server

Start an HTTP server:

```bash
python server.py --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 单轮 Chat（返回完整文本）

```bash
curl -X POST http://127.0.0.1:8000/chat \
	-H "Content-Type: application/json" \
	-d '{"prompt":"用繁中回覆一句：測試成功","model":"gemini-2.5-pro"}'
```

### 多轮 Chat（Multi-turn via HTTP API）

第一轮请求和单轮一样，但返回的 JSON 中会包含 `conversation_id`, `response_id`, `choice_id`：

```json
{
  "text": "昆凌（Hannah Quinlivan）...",
  "images": [],
  "conversation_id": "c_xxxx",
  "response_id": "r_xxxx",
  "choice_id": "rc_xxxx"
}
```

将这三个 ID 带入下一轮请求即可保持对话记忆：

```bash
curl -X POST http://127.0.0.1:8000/chat \
	-H "Content-Type: application/json" \
	-d '{
	  "prompt": "她的年龄",
	  "model": "gemini-2.5-pro",
	  "conversation_id": "c_xxxx",
	  "response_id": "r_xxxx",
	  "choice_id": "rc_xxxx"
	}'
```

> 新建对话：不传入这三个 ID 即视为全新会话。

### Stream（SSE 流式输出）

```bash
curl -N -X POST http://127.0.0.1:8000/stream \
	-H "Content-Type: application/json" \
	-d '{"prompt":"講一個故事"}'
```

流式输出的 `done` 事件也会携带 `conversation_id`, `response_id`, `choice_id`，供下一轮使用。

---

## Streamlit 聊天界面

```bash
# 先启动后端 server（保持后台运行）
python server.py --host 127.0.0.1 --port 8000

# 新开一个终端，启动聊天 UI
conda activate dev
streamlit run app.py
```

然后浏览器访问 `http://localhost:8501`，支持：
- 多轮上下文对话记忆
- 点击「**➕ 创建新会话**」清空历史重新开始
- 模型切换
- 调试面板（显示当前 Session IDs 和对话轮数）

---

## Cookie file format

The cookies directory should contain one or more `*.json` files exported from Chrome/extensions.
Each file must be a JSON list of objects including at least: `domain`, `name`, `value`.

## Notes
- If you see `SNlM0e token not found`, your cookies are likely expired.
```
chrome.exe --user-data-dir="C:\...\GeminiFlow\user_cookies\.pw-profile"
```
