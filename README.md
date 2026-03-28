# gemini_flow

> Fork of [JasonHongGG/GeminiFlow](https://github.com/JasonHongGG/GeminiFlow) — 升级支持多轮对话（Multi-turn Chat）

## Install

```bash
python -m pip install .
```

通过 GitHub 安装：

```bash
python -m pip install "git+https://github.com/zeronorm/GeminiFlow.git"
```

开发模式：

```bash
python -m pip install -e .
```

安装后可在任意目录直接执行：

```bash
gemini-flow --help
gemini-flow-server --help
python -m gemini_flow --help
```

如果本机是离线环境，安装本地仓库时可加：

```bash
python -m pip install . --no-build-isolation
```

## Python Import

除了命令行，也可以直接作为 Python 包导入使用：

```python
from gemini_flow import Gemini, ChatSession, export_cookies, create_app

cookies_path = export_cookies(output_dir="user_cookies")

client = Gemini(cookies_dir="user_cookies")
session = ChatSession()
text = client.chat("用繁中回覆一句：測試成功", chat_session=session)

app = create_app()
```

异步接口：

```python
from gemini_flow import Gemini, aexport_cookies, serve

await aexport_cookies(output_dir="user_cookies")

client = Gemini(cookies_dir="user_cookies")
text = await client.achat("hello")

await serve(host="127.0.0.1", port=8000)
```

## Run

### 单轮模式（Single-turn）

```bash
gemini-flow chat -c /path/to/user_cookies "用繁中回覆一句：測試成功"
```

Choose model:

```bash
gemini-flow chat -m gemini-3-pro -c /path/to/user_cookies "用繁中回覆一句：測試成功"
gemini-flow chat -m gemini-3-flash -c /path/to/user_cookies "用繁中回覆一句：測試成功"
```

Debug mode (prints token/response previews):

```bash
gemini-flow chat --debug -c /path/to/user_cookies "hello"
```

### 多轮对话模式 (Multi-turn CLI)

不带初始 prompt 启动，进入交互式 REPL，支持多轮上下文记忆：

```bash
gemini-flow chat -c /path/to/user_cookies
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
gemini-flow-server --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 同步 Cookie

服务端支持通过接口触发一次 Chrome cookie 导出：

```bash
curl -X POST http://127.0.0.1:8000/sync-cookies \
  -H "Content-Type: application/json" \
  -d '{}'
```

如果未在请求体中传 `output_dir`，会默认使用环境变量 `GEMINI_FLOW_COOKIE_SYNC_DIR`。

也可以显式指定导出目录和 profile：

```bash
curl -X POST http://127.0.0.1:8000/sync-cookies \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "user_cookies",
    "output_filename": "auth_Gemini.json",
    "profile_directory": "Profile 1"
  }'
```

可选字段：
- `output_dir`: 导出目录，未传时回退到 `GEMINI_FLOW_COOKIE_SYNC_DIR`
- `output_filename`: 导出文件名，默认 `auth_Gemini.json`
- `chrome_user_data_dir`: Chrome User Data 根目录
- `profile_directory`: 指定 profile，例如 `Default` 或 `Profile 1`
- `debug`: 是否输出调试信息

返回值会包含最终导出的 `output_path` 和实际使用的 `profile_directory`。

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

### 浏览器 Cookie 同步

可以通过环境变量 `GEMINI_FLOW_COOKIE_SYNC_DIR` 指定一个“浏览器 Cookie 同步目录”。
程序在每次读取 cookies 前，会先把该目录下的 `*.json` 文件同步到当前运行使用的 `cookies_dir`（例如默认的 `user_cookies`）。

```bash
export GEMINI_FLOW_COOKIE_SYNC_DIR=/path/to/browser-cookie-sync
gemini-flow chat -c /path/to/user_cookies "hello"
```

说明：
- 同步源目录只读取顶层的 `*.json` 文件。
- 如果目标目录里已存在同名文件，只有源文件更新后才会覆盖。
- 未设置该环境变量时，保持当前行为不变。

### 从 Chrome 导出 Cookie

可以直接从当前活跃的 Chrome profile 导出 Gemini 所需 cookies：

```bash
export GEMINI_FLOW_COOKIE_SYNC_DIR=/path/to/browser-cookie-sync
gemini-flow export-cookies
```

也可以显式指定导出目录：

```bash
gemini-flow export-cookies --output-dir /path/to/user_cookies
```

可选参数：
- `--output-filename` 自定义导出文件名，默认 `auth_Gemini.json`
- `--chrome-user-data-dir` 指定 Chrome User Data 根目录
- `--profile-directory` 指定 profile，例如 `Default` 或 `Profile 1`

说明：
- 未传 `--output-dir` 时，默认读取 `GEMINI_FLOW_COOKIE_SYNC_DIR`
- 如果两者都没有提供，命令会直接报错
- 当前实现按 macOS Chrome 目录结构读取活跃 profile

图片输出目录仍可通过 `GEMINI_FLOW_IMAGE_DIR` 指定。

## Notes
- If you see `SNlM0e token not found`, your cookies are likely expired.
```
chrome.exe --user-data-dir="C:\...\GeminiFlow\user_cookies\.pw-profile"
```
