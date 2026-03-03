# toyclaw 

一個「超小型 openclaw 概念版」：

- LLM 用 XML 標記送指令（例如 `<shell>...</shell>`）
- toyclaw 執行後也用標記回傳（`<json>...</json>` + `<markdown>...</markdown>`）

## 核心檔案

- `toyclaw.py`：XML 訊息解析 + shell 執行器

## 快速開始

```bash
cd /Users/cccclaw/Desktop/openclaw/toyclaw
python3 toyclaw.py --message '<shell>echo "print(\"Hello, World!\")" > hello.py</shell>'
cat hello.py
```

你會看到類似輸出：

```xml
<json>{ ... "ok": true, "exit_code": 0 ... }</json>
<markdown>✅ 指令執行成功</markdown>
```

## 互動模式

```bash
python3 toyclaw.py --cwd .
```

輸入：

```xml
<shell>ls -la</shell>
```

## 掛上 Ollama（qwen2.5:3b）

先確認模型存在：

```bash
ollama pull qwen2.5:3b
```

執行一個任務（LLM 會自行產生 XML 指令並迭代）：

```bash
python3 toyclaw.py \
  --cwd /Users/cccclaw/Desktop/openclaw/toyclaw \
  --ollama-model qwen2.5:3b \
  --llm-task '建立 hello.py，內容印出 Hello from toyclaw，最後用 <final> 回報完成'
```

## 協議（目前支援）

- `<shell>...</shell>`：執行 shell
- `<markdown>...</markdown>`：接收（pass-through）
- `<json>...</json>`：接收（pass-through）

未知標記會回傳：

- `<json>{"ok": false, "error": "unsupported_tag", ...}</json>`
- `<markdown>不支援的標記...</markdown>`

## 下一步可以加

1. 多工具標記（`<read_file>`, `<write_file>`, `<http_get>`）
2. 指令白名單 / sandbox（安全）
3. 多回合 state（任務 id、trace id）
4. 接 LLM API（讓模型直接產生 XML、迭代直到完成）
