# MovieTranslator 🎬

跨平台（Windows / Linux）电影字幕翻译软件。把视频文件一键变成高质量双语 SRT 字幕：

**视频 → PyAV 提取音频 → Faster Whisper 语音识别 → LLM 全局上下文翻译 → SRT**

## 特点

- **全局一致性翻译**：不同于传统分段翻译，整片字幕一次性交给大模型建立全局上下文（先生成人名/术语对照表），再分批输出译文，人名、术语、语气全片统一。
- **省 token**：发给 LLM 的只有行号+文本，时间戳全程留在本地。
- **兼容任意 OpenAI 风格 API**：OpenAI / DeepSeek / 通义 / Kimi / Ollama 本地模型均可，填 base_url + api_key + model 即用。
- **本地识别**：Faster Whisper（默认 large-v2，CPU int8），无需上传音频到任何服务。
- **隐私友好**：中间文件只存本地缓存目录，每次启动自动清空。
- **网页 UI**：任务进度实时推送（分阶段进度条 + 日志），支持源语言选择、剧情简介（提升翻译准确性）、双语/纯译文切换、字幕与识别参数设置。

## 快速开始（一键启动）

**推荐方式**：从 [Releases](../../releases) 下载最新的 `MovieTranslator-*.zip` 并解压——发行包已含构建好的前端，无需 Node.js。前提：已安装 Python ≥ 3.10 并加入 PATH。

- **Windows**：双击 `start.bat`（GPU 机器首次在终端运行 `start.bat --gpu`）
- **Ubuntu / Linux**：`./start.sh`（GPU 机器首次 `./start.sh --gpu`）

首次运行脚本会自动创建虚拟环境、安装依赖（需几分钟），之后每次直接启动并自动打开浏览器 **http://127.0.0.1:8760**：

1. 在「设置」页填好 LLM 的 base_url / api_key / model，点「测试连接」；
2. 回到首页选择视频文件、源语言（可 auto）、目标语言，可选填剧情简介；
3. 点「开始翻译」，等待完成后下载 SRT。

whisper 模型仅在本地缓存中不存在时才下载（首次选用某模型时自动进行，large-v2 约 3GB）；已下载的模型完全离线加载。设置页会显示当前模型是否已下载。

### GPU 加速（NVIDIA CUDA）

CPU 可用但较慢（large-v2 约 1× 实时）。有 NVIDIA GPU 时：

```bash
cd backend
.venv/bin/pip install -e ".[gpu]"    # 安装 nvidia-cublas-cu12 / nvidia-cudnn-cu12
```

然后在设置页把「设备」切到 **CUDA**（或「自动」），计算精度推荐 `float16` 或 `int8_float16`。设置页会显示本机是否检测到可用 CUDA 设备；无 GPU 时选 CUDA 会得到明确的错误提示。

**Windows 注意**：pip 安装的 cuBLAS/cuDNN DLL 位于 `site-packages\nvidia\*\bin`，不在系统 DLL 搜索路径上——程序启动使用 CUDA 时会自动注册这些目录，装完 `.[gpu]` 重启即可。如果仍报 `cublas64_12.dll not found`，备选方案：从 [Purfview/whisper-standalone-win Releases](https://github.com/Purfview/whisper-standalone-win/releases) 下载 `cuBLAS.and.cuDNN` 压缩包，把 DLL 解压到 backend 目录或任一 PATH 目录。

## 从源码运行（git clone 用户）

源码仓库不含前端构建产物，需要先构建一次前端（需 Node.js ≥ 18）：

```bash
git clone <本仓库地址> && cd MovieTranslator
cd frontend && npm install && npm run build && cd ..
./start.sh        # Windows: start.bat
```

或全手动：

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -e .          # Windows: .venv\Scripts\pip install -e .
.venv/bin/python -m app.main        # 监听地址与端口读自设置文件
```

## 局域网访问与 MCP 服务

两项都**默认关闭**，不做任何设置时程序只监听 `127.0.0.1`，行为与以往一致。

### 局域网访问

在「设置 → 局域网访问与 MCP 服务」里开启并**重启程序**，手机、笔记本等同网段设备即可打开网页发起翻译。设置页会给出带访问令牌的完整链接，在别的设备上打开一次即可（令牌随后存为 Cookie，不必再带）。

⚠️ 本程序的接口默认假定使用者就是本机用户：可以浏览**本机任意目录**、对任意路径发起任务、下载含片源路径的任务日志。所以开启后请保持「需要访问令牌」开启。本机（`127.0.0.1`）始终免验证，不会把自己锁在外面。

⚠️ **只提供明文 HTTP，没有 TLS**。不要把端口直接转发到公网——需要在外网使用时请套一层反向代理（Caddy / Nginx）或用 Tailscale、Cloudflare Tunnel 之类的隧道。

### MCP 服务

开启后（开关立即生效，无需重启），Claude 等支持 MCP 的客户端可以直接驱动这台机器完成翻译。传输为 streamable HTTP，地址 `http://<地址>:<端口>/mcp`：

```bash
claude mcp add --transport http movietranslator http://192.168.1.10:8760/mcp \
  --header "Authorization: Bearer <设置页里的访问令牌>"
```

提供的工具：列出目录里的视频、列音轨、发起单片翻译、发起整目录批量翻译、查询任务/批量进度、取消、取回字幕全文、取回任务日志、查询服务状态。

任务在后台运行，发起后立刻返回任务号，由客户端轮询进度——一部影片要跑几十分钟到几小时。

**不提供修改设置的能力**：识别模型、API key 等只能在网页设置页改动，远程无法改。

## 开发

```bash
# 后端热重载
cd backend && .venv/bin/uvicorn app.main:app --port 8760 --reload
# 前端 dev server（/api 代理到 8760）
cd frontend && npm run dev
# 测试
cd backend && .venv/bin/pytest
```

## 项目结构

```
backend/app/services/   # 核心：audio(PyAV) → asr(faster-whisper) → segmenter
                        #       → translator(LLM 全局上下文协议) → subtitle(SRT)
backend/app/api/        # REST + SSE 端点
frontend/src/views/     # 翻译任务 / 设置 / 提示词 三个页面
```

## 隐私说明

- 语音识别默认在本地运行，音频不上传任何服务器
- 发给翻译 API 的只有字幕文本（不含时间戳、不含音频）
- API key 存储在本机用户配置目录，不在项目文件夹内
- 中间文件只存本地缓存目录，每次启动自动清空
- 默认只监听 `127.0.0.1`；局域网访问与 MCP 服务都需要手动开启
- 任务日志与 MCP 返回的内容都不含 API key 或访问令牌

## 贡献

欢迎 Issue 和 PR。提交 PR 前请确保 `cd backend && pytest` 通过；打 `v*` 标签会自动构建并发布 Release。

## 许可

[MIT](LICENSE)
