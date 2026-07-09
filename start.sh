#!/usr/bin/env bash
# MovieTranslator 一键启动（Ubuntu / Linux）
# 首次运行自动创建虚拟环境并安装依赖；之后直接启动。
# GPU 机器首次可执行:  ./start.sh --gpu
set -e
cd "$(dirname "$0")/backend"

if [ ! -d .venv ]; then
    echo "[MovieTranslator] 首次运行：创建虚拟环境并安装依赖，需要几分钟…"
    python3 -m venv .venv
    .venv/bin/pip install -e .
fi

if [ "$1" = "--gpu" ]; then
    echo "[MovieTranslator] 安装 CUDA 运行库…"
    .venv/bin/pip install -e ".[gpu]"
fi

# 服务就绪后自动打开浏览器
( sleep 2; xdg-open "http://127.0.0.1:8760" >/dev/null 2>&1 || true ) &

echo "[MovieTranslator] 启动中，浏览器访问 http://127.0.0.1:8760 （Ctrl+C 退出）"
exec .venv/bin/python -m uvicorn app.main:app --port 8760
