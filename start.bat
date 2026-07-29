@echo off
rem MovieTranslator 一键启动（Windows）
rem 首次运行自动创建虚拟环境并安装依赖；之后直接启动。
rem GPU 机器首次可执行:  start.bat --gpu
chcp 65001 >nul
setlocal
cd /d "%~dp0backend"

if not exist .venv (
    echo [MovieTranslator] 首次运行：创建虚拟环境并安装依赖，需要几分钟…
    python -m venv .venv
    if errorlevel 1 goto :error
    .venv\Scripts\pip install -e .
    if errorlevel 1 goto :error
)

if "%~1"=="--gpu" (
    echo [MovieTranslator] 安装 CUDA 运行库…
    .venv\Scripts\pip install -e ".[gpu]"
    if errorlevel 1 goto :error
)

rem 端口来自设置文件（默认 8760），所以这里要问程序自己
set PORT=8760
for /f %%p in ('.venv\Scripts\python -c "from app.core.config import load_settings; print(load_settings().server.port)" 2^>nul') do set PORT=%%p

rem 3 秒后自动打开浏览器（本机回环始终免令牌）
start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:%PORT%"

echo [MovieTranslator] 启动中，浏览器访问 http://127.0.0.1:%PORT% （关闭本窗口即退出）
rem 用 python -m app.main 而不是 uvicorn：绑定地址（本机/局域网）与端口只有 main() 会读设置
.venv\Scripts\python -m app.main
goto :eof

:error
echo.
echo [MovieTranslator] 安装失败：请确认已安装 Python 3.10 及以上并加入 PATH。
pause
