@echo off
chcp 65001 >nul
echo ==========================================
echo    R2P-Guard 一键启动脚本
echo ==========================================
echo.

:: 检查Python3
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ 错误：未找到 Python，请先安装 Python 3
        pause
        exit /b 1
    )
)

echo ✅ 找到 Python
echo.

:: 获取脚本所在目录
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

echo 正在启动服务...
echo.

:: 启动前端服务器 (8002端口)
echo 🚀 启动前端服务 (http://localhost:8002)...
start "R2P-Guard Frontend" python -m http.server 8002
echo    前端服务已启动

:: 检查8000端口（视频分析后端）
netstat -an | findstr ":8000.*LISTENING" >nul
if %errorlevel% equ 0 (
    echo ✅ 检测到视频分析服务 (8000端口) 已运行
) else (
    echo ⚠️  警告：视频分析服务 (8000端口) 未启动
    echo    如需视频分析功能，请确保 api.py 已运行
)

:: 检查8001端口（邮箱服务）
netstat -an | findstr ":8001.*LISTENING" >nul
if %errorlevel% equ 0 (
    echo ✅ 检测到邮箱服务 (8001端口) 已运行
) else (
    echo ⚠️  警告：邮箱服务 (8001端口) 未启动
    echo    如需邮箱验证功能，请确保 email_server.py 已运行
)

echo.
echo ==========================================
echo    服务已启动！
echo ==========================================
echo.
echo 📱 前端地址：http://localhost:8002/111/welcome/app_welcome.html
echo.
echo 按任意键退出...
pause >nul
