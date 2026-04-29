#!/bin/bash

echo "=========================================="
echo "   R2P-Guard 一键启动脚本"
echo "=========================================="
echo ""

# 检查Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 Python3，请先安装 Python 3"
    exit 1
fi

echo "✅ 找到 Python3"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "正在启动服务..."
echo ""

# 启动前端服务器 (8002端口)
echo "🚀 启动前端服务 (http://localhost:8002)..."
python3 -m http.server 8002 > /dev/null 2>&1 &
FRONTEND_PID=$!
echo "   前端服务已启动 (PID: $FRONTEND_PID)"

# 检查8000端口（视频分析后端）
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✅ 检测到视频分析服务 (8000端口) 已运行"
else
    echo "⚠️  警告：视频分析服务 (8000端口) 未启动"
    echo "   如需视频分析功能，请确保 api.py 已运行"
fi

# 检查8001端口（邮箱服务）
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✅ 检测到邮箱服务 (8001端口) 已运行"
else
    echo "⚠️  警告：邮箱服务 (8001端口) 未启动"
    echo "   如需邮箱验证功能，请确保 email_server.py 已运行"
fi

echo ""
echo "=========================================="
echo "   服务已启动！"
echo "=========================================="
echo ""
echo "📱 前端地址：http://localhost:8002/111/welcome/app_welcome.html"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo ""

# 等待用户中断
trap "echo ''; echo '正在停止服务...'; kill $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait $FRONTEND_PID
