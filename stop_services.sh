#!/bin/bash
# PID智能整定系统 - 服务停止脚本

echo "停止PID智能整定系统..."

# 停止后端（通过3443端口查找进程）
echo "停止后端API（端口3443）..."
BACKEND_PID=$(lsof -ti:3443 2>/dev/null)
if [ -n "$BACKEND_PID" ]; then
    kill $BACKEND_PID 2>/dev/null
    sleep 1
    # 如果进程还在运行，强制杀死
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill -9 $BACKEND_PID 2>/dev/null
    fi
    echo "✓ 后端已停止 (PID: $BACKEND_PID)"
else
    echo "- 后端未运行"
fi

# 停止前端（通过5273端口查找进程）
echo "停止前端服务（端口5273）..."
FRONTEND_PID=$(lsof -ti:5273 2>/dev/null)
if [ -n "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null
    sleep 1
    # 如果进程还在运行，强制杀死
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill -9 $FRONTEND_PID 2>/dev/null
    fi
    echo "✓ 前端已停止 (PID: $FRONTEND_PID)"
else
    echo "- 前端未运行"
fi

echo ""
echo "所有服务已停止"
