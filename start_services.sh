#!/bin/bash
# PID智能整定系统 - 服务启动脚本

echo "启动PID智能整定系统..."

# 检查并停止已有服务（使用端口号查找）
echo "检查现有服务..."
BACKEND_PID=$(lsof -ti:3443 2>/dev/null)
if [ -n "$BACKEND_PID" ]; then
    echo "停止现有后端服务 (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
    sleep 1
fi

FRONTEND_PID=$(lsof -ti:5273 2>/dev/null)
if [ -n "$FRONTEND_PID" ]; then
    echo "停止现有前端服务 (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null
    sleep 1
fi

# 启动后端API (端口3443)
echo "启动后端API (端口3443)..."
cd /run/code/dinglei/pid_tuning/backend
nohup python agents_multiagent.py > agents_multiagent.log 2>&1 &
BACKEND_NEW_PID=$!
echo "后端PID: $BACKEND_NEW_PID"

# 等待后端启动
sleep 3

# 启动前端服务 (端口5273)
echo "启动前端服务 (端口5273)..."
cd /run/code/dinglei/pid_tuning/frontend
nohup python -m http.server 5273 > frontend.log 2>&1 &
FRONTEND_NEW_PID=$!
echo "前端PID: $FRONTEND_NEW_PID"

# 等待服务启动
sleep 2

# 验证服务状态
echo ""
echo "验证服务状态..."
if lsof -i:3443 >/dev/null 2>&1; then
    echo "✓ 后端API运行正常 (端口3443)"
else
    echo "✗ 后端API启动失败"
    echo "查看日志: tail -f /run/code/dinglei/pid_tuning/backend/agents_multiagent.log"
fi

if lsof -i:5273 >/dev/null 2>&1; then
    echo "✓ 前端服务运行正常 (端口5273)"
else
    echo "✗ 前端服务启动失败"
    echo "查看日志: tail -f /run/code/dinglei/pid_tuning/frontend/frontend.log"
fi

echo ""
echo "访问地址:"
echo "  前端界面: http://pid.dicp.sixseven.ltd:5924"
echo "  后端API:  http://pidend.dicp.sixseven.ltd:5924"
echo ""
echo "查看日志:"
echo "  后端: tail -f /run/code/dinglei/pid_tuning/backend/agents_multiagent.log"
echo "  前端: tail -f /run/code/dinglei/pid_tuning/frontend/frontend.log"
