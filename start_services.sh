#\!/bin/bash
# PID智能整定系统 - 服务启动脚本

echo "启动PID智能整定系统..."

# 检查并停止已有服务
echo "检查现有服务..."
ps aux | grep backend_api.py | grep -v grep | awk "{print \$2}" | xargs kill 2>/dev/null
ps aux | grep "http.server 5273" | grep -v grep | awk "{print \$2}" | xargs kill 2>/dev/null
sleep 2

# 启动后端API (端口3443)
echo "启动后端API (端口3443)..."
cd /run/code/dinglei/pid
nohup python backend_api.py > backend.log 2>&1 &
echo "后端PID: \$\!"

# 等待后端启动
sleep 3

# 启动前端服务 (端口5273)
echo "启动前端服务 (端口5273)..."
cd /run/code/dinglei/pid/frontend
nohup python -m http.server 5273 > frontend.log 2>&1 &
echo "前端PID: \$\!"

# 等待服务启动
sleep 2

# 验证服务状态
echo ""
echo "验证服务状态..."
if netstat -tlnp 2>/dev/null | grep -q ":3443"; then
    echo "✓ 后端API运行正常 (端口3443)"
else
    echo "✗ 后端API启动失败"
fi

if netstat -tlnp 2>/dev/null | grep -q ":5273"; then
    echo "✓ 前端服务运行正常 (端口5273)"
else
    echo "✗ 前端服务启动失败"
fi

echo ""
echo "访问地址:"
echo "  前端界面: http://pid.dicp.sixseven.ltd:5924"
echo "  后端API:  http://pidend.dicp.sixseven.ltd:5924"
echo "  本地前端: http://192.168.3.202:5273"
echo "  本地后端: http://192.168.3.202:3443"
echo ""
echo "查看日志:"
echo "  后端: tail -f /run/code/dinglei/pid/backend.log"
echo "  前端: tail -f /run/code/dinglei/pid/frontend/frontend.log"
