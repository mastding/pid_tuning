#\!/bin/bash
# PID智能整定系统 - 服务停止脚本

echo "停止PID智能整定系统..."

# 停止后端
echo "停止后端API..."
ps aux | grep backend_api.py | grep -v grep | awk "{print \$2}" | xargs kill 2>/dev/null
if [ \$? -eq 0 ]; then
    echo "✓ 后端已停止"
else
    echo "- 后端未运行"
fi

# 停止前端
echo "停止前端服务..."
ps aux | grep "http.server 5273" | grep -v grep | awk "{print \$2}" | xargs kill 2>/dev/null
if [ \$? -eq 0 ]; then
    echo "✓ 前端已停止"
else
    echo "- 前端未运行"
fi

echo ""
echo "所有服务已停止"
