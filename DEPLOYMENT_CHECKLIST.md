# 部署检查清单

## 部署前检查

### 环境检查
- [ ] Python 3.11+ 已安装
- [ ] pip 已安装并更新到最新版本
- [ ] 服务器内存 >= 8GB
- [ ] 磁盘空间 >= 10GB

### 依赖检查
```bash
# 检查Python版本
python --version

# 检查pip版本
pip --version

# 安装依赖
pip install -r requirements.txt
```

### 配置检查
- [ ] .env 文件已创建
- [ ] LLM_API_KEY 已配置
- [ ] LLM_BASE_URL 已配置
- [ ] 端口3443未被占用

```bash
# 检查端口占用
lsof -i :3443

# 如果被占用，杀死进程
kill -9 <PID>
```

## 部署步骤

### 1. 停止旧服务
```bash
./stop_services.sh
```

### 2. 备份当前版本
```bash
cd /run/code/dinglei
tar -czf pid_backup_$(date +%Y%m%d_%H%M%S).tar.gz pid/
```

### 3. 更新代码
```bash
# 如果从git拉取
git pull origin main

# 或者手动复制文件
```

### 4. 安装/更新依赖
```bash
pip install -r requirements.txt --upgrade
```

### 5. 启动服务
```bash
./start_services.sh
```

### 6. 验证服务
```bash
# 检查进程
ps aux | grep backend_api.py

# 检查日志
tail -f backend.log

# 测试API
curl http://localhost:3443/api/status
```

## 部署后验证

### 功能测试
- [ ] 前端页面可以访问
- [ ] 可以上传CSV文件
- [ ] 可以选择控制器品牌和策略
- [ ] 点击开始整定后有响应
- [ ] 可以看到智能体协作过程
- [ ] 可以看到最终结果

### 性能测试
- [ ] 响应时间 < 2秒（首字节）
- [ ] 整定完成时间 < 60秒
- [ ] 内存使用 < 2GB
- [ ] CPU使用 < 50%

### 日志检查
```bash
# 检查是否有错误
grep -i error backend.log

# 检查是否有警告
grep -i warning backend.log
```

## 回滚步骤

如果部署失败，执行以下步骤回滚：

### 1. 停止新服务
```bash
./stop_services.sh
```

### 2. 恢复备份
```bash
cd /run/code/dinglei
rm -rf pid/
tar -xzf pid_backup_YYYYMMDD_HHMMSS.tar.gz
```

### 3. 重启服务
```bash
cd pid
./start_services.sh
```

### 4. 验证回滚
```bash
curl http://localhost:3443/api/status
```

## 监控和维护

### 日常监控
```bash
# 每天检查日志大小
du -h backend.log

# 如果日志过大，轮转日志
mv backend.log backend.log.$(date +%Y%m%d)
touch backend.log
```

### 定期备份
```bash
# 每周备份一次
0 2 * * 0 cd /run/code/dinglei && tar -czf pid_backup_$(date +%Y%m%d).tar.gz pid/
```

### 性能监控
```bash
# 监控内存使用
ps aux | grep backend_api.py | awk {print }

# 监控CPU使用
top -p $(pgrep -f backend_api.py)
```

## 故障处理

### 服务无响应
1. 检查进程是否存在
2. 检查日志文件
3. 重启服务
4. 如果仍无响应，回滚到上一版本

### 内存泄漏
1. 监控内存使用趋势
2. 定期重启服务（每周一次）
3. 检查代码中的内存泄漏

### LLM调用失败
1. 检查API密钥是否过期
2. 检查网络连接
3. 检查LLM服务状态
4. 查看详细错误日志

## 安全检查

### 访问控制
- [ ] 只允许授权IP访问
- [ ] 使用HTTPS（如果暴露到公网）
- [ ] API密钥不在代码中硬编码

### 数据安全
- [ ] 上传的CSV文件定期清理
- [ ] 敏感数据不记录到日志
- [ ] 备份文件加密存储

## 联系方式

如遇到问题，请联系：
- 技术支持：[联系方式]
- 紧急联系：[联系方式]
