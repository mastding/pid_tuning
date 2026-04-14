import pandas as pd
import numpy as np
import tempfile
import os

# 模拟数据1_part1.csv的结构
data = {
    'Test': pd.date_range('2025-09-25 00:00:00', '2025-11-09 23:59:55', periods=200).strftime('%Y/%m/%d %H:%M:%S'),
    'PV': np.random.rand(200) * 100,
    'SV': np.random.rand(200) * 100,
    'MV': np.random.rand(200) * 100
}

df = pd.DataFrame(data)

# 保存为测试CSV文件
with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
    df.to_csv(tmp_file.name, index=False, encoding='utf-8')
    test_csv_path = tmp_file.name

try:
    print(f"测试文件路径: {test_csv_path}")
    print(f"文件内容预览:\n{df.head()}")
    
    # 测试1: 读取CSV文件
    from backend.services.data_service import _detect_time_column
    
    # 读取CSV文件
    raw_df = pd.read_csv(test_csv_path)
    print(f"\n读取的CSV文件列名: {list(raw_df.columns)}")
    
    # 测试2: 自动检测时间列
    time_column = _detect_time_column(raw_df)
    print(f"\n自动检测到的时间列: {time_column}")
    
    if time_column:
        print(f"时间列数据类型: {raw_df[time_column].dtype}")
        print(f"时间列前5行: {list(raw_df[time_column].head())}")
        
        # 测试3: 转换为datetime
        try:
            raw_df[time_column] = pd.to_datetime(raw_df[time_column])
            print(f"\n转换为datetime后的数据类型: {raw_df[time_column].dtype}")
            print(f"转换后时间列前5行: {list(raw_df[time_column].head())}")
            
            # 测试4: 时间范围
            min_time = raw_df[time_column].min()
            max_time = raw_df[time_column].max()
            print(f"\n时间范围: {min_time} 至 {max_time}")
        except Exception as e:
            print(f"\n转换时间列失败: {e}")
    else:
        print("\n未检测到时间列")
        
    # 测试5: 测试load_pid_dataset函数
    from backend.services.data_service import load_pid_dataset
    
    try:
        dataset = load_pid_dataset(test_csv_path)
        print(f"\nload_pid_dataset成功")
        print(f"清洗后的数据行数: {len(dataset['cleaned_df'])}")
        print(f"清洗后的列名: {list(dataset['cleaned_df'].columns)}")
        
        # 测试6: 测试slice_csv_by_time端点
        from backend.api.tune_app import slice_csv_by_time
        from fastapi.testclient import TestClient
        from fastapi import FastAPI, File, UploadFile, Form
        import uvicorn
        
        app = FastAPI()
        app.post("/api/tuning/csv/slice-by-time")(slice_csv_by_time)
        
        client = TestClient(app)
        
        with open(test_csv_path, 'rb') as f:
            response = client.post(
                "/api/tuning/csv/slice-by-time",
                files={"file": ("test.csv", f, "text/csv")},
                data={
                    "start_time": "2025-09-25 00:00:00",
                    "end_time": "2025-09-26 00:00:00"
                }
            )
        
        print(f"\nslice_csv_by_time响应状态码: {response.status_code}")
        print(f"slice_csv_by_time响应内容: {response.json()}")
        
    except Exception as e:
        print(f"\nload_pid_dataset失败: {e}")
        import traceback
        traceback.print_exc()
        
finally:
    # 清理测试文件
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
        print(f"\n测试文件已清理: {test_csv_path}")
