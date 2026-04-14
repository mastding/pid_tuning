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
    
    # 读取CSV文件
    raw_df = pd.read_csv(test_csv_path)
    print(f"\n读取的CSV文件列名: {list(raw_df.columns)}")
    
    # 实现自动检测时间列的函数
    def detect_time_column(df):
        if df is None or len(df) == 0:
            return None
        
        time_column = None
        max_valid_count = 0
        
        for col in df.columns:
            try:
                # 尝试将列解析为 datetime
                parsed = pd.to_datetime(df[col], errors='coerce')
                valid_count = parsed.notna().sum()
                
                if valid_count > max_valid_count and valid_count > len(df) * 0.5:  # 至少50%的数据是有效时间
                    max_valid_count = valid_count
                    time_column = col
            except Exception:
                continue
        
        return time_column
    
    # 测试自动检测时间列
    time_column = detect_time_column(raw_df)
    print(f"\n自动检测到的时间列: {time_column}")
    
    if time_column:
        print(f"时间列数据类型: {raw_df[time_column].dtype}")
        print(f"时间列前5行: {list(raw_df[time_column].head())}")
        
        # 测试转换为datetime
        try:
            raw_df[time_column] = pd.to_datetime(raw_df[time_column])
            print(f"\n转换为datetime后的数据类型: {raw_df[time_column].dtype}")
            print(f"转换后时间列前5行: {list(raw_df[time_column].head())}")
            
            # 测试时间范围
            min_time = raw_df[time_column].min()
            max_time = raw_df[time_column].max()
            print(f"\n时间范围: {min_time} 至 {max_time}")
        except Exception as e:
            print(f"\n转换时间列失败: {e}")
    else:
        print("\n未检测到时间列")
        
    # 测试时间范围切分
    if time_column:
        raw_df[time_column] = pd.to_datetime(raw_df[time_column])
        
        start_dt = pd.to_datetime("2025-09-25 00:00:00")
        end_dt = pd.to_datetime("2025-09-26 00:00:00")
        
        mask = (raw_df[time_column] >= start_dt) & (raw_df[time_column] <= end_dt)
        sliced_df = raw_df[mask].copy()
        
        print(f"\n时间范围内的数据行数: {len(sliced_df)}")
        if len(sliced_df) > 0:
            print(f"切分后数据的时间范围: {sliced_df[time_column].min()} 至 {sliced_df[time_column].max()}")
        else:
            print("切分后没有数据")
        
finally:
    # 清理测试文件
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
        print(f"\n测试文件已清理: {test_csv_path}")
