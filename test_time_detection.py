import csv
import re

# 增强的时间解析函数，复制自前端app.js
def parseDateTime(value):
    # 尝试直接解析
    try:
        from datetime import datetime
        d = datetime.strptime(value, '%Y/%m/%d %H:%M:%S')
        return d
    except ValueError:
        pass
    
    # 尝试其他格式
    try:
        from datetime import datetime
        d = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        return d
    except ValueError:
        pass
    
    # 处理其他格式
    try:
        from datetime import datetime
        d = datetime.strptime(value, '%Y/%m/%d %H:%M')
        return d
    except ValueError:
        pass
    
    try:
        from datetime import datetime
        d = datetime.strptime(value, '%Y-%m-%d %H:%M')
        return d
    except ValueError:
        pass
    
    return None

# 移除BOM字符
def remove_bom(text):
    if text.startswith('\ufeff'):
        return text[1:]
    return text

# 测试时间列检测
def test_time_column_detection(csv_path):
    print(f"测试文件: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # 移除BOM字符
        headers = [remove_bom(h) for h in headers]
        print(f"列名: {headers}")
        
        # 自动检测时间列
        best_time_column_index = -1
        best_time_count = 0
        
        # 尝试解析每一列，找到最可能是时间列的列
        for col_index, header in enumerate(headers):
            time_values = []
            valid_time_count = 0
            
            # 重置reader
            f.seek(0)
            next(reader)  # 跳过表头
            
            # 最多检查前200行
            for i, row in enumerate(reader):
                if i >= 200:
                    break
                if len(row) > col_index:
                    value = row[col_index].strip()
                    if value:
                        # 尝试解析时间
                        d = parseDateTime(value)
                        if d:
                            valid_time_count += 1
                            time_values.append(value)
            
            # 选择时间值最多的列
            if valid_time_count > best_time_count:
                best_time_count = valid_time_count
                best_time_column_index = col_index
        
        if best_time_column_index == -1:
            print("未找到时间列")
            return
        
        print(f"检测到时间列: {headers[best_time_column_index]}")
        print(f"有效时间值数量: {best_time_count}")
        
        # 解析所有时间值，找出时间范围
        timestamps = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) > best_time_column_index:
                    ts = row[best_time_column_index].strip()
                    if ts:
                        d = parseDateTime(ts)
                        if d:
                            timestamps.append(d)
        
        if timestamps:
            timestamps.sort()
            start_time = timestamps[0]
            end_time = timestamps[-1]
            print(f"时间范围: {start_time} 至 {end_time}")
        else:
            print("未找到有效时间值")

# 测试文件
csv_path = r"D:\code\pid_tuning\backend\state\task_artifacts\task_1775979362866_13\uploaded__1_part1.csv"
test_time_column_detection(csv_path)
