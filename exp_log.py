import numpy as np

def calculate_statistics(data):
    if not data:
        return "The list is empty. Please provide valid data."
    
    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)
    variance = np.var(data)
    Coefficient_of_Variation=variance/mean
    return {
        "mean": mean,
        "max": maximum,
        "min": minimum,
        "Coefficient of Variation": Coefficient_of_Variation
    }

# 示例列表
data = [116732.3671875, 116773.765625, 116745.1484375, 116749.015625]

# 计算统计值
stats = calculate_statistics(data)
print("统计结果：")
print(f"平均值: {stats['mean']}")
print(f"最大值: {stats['max']}")
print(f"最小值: {stats['min']}")
print(f"变异系数: {stats['Coefficient of Variation']}")
