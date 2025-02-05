import os
import json
import time
import math
import numpy as np
from LoRAPG import main
from RLAC import main as mainrl
from RLSAC import main as mainsac
from gumbel_softmax import main as mainsoftmax

import os
import shutil

def copy_matching_folders(base_path="output", gnum=1, destination="output/value_solver/1"):
    target_prefix = f"real_{gnum}"
    # 确保目标路径存在，如果不存在则创建
    os.makedirs(destination, exist_ok=True)

    # 遍历 output 文件夹下的一级子文件夹
    for dir_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, dir_name)
        print(dir_name, target_prefix)
        # 只处理以 target_prefix 开头的文件夹
        if dir_name.startswith(target_prefix) and os.path.isdir(folder_path):
            # 目标复制路径
            dest_path = os.path.join(destination, dir_name)

            # 复制文件夹
            try:
                shutil.move(folder_path, dest_path)
                print(f"Copied: {folder_path} -> {dest_path}")
            except Exception as e:
                print(f"Failed to copy {folder_path} to {dest_path}: {e}")

def calculate_statistics(data):
    if not data:
        return "The list is empty. Please provide valid data."

    mean = np.mean(data)
    maximum = np.max(data)
    minimum = np.min(data)
    variance = np.var(data)
    Coefficient_of_Variation = math.sqrt(variance) / mean
    return {
        "mean": mean,
        "max": maximum,
        "min": minimum,
        "Coefficient of Variation": Coefficient_of_Variation
    }

def collect_json_data(base_path="output", graph_num=1):
    # 初始化结果存储
    loss_data = []
    time_data = []

    # 确保目标路径存在
    if not os.path.exists(base_path):
        print(f"Base path '{base_path}' does not exist.")
        return loss_data, time_data

    # 遍历 output 文件夹下的所有一级子文件夹
    for dir_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, dir_name)

        # 只处理以 "real_1" 开头的文件夹
        if dir_name.startswith(f"real_{graph_num}") and os.path.isdir(folder_path):
            json_file_path = os.path.join(folder_path, "parameters.json")

            if os.path.exists(json_file_path):
                try:
                    # 读取 p.json 文件
                    with open(json_file_path, "r") as f:
                        data = json.load(f)
                    # 提取 total loss 和 total time
                    total_loss = data.get("total loss", "N/A")
                    total_time = data.get("total time", "N/A")
                    # 保存结果
                    loss_data.append(total_loss)
                    time_data.append(total_time)
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")
            else:
                print(f"File not found: {json_file_path}")

    return loss_data, time_data


if __name__ == "__main__":
    """
    这个接口比较假，只影响保存路径和循环次数，需要自己去experiment_gradient里先改参数
    跑之前要先把output里子图一样的清空，不然都会被算作本次测试的数据然后被拉进这个方法的output下，我是直接剪切过去的。
    """
    num_times = 5
    graph_num = 3
    type_name = "DG"  #本次测试的名称，如果使用的不是RLSAC\AC\PG，则会调用LoRAPG且只会改变存储数据的路径而不改变参数

    output_folder = f"output/{type_name}/{graph_num}"  # 目标路径
    for _ in range(0, num_times):
        if type_name == "RLSAC":
            mainsac(graph_num)  # 注意从哪里导入的main
        elif type_name == "AC":
            mainrl("AC", graph_num)
        elif type_name == "PG":
            mainrl("PG", graph_num)
        elif type_name == "GS":
            mainsoftmax("GS", graph_num)
        elif type_name == "DG":
            mainsoftmax("DG", graph_num)
        else:
            main(graph_num)


    copy_matching_folders(gnum=graph_num, destination=output_folder)
    loss_all, time_all = collect_json_data(output_folder, graph_num)
    print(loss_all)
    # 计算统计值
    stats = calculate_statistics(loss_all)
    time_avg = np.mean(time_all)
    print_text = {}

    result_string = (
        f"统计结果：\n"
        f"平均值: {stats['mean']}\n"
        f"最大值: {stats['max']}\n"
        f"最小值: {stats['min']}\n"
        f"变异系数: {stats['Coefficient of Variation']}\n"
        f"平均时间: {time_avg}"
    )

    print(result_string)

    with open(f"{output_folder}/result.txt", "w", encoding="utf-8") as f:
        f.write(result_string)


