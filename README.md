# 网络节点优化

该项目旨在优化物流网络，以最小化节点运输成本与路径建设成本。输入一张图与中心节点，返回子树。

## 构建真实数据集

`data` 目录中的 `distance.csv` 存储了粤港澳大湾区的物流节点之间边的距离，其中编号为 0-10 的节点为中心节点，它们与任何节点之间都存在边。

运行 `data` 目录下的 `data_process.py` 以构建数据集。第一次运行时，请确保 `data` 目录底部有 `distance.csv` 文件，只需运行一次加载数据集。运行结束后，`data` 目录下会出现 `subgraph_n.gpickle` 文件，其中 `$n \in [0, 10]$`，后续无需再运行。

## Value_Solver算法

运行 `experiment_gradient.py` 以执行 `value_solver` 方法。

### 配置说明：

- **real_graph**：通过设置此变量来确定数据集是否调用真实数据集。
  - `real_graph == True`：设置 `subgraph`，$subgraph \in [0, 10]$ 来选定真实数据集的编号。
  - 在 `real_graph == True` 的条件下，设置 `n`、`p`、`seed` 变量来控制随机图的节点数、任意两节点间存在边的概率以及随机种子。

### 优化过程：

该优化过程一共分为三步。可以通过设置以下参数来控制每一步的优化方式：
- **lor**：确定第三步优化是否使用 `lora`。
- **not_reached_penalty**：确定第一步优化是否使用 `not_reached_penalty`。

### 运行结果：

运行完成后，在 `output` 目录下可以找到对应的文件夹，文件名中包含运行时间。文件夹中存储的文件包括：
- **parameters.json**：记录了运行的超参数、最终优化损失和运行时间。
- **training_log_i.csv**：存储了第 `i` 步优化的损失曲线。

## 其他测试方法

- **example.py**：其他相关的测试方法示例,将你的函数放置在other_methods目录底，并在`example.py`中引入你的优化函数，该程序将运行一定次数优化函数并计算优化结果的平均值与方差。

##  理论上界

运行`upper_bound.py`可以计算理论上界。
- **最小生成树**：执行最小生成树算法，路径建设成本最小化，再在最小生成树上使用最短路径算法计算运输成本。
- **最短路径**：执行最短路径算法，路径运输成本最小化，再在最小生成树上使用最小生成树算法计算路径建设成本。