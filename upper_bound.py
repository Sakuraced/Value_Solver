import networkx as nx
import numpy as np
import random
import pickle

def generate_random_graph(n, p, seed=None):
    np_random_state = np.random.get_state()
    random_state = random.getstate()

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    G = nx.gnp_random_graph(n, p, seed=seed)
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 10)  # 边的权重为0到10之间的浮点数

    np.random.set_state(np_random_state)
    random.setstate(random_state)

    return G

def minimum_spanning_tree(G):
    # 使用Kruskal或Prim算法计算最小生成树
    MST = nx.minimum_spanning_tree(G, weight='weight')
    return MST

def shortest_path_sum(G, center_node):
    # 计算从center_node到所有其他节点的最短路径之和
    length = nx.single_source_dijkstra_path_length(G, center_node, weight='weight')
    mean_length = sum(length.values())
    return mean_length

# 示例
n = 200  # 节点数
p = 0.1  # 边的生成概率
seed = 44  # 随机数种子
subgraph_node=9
center_node=0
with open(f'data/subgraph_{subgraph_node}.gpickle', 'rb') as file:
        G = pickle.load(file)
# G = generate_random_graph(n, p, seed)
print(len(G.nodes))
num_edges_center=0
for u, v, weight in G.edges(data='weight'):
    if u==center_node or v== center_node:
        num_edges_center+=1
print(num_edges_center)
# 计算最小生成树
MST = minimum_spanning_tree(G)
MST_weight_sum=0
print("Minimum Spanning Tree edges with weights:")
for u, v, weight in MST.edges(data='weight'):
    MST_weight_sum+=weight
    # print(f"({u}, {v}) with weight {weight:.2f}")
print("MST weight:",MST_weight_sum)
shortest_path_mean=shortest_path_sum(MST, center_node)
print(f"SUM of shortest paths from node {center_node} to all other nodes in MST: {shortest_path_mean:.2f}")



# 计算最短路径之和
shortest_path_mean = shortest_path_sum(G, center_node)
print(f"SUM of shortest paths from node {center_node} to all other nodes: {shortest_path_mean:.2f}")

path_lengths = nx.single_source_dijkstra_path_length(G, center_node, weight='weight')
paths = nx.single_source_dijkstra_path(G, center_node, weight='weight')
# 计算每个最短路径需要的边和这些边的权重和
visited_edges = set()
total_weight = 0
for target_node, path in paths.items():
    if target_node != center_node:  # 排除中心节点

        edges_in_path = []
         # 用于记录已经访问过的边

        # 遍历路径中的相邻节点，提取边并计算权重
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_weight = G[u][v]['weight']

            # 确保边 (u, v) 只被计算一次，考虑有向图的情况
            edge = (u, v) if (u, v) not in visited_edges and (v, u) not in visited_edges else None
            if edge is not None:
                edges_in_path.append((u, v, edge_weight))
                total_weight += edge_weight
                visited_edges.add((u, v))  # 标记这条边为已访问
                visited_edges.add((u, v))
print(f"SUM of weight on SPT: {total_weight:.2f}")