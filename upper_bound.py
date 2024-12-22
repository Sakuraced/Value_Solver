import networkx as nx
import numpy as np
import random
import pickle
from collections import defaultdict
import sys

from sympy.printing.pretty.pretty_symbology import center


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

class UBA:
    def __init__(self, G, center_node):
        self.G = G
        self.center_node = center_node
        self.shortest_path_list = nx.single_source_dijkstra_path_length(G, center_node, weight='weight')
        self.remain_path_sum = sum(self.shortest_path_list.values())
        self.adj_list = defaultdict(list)
        self.selected = np.zeros(len(G.nodes), dtype=bool)
        self.ans = np.inf
        self.now_ans = 0
        self.now_path_list = np.zeros(len(G.nodes))

    def upper_bound_all(self):
        for u, v, weight in self.G.edges(data='weight'):
            self.adj_list[u].append((v, weight))
            self.adj_list[v].append((u, weight))
        self.selected[self.center_node] = True
        self.find_answer(self.center_node)
        return self.ans

    def find_answer(self, now_node):
        if self.selected.all():
            self.ans = min(self.ans, self.now_ans)
            return
        if self.now_ans > self.ans:
            return
        e_ans = self.now_ans + self.remain_path_sum
        if e_ans > self.ans:
            return
        for next_node, weight in self.adj_list[now_node]:
            if not self.selected[next_node]:
                temp_now_ans = self.now_ans
                self.selected[next_node] = True
                self.now_path_list[next_node] = self.now_path_list[now_node] + weight
                self.remain_path_sum -= self.shortest_path_list[next_node]
                self.now_ans += weight + self.now_path_list[next_node]

                self.find_answer(next_node)

                self.selected[next_node] = False
                self.remain_path_sum += self.shortest_path_list[next_node]
                self.now_ans = temp_now_ans

def main():
    # 示例
    n = 200  # 节点数
    p = 0.1  # 边的生成概率
    seed = 44  # 随机数种子
    subgraph_node=0
    center_node=0
    cal_all = False
    with open(f'data/subgraph_{subgraph_node}.gpickle', 'rb') as file:
            G = pickle.load(file)
    # G = generate_random_graph(n, p, seed)
    print(len(G.nodes))
    num_edges_center=0
    for u, v, weight in G.edges(data='weight'):
        if u==center_node or v== center_node:
            num_edges_center+=1
    print(num_edges_center)

    if cal_all:
        # print(sys.getrecursionlimit())  看递归深度用的
        ALL = UBA(G, center_node)
        uba_ans = ALL.upper_bound_all()
        print(f"upper bound of the sum of both: {uba_ans}")

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

if __name__ == "__main__":
    main()