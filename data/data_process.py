from tqdm import tqdm
import numpy as np
import networkx as nx
import pandas as pd
import pickle

#  路径长度到建设成本的函数
def construction_cost(length):
    return length*10

#  路径长度到运输成本的函数
def transport_cost(length):
    d = length
    e_tot = 7200000.0
    r_max = 28.0
    e_takeoff = 114000.0
    n_cycle = 1500.0
    c_battery = 10000.0
    c_fly = 0.085
    c_truck = 1.45
    e_perkm = e_tot / r_max
    if d <= 28:
        c_tot = (c_fly*d) + (np.ceil((e_takeoff + e_perkm*d) / e_tot) * c_battery / n_cycle)
    else:
        c_tot = (4 - (d-28)/(300-28)) * c_truck*d
    return c_tot

def generate_real_graph(center_nodes=[0,1,2,3,4,5,6,7,8,9,10]):
    df = pd.read_csv('data/distance.csv', header=None)
    df2 = pd.read_csv("data/node.csv",encoding = "gbk")
    print('building graph')

    num_nodes = df2["id"].max()
    G = nx.Graph()

    # 构造点
    nodes = [(int(row["id"])-1, {"weight":row["need"]}) for _, row in df2.iterrows()]
    G.add_nodes_from(nodes)

    # 构造边
    edges = df.to_numpy()
    # 减去1以调整索引（假设节点从1开始）
    edges[:, 0] -= 1  # 将 node1 减去 1
    edges[:, 1] -= 1  # 将 node2 减去 1

    # 构造边的列表，其中每个元组 (node1, node2, {"weight":distance, "construction": 建设成本, "transport": 运输成本})
    edge_list = [(int(node1), int(node2), {"weight":dist, "construction": construction_cost(dist), "transport":transport_cost(dist)})
                 for node1, node2, dist in edges]

    # 批量添加边
    G.add_edges_from(edge_list)

    # 对每个中心节点生成子图
    for center in tqdm(center_nodes,desc='building subgraph'):
        subgraph_nodes = [center]
        for node in range(num_nodes):
            if node != center:
                # 计算当前节点到所有中心节点的距离
                distances = [G[node][c]['weight'] if G.has_edge(node, c) else np.inf for c in center_nodes]
                # 找到距离最小的中心节点
                closest_center = center_nodes[np.argmin(distances)]
                if closest_center == center:
                    subgraph_nodes.append(node)

        # 从原图中提取子图
        subgraph = G.subgraph(subgraph_nodes).copy()
        subgraph.remove_nodes_from(list(nx.isolates(subgraph)))  # 节点有了权重之后提取子图默认不删独立点，得帮它删了
        nodes = list(subgraph.nodes())

        # 确保中心节点在图中
        if center not in nodes:
            raise ValueError(f"center_node {center} 不在子图中")

        # 将中心节点放在第一个，其余节点按顺序排列
        ordered_nodes = [center] + [node for node in nodes if node != center]

        # 创建新的编号映射
        mapping = {node: idx for idx, node in enumerate(ordered_nodes)}
        # 使用networkx的relabel_nodes进行重标
        relabeled_subgraph = nx.relabel_nodes(subgraph, mapping)

        # 用来输出到文件看看点的权值继承对不对
        # nod_sub = subgraph.nodes(data=True)
        # nod_relabel = relabeled_subgraph.nodes(data=True)
        # print(mapping)
        # with open("output_raw.txt", "w", encoding="utf-8") as f:
        #     for node, attrs in nod_sub:
        #         f.write(f"{node}: {attrs}\n")
        # with open("output_relabel.txt", "w", encoding="utf-8") as f:
        #     for node, attrs in nod_relabel:
        #         f.write(f"{node}: {attrs}\n")

        # subgraph = nx.relabel_nodes(subgraph, {node: idx for idx, node in enumerate(subgraph.nodes())})
        # print(subgraph.nodes())

        # print(relabeled_subgraph.nodes())
        with open(f'data/subgraph_{center}.gpickle', "wb") as f:
            pickle.dump(relabeled_subgraph, f)
        # nx.write_gpickle(relabeled_subgraph, f'subgraph_{center}.gpickle')

    return
if __name__ == '__main__':
    generate_real_graph()
