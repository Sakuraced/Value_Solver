from tqdm import tqdm
import numpy as np
import networkx as nx
import pandas as pd
import pickle
def generate_real_graph(center_nodes=[0,1,2,3,4,5,6,7,8,9,10]):
    df = pd.read_csv('distance.csv', header=None)
    num_nodes = df[0].max()
    G = nx.Graph()
    # 构造边的列表
    edges = df.to_numpy()  # 转换为NumPy数组，避免逐行处理
    print('building graph')
    # 减去1以调整索引（假设节点从1开始）
    edges[:, 0] -= 1  # 将 node1 减去 1
    edges[:, 1] -= 1  # 将 node2 减去 1

    # 构造边的列表，其中每个元组 (node1, node2, {'weight': distance})
    edge_list = [(int(node1), int(node2), {'weight': dist}) for node1, node2, dist in edges]

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
        nodes = list(subgraph.nodes())

        # 确保中心节点在图中
        if center not in nodes:
            raise ValueError(f"center_node {center} 不在子图中")

        # 将中心节点放在第一个，其余节点按顺序排列
        ordered_nodes = [center] + [node for node in nodes if node != center]

        # 创建新的编号映射
        mapping = {node: idx for idx, node in enumerate(ordered_nodes)}
        # print(mapping)
        # 使用networkx的relabel_nodes进行重标
        relabeled_subgraph = nx.relabel_nodes(subgraph, mapping)
        # subgraph = nx.relabel_nodes(subgraph, {node: idx for idx, node in enumerate(subgraph.nodes())})
        # print(subgraph.nodes())

        # print(relabeled_subgraph.nodes())
        with open(f'subgraph_{center}.gpickle', "wb") as f:
            pickle.dump(relabeled_subgraph, f)
        # nx.write_gpickle(relabeled_subgraph, f'subgraph_{center}.gpickle')

    return
if __name__ == '__main__':
    generate_real_graph()
