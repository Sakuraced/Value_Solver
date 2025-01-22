import copy

from numpy import dtype
from tqdm import tqdm
import numpy as np
import random
import networkx as nx
import torch
import pandas as pd
import pickle
class Graph:
    """
    The Graph class
    edge_index(2 * num_of_edge): the node index of every edge in the graph, the n_th edge is (edge_index[0][n], edge_index[1][n])
    x(num_of_nodes): shortest path lengths from the given center_node to all other nodes in the graph
    adj(num_of_nodes, num_of_nodes): distance adjacency matrix of the graph
    construction_adj(num_of_nodes, num_of_nodes): construction cost adjacency matrix of the graph
    transport_adj(num_of_nodes, num_of_nodes): transport cost adjacency matrix of the graph
    edge_attr(num_of_edges): normalized edge attributes of the graph
    g: the graph stored as nx.Graph
    K(num_of_nodes): the need of nodes
    """
    device: torch.device
    edge_index: torch.Tensor
    adj: torch.Tensor
    construction_adj: torch.Tensor
    transport_adj: torch.Tensor
    edge_attr: torch.Tensor
    g: nx.Graph
    K: torch.Tensor

    def __init__(self, edge_index, x, center_node=None, adj=None, transport_adj=None, construction_adj=None ,edge_attr=None,g=None,device='cpu'):
        self.device=device
        self.edge_index = edge_index.to(device)
        self.x = x.to(device)
        self.edge_attr = edge_attr.to(device)
        self.adj=adj.to(device)
        self.construction_adj=construction_adj.to(device)
        self.transport_adj=transport_adj.to(device)
        self.g=g
        self.center_node=center_node
        self.K=torch.tensor([data["weight"] for _, data in g.nodes(data=True)], dtype=torch.float32).to(device)

    def degrees(self):
        source_nodes = self.edge_index[0]
        degrees = torch.bincount(source_nodes, minlength=self.x.size(0))
        return degrees
    def show(self):
        print("edge_index:",self.edge_index.size())
        print(self.edge_index)
        print("x:",self.x.size())
        print(self.x)
        print("edge_attr:", self.edge_attr.size())
        print(self.edge_attr)
        print("adj:", self.adj.size())
        print(self.adj)
        print("construction_adj:", self.construction_adj.size())
        print(self.construction_adj)
        print("transport_adj:", self.transport_adj.size())
        print(self.transport_adj)
        print(f"K{self.K.size()}")
        print(self.K)

def calculate_node_features(G, center_node):
    length = nx.single_source_dijkstra_path_length(G, center_node, weight='weight')
    print(len(G.nodes()))
    features = {node: length[node] for node in G.nodes()}
    features = np.array([features[node] for node in range(len(G.nodes()))])
    return features

def generate_random_graph(n, p, center_node=0, seed=None,device='cpu'):
    """
    生成一个随机图像
    """
    np_random_state = np.random.get_state()
    random_state = random.getstate()
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    G = nx.gnp_random_graph(n, p, seed=seed)
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 10)
    np.random.set_state(np_random_state)
    random.setstate(random_state)
    node_features = calculate_node_features(G, center_node)
    # node_features_array = np.array([node_features[node] for node in G.nodes()])
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    edge_attr = []
    for u, v, data in G.edges(data=True):
        edge_attr.append(data['weight'])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_attr=torch.exp(-edge_attr)

    adj_matrix = nx.to_numpy_array(G)

    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
    adj0_matrix = copy.deepcopy(adj_matrix)
    large_number = 1e8
    adj_matrix[adj_matrix == 0] = large_number


    adj_matrix[center_node, center_node] = 0.0
    adj0_matrix[center_node, center_node] = 1
    graph = Graph(edge_index=edge_index, x=x, center_node=center_node, adj=adj0_matrix, weight_adj=adj_matrix,
                  edge_attr=edge_attr, g=G, device=device)

    return graph

def generate_real_graph(subgraph_node=0, center_node=0,device='cpu', args=None):
    """
    从文件中加载图
    subgraph_node:图编号
    center_node:中心节点编号
    """
    with open(f'data/subgraph_{subgraph_node}.gpickle', 'rb') as file:
        G = pickle.load(file)
    node_features = calculate_node_features(G, center_node)
    K=torch.tensor([data["weight"] for _, data in G.nodes(data=True)], dtype=torch.float32).to(device)
    # for i in range(len(G.nodes)):
    #     print(node_features[i])
    # node_features_array = np.array([node_features[node] for node in G.nodes()])
    large_number = 1e5
    self_edge_list = [(i, i, {"weight":1e-10, "construction": 0, "transport":large_number})
                 for i in range(len(G.nodes))if K[i]==0]

    if args["self_loop"]:
        G.add_edges_from(self_edge_list)

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)

    # 提取边属性
    edge_attr = torch.tensor([data['weight'] for _, _, data in G.edges(data=True)], dtype=torch.float)
    edge_attr = torch.exp(-edge_attr)

    # 直接生成邻接矩阵
    adj_matrix = torch.tensor(nx.to_numpy_array(G,nodelist=sorted(G.nodes), weight='weight'), dtype=torch.float32).to(device)
    construction_adj = torch.tensor(nx.to_numpy_array(G, nodelist=sorted(G.nodes), weight='construction'), dtype=torch.float32).to(device)
    transport_adj = torch.tensor(nx.to_numpy_array(G, nodelist=sorted(G.nodes), weight='transport'), dtype=torch.float32).to(device)

    adj_matrix[center_node, center_node] = 1
    
    transport_adj[adj_matrix == 0] = large_number
    transport_adj[center_node, center_node] = 0.0
    construction_adj[adj_matrix == 0] = large_number
    construction_adj[center_node, center_node] = 0.0


    graph = Graph(edge_index=edge_index, x=x, center_node=center_node, adj=adj_matrix,
                  construction_adj=construction_adj, transport_adj=transport_adj,
                  edge_attr=edge_attr, g=G, device=device)

    return graph


def generate_complete_graph(n, weight_range=(1, 100)):
    # 创建随机权值的完全图
    g = nx.complete_graph(n)  # 完全图结构
    # 为每条边添加随机权重
    for u, v in g.edges():
        g[u][v]['weight'] = random.randint(*weight_range)  # 随机权值
    return g

