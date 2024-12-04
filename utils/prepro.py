import copy
from tqdm import tqdm
import numpy as np
import random
import networkx as nx
import torch
import pandas as pd
import pickle

class Graph:
    def __init__(self, edge_index, x, center_node=None, adj=None, weight_adj=None, edge_attr=None,g=None,device='cpu'):
        self.device=device
        self.edge_index = edge_index.to(device)
        self.x = x.to(device)
        self.edge_attr = edge_attr.to(device)
        self.adj=adj.to(device)
        self.weight_adj=weight_adj.to(device)
        self.normalized_weight_adj=torch.exp(-(self.weight_adj-torch.diag((torch.diag(self.weight_adj))))).to(device)
        self.g=g
        self.center_node=center_node
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
        print("weight_adj:", self.weight_adj.size())
        print(self.weight_adj)
        print("normalized_weight_adj:", self.normalized_weight_adj.size())
        print(self.normalized_weight_adj)

def calculate_node_features(G, center_node):
    length = nx.single_source_dijkstra_path_length(G, center_node, weight='weight')
    features = {node: length[node] for node in G.nodes()}
    features = np.array([features[node] for node in range(len(G.nodes()))])
    return features

def generate_random_graph(n, p, center_node=0, seed=None,device='cpu'):
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

    adj_matrix = nx.to_numpy_matrix(G)

    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
    adj0_matrix = copy.deepcopy(adj_matrix)
    large_number = 1e8
    adj_matrix[adj_matrix == 0] = large_number


    adj_matrix[center_node, center_node] = 0.0
    adj0_matrix[center_node, center_node] = 1
    graph = Graph(edge_index=edge_index, x=x, center_node=center_node, adj=adj0_matrix, weight_adj=adj_matrix,
                  edge_attr=edge_attr, g=G, device=device)

    return graph

def generate_real_graph(subgraph_node=0, center_node=0,device='cpu'):
    with open(f'data/subgraph_{subgraph_node}.gpickle', 'rb') as file:
        G = pickle.load(file)
    node_features = calculate_node_features(G, center_node)
    # for i in range(len(G.nodes)):
    #     print(node_features[i])
    # node_features_array = np.array([node_features[node] for node in G.nodes()])

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    edge_attr = []
    for u, v, data in G.edges(data=True):
        edge_attr.append(data['weight'])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_attr=torch.exp(-edge_attr)

    adj_matrix = torch.zeros((len(G.nodes),len(G.nodes))).to(device)
    for u, v, data in G.edges(data=True):
        adj_matrix[u, v]=data['weight']
        adj_matrix[v, u]=data['weight']

    adj0_matrix = copy.deepcopy(adj_matrix)
    large_number = 1e5
    adj_matrix[adj_matrix == 0] = large_number

    adj_matrix[center_node, center_node] = 0.0
    adj0_matrix[center_node, center_node] = 1


    graph = Graph(edge_index=edge_index, x=x, center_node=center_node, adj=adj0_matrix, weight_adj=adj_matrix,
                  edge_attr=edge_attr, g=G, device=device)

    return graph

