import copy

import torch
from torch.xpu import device

from utils.softmax import masked_softmax,masked_gumbel_softmax
import torch.nn.functional as F
import numpy as np

def mask_generation(node_features, edge_index):
    # 掩码，根据要求去除一些边（将mask设为0）
    mask_1 = node_features[edge_index[0]] <= node_features[edge_index[1]]
    # mask_2 = node_features[edge_index[0]] == node_features[edge_index[1]]
    # mask = torch.logical_or(mask_1, mask_2)

    return mask_1

def param_to_adj(graph, param_mask, param, lora=None):
    n = graph.x.size()[0]
    center_node = graph.center_node


    edge_index = graph.edge_index
    device = graph.device
    mask = torch.zeros((n, n)).to(device)
    mask[center_node, center_node] = 1
    mask[edge_index[0, param_mask], edge_index[1, param_mask]] = 1
    mask[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = 1
    
    pred_adj = torch.zeros((n, n)).to(device)
    pred_adj[center_node, center_node] = param[0][0]
    pred_adj[edge_index[0, param_mask], edge_index[1, param_mask]] = param[1][param_mask]
    pred_adj[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = param[1][~param_mask]
    if lora!=None:
        lora=torch.mul(lora, mask)
        pred_adj+=lora

    pred_adj = masked_softmax(pred_adj, mask)
    return pred_adj

def param_to_adj_gumbel(graph, param_mask, param, lora=None):
    n = graph.x.size()[0]
    center_node = graph.center_node


    edge_index = graph.edge_index
    device = graph.device
    mask = torch.zeros((n, n)).to(device)
    mask[center_node, center_node] = 1
    mask[edge_index[0, param_mask], edge_index[1, param_mask]] = 1
    mask[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = 1
    
    pred_adj = torch.zeros((n, n)).to(device)
    pred_adj[center_node, center_node] = param[0][0]
    pred_adj[edge_index[0, param_mask], edge_index[1, param_mask]] = param[1][param_mask]
    pred_adj[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = param[1][~param_mask]
    if lora!=None:
        lora=torch.mul(lora, mask)
        pred_adj+=lora

    pred_adj = masked_gumbel_softmax(pred_adj, mask)
    return pred_adj
def param_to_adj_direct(graph, param_mask, param,lora=None):
    n = graph.x.size()[0]
    center_node = graph.center_node
    edge_index = graph.edge_index
    device = graph.device
    mask = torch.zeros((n, n)).to(device)
    mask[center_node, center_node] = 1
    mask[edge_index[0, param_mask], edge_index[1, param_mask]] = 1
    mask[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = 1
    pred_adj = torch.zeros((n, n)).to(device)
    pred_adj[center_node, center_node] = param[0][0]
    pred_adj[edge_index[0, param_mask], edge_index[1, param_mask]] = param[1][param_mask]
    pred_adj[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = param[1][~param_mask]

    if lora!=None:
        lora=torch.mul(lora, mask)
        pred_adj+=lora
    pred_adj = masked_softmax(pred_adj, mask)
    

    pred_adj = pred_adj.T
    _, max_indices = pred_adj.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(pred_adj).scatter_(-1, max_indices, 1.0)
    pred_adj = (y_hard - pred_adj).detach() + pred_adj
    pred_adj = pred_adj.T


    return pred_adj
def param_to_adj_work(graph, param_mask, param,lora=None):
    n = graph.x.size()[0]
    center_node = graph.center_node
    edge_index = graph.edge_index
    device = graph.device
    mask = torch.zeros((n, n)).to(device)
    mask[center_node, center_node] = 1
    mask[edge_index[0, param_mask], edge_index[1, param_mask]] = 1
    mask[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = 1
    pred_adj = torch.zeros((n, n)).to(device)
    pred_adj[center_node, center_node] = param[0][0]
    pred_adj[edge_index[0, param_mask], edge_index[1, param_mask]] = param[1][param_mask]
    pred_adj[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = param[1][~param_mask]

    if lora!=None:
        lora=torch.mul(lora, mask)
        pred_adj+=lora
    pred_adj = masked_softmax(pred_adj, mask)
    

    pred_adj = pred_adj.T
    _, max_indices = pred_adj.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(pred_adj).scatter_(-1, max_indices, 1.0)
    pred_adj = (y_hard - pred_adj).detach() + pred_adj
    pred_adj = pred_adj.T
    pred_adj = prune_directed_graph(pred_adj, graph.K.detach(), center_node)

    return pred_adj


def matrix_to_adj(graph, param_mask, matrix):
    n = graph.x.size()[0]
    center_node = graph.center_node


    edge_index = graph.edge_index
    device = graph.device
    mask = torch.zeros((n, n)).to(device)
    mask[center_node, center_node] = 1
    mask[edge_index[0, param_mask], edge_index[1, param_mask]] = 1
    mask[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = 1
    

    pred_adj = masked_softmax(matrix, mask)

    pred_adj = pred_adj.T
    _, max_indices = pred_adj.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(pred_adj).scatter_(-1, max_indices, 1.0)
    pred_adj = (y_hard - pred_adj).detach() + pred_adj
    pred_adj = pred_adj.T

    return pred_adj

def mask_to_adj(graph, mask):
    """
    将边序列表达的mask转为邻接矩阵的形式
    """
    n = graph.x.size()[0]
    device = graph.device
    edge_index = graph.edge_index
    matrix = torch.zeros((n, n), dtype=torch.float32).to(device)
    matrix[edge_index[0, mask], edge_index[1, mask]] = 1
    matrix[edge_index[1, ~mask], edge_index[0, ~mask]] = 1
    return matrix

def prune_directed_graph(adj_matrix, node_weights, center_node):
    n = len(node_weights)  # 节点数量

    in_degrees = torch.sum(adj_matrix!=0, dim=0)  # 入度

    # 找到初始的叶子节点
    stack = [i for i in range(n) if in_degrees[i] == 0 and node_weights[i] == 0]

    while stack:
        node = stack.pop()
        if node == center_node:
            continue
        for succ in torch.where(adj_matrix[node, :] > 0)[0]:  # 找到所有被 node 指向的后继节点
            adj_matrix[node, succ] = 0  # 移除边
            in_degrees[succ] -= 1  # 更新后继节点的入度
            if in_degrees[succ] <= 0 and node_weights[succ] == 0:
                stack.append(succ)

    # 返回剪枝后的邻接矩阵和点权
    return adj_matrix