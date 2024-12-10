import copy

import torch
from utils.softmax import masked_softmax
import torch.nn.functional as F

def mask_generation(node_features, edge_index):
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
    
    # for i in range(n):
    #     print(f'{mask[i,0].item():.1f}')
    pred_adj = torch.zeros((n, n)).to(device)
    pred_adj[center_node, center_node] = param[0][0]
    pred_adj[edge_index[0, param_mask], edge_index[1, param_mask]] = param[1][param_mask]
    pred_adj[edge_index[1, ~param_mask], edge_index[0, ~param_mask]] = param[1][~param_mask]
    if lora!=None:
        lora=torch.mul(lora, mask)
        pred_adj+=lora

    pred_adj = masked_softmax(pred_adj, mask)
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