import torch
import copy
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from scipy.sparse.csgraph import dijkstra, floyd_warshall
from scipy.sparse import csr_matrix
import numpy as np

def custom_loss_1(P, Graph, loss_args):
    """
    P是pred_adj
    Graph是原图
    """
    g=Graph
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    P_transport = g.transport_adj
    P_construction = g.construction_adj
    device = g.device
    n=P.size()[0]
    K = g.K
    nw_trans = torch.mul(P,P_transport)
    nw_cons = torch.mul(P,P_construction)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw_trans)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)
        # if torch.all(K[1:] <= 0.1):
        #     break
    return 1/n*min_path * lamda, (1 - lamda) * nw_cons.sum()

def custom_loss_2(P, g, loss_args):
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    not_reached_weight = loss_args['not_reached_weight']
    P_transport = g.transport_adj
    P_construction = g.construction_adj
    device = g.device
    n=P.size()[0]
    K = g.K
    nw_trans = torch.mul(P, P_transport)
    nw_cons = torch.mul(P, P_construction)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw_trans)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)
    not_reached = torch.matmul(K, g.x).sum()
    return 1/n*min_path * lamda, (1 - lamda) * nw_cons.sum(), not_reached * not_reached_weight

def test_loss(P, g, loss_args):
    lamda = loss_args['lamda']
    not_reached_weight = loss_args['not_reached_weight']
    iterations = 100
    P_transport = g.transport_adj
    P_construction = g.construction_adj
    device = g.device
    n=P.size()[0]
    K = g.K
    nw_trans = torch.mul(P, P_transport)
    nw_cons = torch.mul(P, P_construction)
    nodes_weight = torch.matmul(torch.ones(n, device=device), nw_trans)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)

    not_reached = torch.matmul(K, g.x).sum()

    return  1/n*min_path * lamda, (1 - lamda) * nw_cons.sum(), not_reached * not_reached_weight


def new_test_loss(P, g):
    P_weight = g.weight_adj
    device = g.device
    n = P.size()[0]
    P = P.detach()

    print_no_edge = True  # 可以选择是否报有没有节点漏了
    if print_no_edge:
        for i in range(n):
            if torch.all(P[i] == 0) and torch.all(P[: i] == 0):
                print(f"节点{i}不具有任何边相连")

    nw = torch.mul(P, P_weight)

    print_to_csv = False  # 可以选择把你的P或者nw打印出来
    if print_to_csv:
        csv_name = "pred_adj.csv"
        df = pd.DataFrame(nw.cpu().numpy())
        df.to_csv(csv_name, index=False, header=False)

    np_nw = torch.where(nw == 0, torch.tensor(float('inf')), nw).cpu().numpy()
    graph_sparse = csr_matrix(np_nw)
    dist_matrix = dijkstra(graph_sparse, return_predecessors=False, indices=g.center_node)
    min_path = np.sum(dist_matrix) - dist_matrix[g.center_node]
    # print(dist_matrix)
    # print(min_path)
    not_reached = torch.tensor(0)  # not_reached不管哈哈哈
    return 1/n*min_path, nw.sum(), not_reached

