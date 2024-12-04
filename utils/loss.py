import torch
import copy
def custom_loss_1(P, Graph, loss_args):
    g=Graph
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    P_weight = g.weight_adj
    device = g.device
    n=P.size()[0]
    K = torch.ones(n,device=device)
    nw = torch.mul(P,P_weight)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)
        # if torch.all(K[1:] <= 0.1):
        #     break
    return min_path * lamda, (1 - lamda) * nw.sum()

def custom_loss_2(P, g, loss_args):
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    not_reached_weight = loss_args['not_reached_weight']
    P_weight = g.weight_adj
    device = g.device
    n=P.size()[0]
    K = torch.ones(n,device=device)
    nw = torch.mul(P,P_weight)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)
    not_reached = torch.matmul(K, g.x).sum()
    return min_path * lamda, (1 - lamda) * nw.sum(), not_reached * not_reached_weight

def test_loss(P, g):
    iterations = 100
    P_weight = g.weight_adj
    device = g.device
    n=P.size()[0]
    K = torch.ones(n,device=device)
    nw = torch.mul(P,P_weight)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)

    not_reached = torch.matmul(K, g.x).sum()

    return  min_path, nw.sum(),not_reached

