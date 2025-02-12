import torch
import copy
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from scipy.sparse.csgraph import dijkstra, floyd_warshall
from scipy.sparse import csr_matrix
import numpy as np
import random
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn
def custom_loss(P, g, loss_args):
    """
        P是pred_adj
        g是原图
    """
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    unreached_weight = loss_args['unreached_weight']

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
    if loss_args["use_unreached"]:
        unreached = torch.matmul(K, g.x).sum()
        return 1/n*min_path * lamda, (1 - lamda) * nw_cons.sum(), unreached * unreached_weight
    else:
        return 1/n*min_path * lamda, (1 - lamda) * nw_cons.sum()

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
    not_reached_weight = loss_args['unreached_weight']
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
    unreached_weight = loss_args['unreached_weight']
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

    unreached = torch.matmul(K, g.x).sum()

    return  1/n*min_path * lamda, (1 - lamda) * nw_cons.sum(), unreached * unreached_weight


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
    unreached = torch.tensor(0)  # unreached不管哈哈哈
    return 1/n*min_path, nw.sum(), unreached


def AC_custom_loss(P, g, loss_args, batch_size=32):
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    not_reached_weight = loss_args['unreached_weight']
    P_transport = g.transport_adj
    P_construction = g.construction_adj
    device = g.device
    n=P.size()[0]
    K = g.K
    def process_tensor(K, batch_size):
        # 获取K的长度
        length = len(K)
        
        # 随机选择batch_size个元素的索引，允许重复
        selected_indices = torch.multinomial(torch.ones(length), batch_size, replacement=True)
        
        # 计算每个位置被选择的次数，返回一个与 K 相同长度的 tensor
        selection_counts = torch.bincount(selected_indices, minlength=length).to(K.device)
        
        # 将selection_counts与K做逐元素乘积
        result = K * selection_counts

        return result
    K=process_tensor(K, batch_size)
    nw_trans = torch.mul(P, P_transport)
    nw_cons = torch.mul(P, P_construction)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw_trans)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)
    return 1/batch_size*min_path * lamda, (1 - lamda) * nw_cons.sum()


def PG_custom_loss(P, g, loss_args, batch_size=32):
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    not_reached_weight = loss_args['unreached_weight']
    P_transport = g.transport_adj
    P_construction = g.construction_adj
    device = g.device
    n=P.size()[0]
    K = g.K
    def process_tensor(K, batch_size):
        # 获取K的长度
        length = len(K)
        
        # 随机选择batch_size个元素的索引，允许重复
        selected_indices = torch.multinomial(torch.ones(length), batch_size, replacement=True)
        
        # 计算每个位置被选择的次数，返回一个与 K 相同长度的 tensor
        selection_counts = torch.bincount(selected_indices, minlength=length).to(K.device)
        
        # 将selection_counts与K做逐元素乘积
        result = K * selection_counts

        return result
    K=process_tensor(K, batch_size)
    P = P.T
    _, max_indices = P.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(P).scatter_(-1, max_indices, 1.0)
    P = (y_hard - P).detach() + P
    P = P.T
    nw_trans = torch.mul(P, P_transport)
    nw_cons = torch.mul(P, P_construction)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw_trans)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)

    return 1/batch_size*min_path * lamda, (1 - lamda) * nw_cons.sum()


def PPO_custom_loss(P, g, loss_args, batch_size=32):
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    not_reached_weight = loss_args['unreached_weight']
    P_transport = g.transport_adj
    P_construction = g.construction_adj
    device = g.device
    n = P.size()[0]
    K = g.K

    def process_tensor(K, batch_size):
        # 获取K的长度
        length = len(K)

        # 随机选择batch_size个元素的索引，允许重复
        need_prob = F.normalize(g.K, p=1, dim=0).to(K.device)
        selected_indices = torch.multinomial(need_prob, batch_size, replacement=True)

        results = []
        for i in selected_indices:
            result = torch.zeros_like(K).to(K.device)
            result[i] = K[i]
            results.append(result)
        return results


    def critic(state, P):
        P = P.detach()
        nw_trans = torch.mul(P, P_transport)
        nw_cons = torch.mul(P, P_construction)
        nodes_weight = torch.matmul(torch.ones(n, device=device), nw_trans)
        min_path = 0
        for i in range(iterations):
            min_path += torch.dot(nodes_weight, state)
            state = torch.matmul(P, state)
        return min_path * lamda + (1 - lamda) * nw_cons.sum()

    def compute_gae(rewards, dones, values, next_values):
        gae_lambda = 0.95
        advantages = torch.zeros_like(rewards).to(rewards.device)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] +  next_values[t] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gae_lambda * next_non_terminal * last_gae
        return advantages

    Ks = process_tensor(K, batch_size)
    P_act = P.clone()
    P_act = P_act.T
    _, max_indices = P_act.max(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(P_act).scatter_(-1, max_indices, 1.0)
    P_act = (y_hard - P_act).detach() + P_act
    P_act = P_act.T
    nw_trans = torch.mul(P_act, P_transport)
    nw_cons = torch.mul(P_act, P_construction)
    nodes_weight = torch.matmul(torch.ones(n, device=device), nw_trans)
    rewards = []
    values = []
    dones = []
    next_values = []
    old_probs = []
    states = []
    actions = []
    for K in Ks:
        min_path = 0
        next_value = critic(K, P)
        for i in range(iterations):
            reward = torch.dot(nodes_weight, K)
            value = next_value
            state_id = torch.argmax(K)

            K = torch.matmul(P_act, K)

            next_value = critic(K, P)
            next_state_id = torch.argmax(K)
            done = 1 if g.center_node == next_state_id else 0

            prob = P[:, state_id]
            dist = Categorical(probs=prob)
            old_prob = dist.log_prob(next_state_id)

            old_probs.append(old_prob.unsqueeze(0))
            dones.append(done)
            rewards.append(reward.unsqueeze(0))
            values.append(value.unsqueeze(0))
            next_values.append(next_value.unsqueeze(0))
            states.append(state_id.unsqueeze(0))
            actions.append(next_state_id.unsqueeze(0))

            if done == 1:
                break
    rewards = torch.cat(rewards)
    actions = torch.cat(actions)
    old_probs = torch.cat(old_probs)
    states = torch.cat(states)
    advantages = compute_gae(rewards, dones, values, next_values)
    trajectories = (states, actions, advantages, old_probs)

    return trajectories, nw_cons.sum()


def SAC_custom_loss(P, g, loss_args, batch_size=32):
    lamda = loss_args['lamda']
    iterations = loss_args['loss_iterations']
    not_reached_weight = loss_args['unreached_weight']
    P_transport = g.transport_adj
    P_construction = g.construction_adj
    device = g.device
    n=P.size()[0]
    K = g.K
    def process_tensor(K, batch_size):
        # 获取K的长度
        length = len(K)
        
        # 随机选择batch_size个元素的索引，允许重复
        selected_indices = torch.multinomial(torch.ones(length), batch_size, replacement=True)
        
        # 计算每个位置被选择的次数，返回一个与 K 相同长度的 tensor
        selection_counts = torch.bincount(selected_indices, minlength=length).to(K.device)
        
        # 将selection_counts与K做逐元素乘积
        result = K * selection_counts

        return result
    K=process_tensor(K, batch_size)
    nw_trans = torch.mul(P, P_transport)
    nw_cons = torch.mul(P, P_construction)
    nodes_weight = torch.matmul(torch.ones(n,device=device), nw_trans)
    min_path=0
    for i in range(iterations):
        min_path += torch.dot(nodes_weight, K)
        K = torch.matmul(P, K)
    P = torch.clamp(P, min=1e-9)
    entropy = -torch.sum(P * torch.log(P), dim=0).sum()
    return 1/batch_size*(min_path) * lamda, (1 - lamda) * nw_cons.sum(), entropy


