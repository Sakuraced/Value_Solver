from sympy.strategies.branch import do_one
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import copy

from transformers.models.prophetnet.modeling_prophetnet import softmax

from utils.prepro import calculate_node_features, Graph
from utils.encode import mask_to_adj
from utils.prepro import Graph
from utils.loss import custom_loss
from utils.softmax import masked_softmax

# Define the graph environment
class GraphEnv:
    """
        graph 原图
        center 中心点
        adj_matrix 已经选中的邻接矩阵
        selected_edges 已经选中的边集
        selected_nodes 已经选中的点集
        current_cost 目前的总开销
        device 加载的运算设备
        输出动作概率是个nxn的矩阵pred
        state是独热编码
    """
    def __init__(self, graph: Graph):
        self.ClassGrpah = graph
        self.x = graph.x
        self.device = graph.device
        self.graph = graph.g
        self.center = graph.center_node
        self.node_need = F.normalize(graph.K, p=1, dim=0).to(self.device)
        self.construction = graph.construction_adj
        self.transport = graph.transport_adj
        self.adj = graph.adj
        self.state = torch.empty(0)
        self.start_node = 0
        self.reset()

    def reset(self):
        start_id = torch.multinomial(self.node_need, num_samples=1, replacement=False)
        self.start_node = start_id
        one_hot = torch.zeros_like(self.node_need).to(self.device)
        one_hot[start_id] = 1
        self.state = one_hot
        return self.state

    def step(self, action):
        # if action == -1:
        #     return self.state, -2 * self.x[self.start_node], True
        state_id = torch.argmax(self.state)
        n = len(self.graph.nodes)
        next_state_id = action
        s_ = torch.zeros_like(self.state).to(self.device)
        s_[next_state_id] = 1
        reward = -self.construction[next_state_id][state_id]
        if next_state_id == self.center:
            done = True
            reward += self.x[self.start_node].item() * 2
        else:
            done = False
        self.state = s_
        return s_, reward, done

    def get_loss(self, pred_adj, loss_args):
        spt, mst = custom_loss(P=pred_adj, g=self.ClassGrpah, loss_args=loss_args)
        return spt, mst


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, mask, device):
        super(PolicyNetwork, self).__init__()
        print(device)
        self.eye = torch.eye(state_dim).to(device).detach()
        self.fc1 = nn.Linear(state_dim, action_dim)
        self.mask = mask
    def forward(self, state):
        output = self.eye.clone()
        weight = self.fc1(output)
        if self.mask is not None:
            w = masked_softmax(E=weight, B=self.mask)
        else:
            w = F.softmax(weight, dim=-1)
        x = state
        x = x @ w.T
        return x


# Define the reinforcement learning agent
class RLAgent:
    def __init__(self, graph, state_dim, action_dim, lr=0.01, mask=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.env = GraphEnv(graph)
        self.policy_net = PolicyNetwork(state_dim, action_dim, mask, device).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.batch_size = 32
        self.gamma = 0.9
        self.state_dim = state_dim
        self.epochs = 4
        self.epsilon = 0.2

    def update(self, trajectories):
        states, actions, rewards, old_probs = trajectories
        # Compute advantages
        advantages = self.compute_advantages(rewards)

        for _ in range(self.epochs):
            probs = self.policy_net(states)
            dist = Categorical(probs)
            new_probs = dist.log_prob(actions)

            # Compute ratio
            ratio = (new_probs - old_probs).exp()

            # Clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_advantages(self, rewards):
        advantages = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            advantages.insert(0, discounted_sum)
        pred_adj = self.policy_net(torch.eye(self.state_dim).to(self.device))
        loss_args = {'loss_iterations': 20, 'lamda': 0.1, 'unreached_weight': 0, "use_unreached": False}
        spt, mst = self.env.get_loss(pred_adj=pred_adj, loss_args=loss_args)
        weight_mst = 1
        pred_adj -= weight_mst * mst / len(pred_adj)
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def train(self, episodes):
        progress_bar = tqdm(range(episodes), desc=f"RL Optimization", dynamic_ncols=True)
        for episode in progress_bar:
            state = self.env.reset()
            total_reward = 0
            states = []
            rewards = []
            old_probs = []
            actions = []

            while True:
                prob = self.policy_net(state)
                dist = Categorical(probs=prob)
                action = dist.sample()
                next_state, reward, done = self.env.step(action)
                states.append(state.detach())
                actions.append(action.detach())
                rewards.append(reward.detach())
                old_probs.append(dist.log_prob(action))

                total_reward += reward
                if done:
                    break
                state = next_state

            # print(reward_list)
            # self.buffer.add(state_list, action_list, reward_list, next_state_list, done_list)
            states = torch.stack(states).to(self.device).detach()
            actions = torch.stack(actions).to(self.device).detach()
            old_probs = torch.stack(old_probs).to(self.device).detach()
            trajectories = (states, actions, rewards, old_probs)
            self.update(trajectories)
            progress_bar.set_postfix(Episode=episode + 1,
                                     Total_Reward = total_reward.item(),
                                     loss = "",
                                     refresh=True)

    def get_all_prob(self):
        self.env.reset()
        prob = self.policy_net(torch.eye(self.state_dim).to(self.device)).detach()
        return prob

def rl_based_solve(graph, num_episodes, mask=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = graph.g
    if mask is not None:
        mask_adj = mask_to_adj(graph=graph, mask=mask)
    else:
        mask_adj = None
    num_nodes = len(g.nodes)
    agent = RLAgent(graph, state_dim=num_nodes, action_dim=num_nodes, lr=0.01, mask=mask_adj, device=device)
    agent.train(episodes=num_episodes)

    pred_adj = agent.get_all_prob()
    max_values, max_indices = torch.max(pred_adj, dim=1, keepdim=True)
    # 创建一个与输入张量形状相同的零张量
    result = torch.zeros_like(pred_adj)
    # 如果该行最大值大于 0，则将最大值位置设为 1
    result.scatter_(1, max_indices, 1)
    # 保留全零行
    result[pred_adj.sum(dim=1) == 0] = 0
    return result.T

# Generate a random graph
def generate_graph(n, weight_range=(1, 100)):
    # 创建完全图
    g = nx.complete_graph(n)  # 完全图结构
    # 为每条边添加随机权重
    for u, v in g.edges():
        g[u][v]['weight'] = random.randint(*weight_range)  # 随机权值
    return g

if __name__ == "__main__":
    num_nodes = 10
    center_node = 0
    G = generate_graph(num_nodes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_features = calculate_node_features(G, center_node)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    # 直接生成邻接矩阵
    adj_matrix = torch.tensor(nx.to_numpy_array(G, nodelist=sorted(G.nodes), weight='weight'), dtype=torch.float32).to(
        device)
    # 提取边属性
    edge_attr = torch.tensor([data['weight'] for _, _, data in G.edges(data=True)], dtype=torch.float)
    edge_attr = torch.exp(-edge_attr)

    adj0_matrix = copy.deepcopy(adj_matrix)
    large_number = 1e5
    adj_matrix[adj_matrix == 0] = large_number

    adj_matrix[center_node, center_node] = 0.0
    adj0_matrix[center_node, center_node] = 1

    graph = Graph(edge_index=edge_index, x=x, center_node=center_node, adj=adj0_matrix, weight_adj=adj_matrix,
                  edge_attr=edge_attr, g=G, device=device)
    # 获取最终的树
    final_tree = rl_based_solve(graph, num_episodes=100)
    print("Final Tree Edges:", final_tree)

    # 可视化最终的树
    # tree_graph = nx.Graph()
    # tree_graph.add_edges_from(final_tree)
    # pos = nx.spring_layout(tree_graph)
    # nx.draw(tree_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    # import matplotlib.pyplot as plt
    # plt.show()
