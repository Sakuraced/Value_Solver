from mpmath.libmp import str_to_man_exp
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.style.core import available
from torch_geometric.nn import GCNConv
import random
from utils.prepro import calculate_node_features, Graph
from utils.encode import mask_to_adj
import copy

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
    """
    def __init__(self, graph, mask):
        self.graph = graph.g
        self.center = graph.center_node
        self.adj_matrix = None
        self.reset()
        self.selected_edges = None
        self.selected_nodes = {self.center}
        self.current_cost = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask = mask

    def reset(self):
        # 初始化邻接矩阵，未选中的边设为0
        self.adj_matrix = np.zeros((len(self.graph.nodes), len(self.graph.nodes)))
        self.selected_edges = set()
        self.selected_nodes = set()

        self.current_cost = 0
        return self.get_state()

    def get_state(self):
        # 返回当前邻接矩阵作为状态
        return self.adj_matrix

    def step(self, action):
        # 动作是一个边 (u, v, weight)
        u, v, weight = action
        self.selected_edges.add((u, v))
        self.adj_matrix[u, v] = weight
        self.selected_nodes.update({u, v})
        self.current_cost += weight
        reward = -weight

        done = len(self.selected_nodes) == len(self.graph.nodes)
        return self.get_state(), reward, done

    def get_available_actions(self):
        n = len(self.graph.nodes)
        adj_ava = torch.zeros((n, n), dtype=torch.float32).to(self.device)
        available_node = [k for k in range(len(self.graph.nodes)) if k not in self.selected_nodes]
        for i in self.selected_nodes:
            for j in available_node:
                if self.graph.has_edge(i, j):
                    adj_ava[i, j] = self.graph[i][j].get('weight', None)
        if self.mask is not None:
            adj_ava = adj_ava * self.mask
        have_actions = len(self.selected_nodes) < len(self.graph.nodes)
        return adj_ava, have_actions

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(PolicyNetwork, self).__init__()
        output_dim = input_dim
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)

    def forward(self, node_features, edge_index):
        # 图卷积层提取节点特征
        x = self.gcn1(node_features, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)
        return x

class PolicyNetwork2(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim=128, fc2_dim=128):
        super(PolicyNetwork2, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.prob = nn.Linear(fc2_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        prob = torch.softmax(self.prob(x), dim=-1)

        return prob


# Define the reinforcement learning agent
class RLAgent:
    def __init__(self, graph, center, input_dim, hidden_dim, lr=0.01, mask=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = GraphEnv(graph, center, mask=mask)
        self.policy_net = PolicyNetwork2(input_dim, input_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state, available_actions):
        adj_matrix = torch.tensor(state, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(list(self.env.graph.edges), dtype=torch.long).t().contiguous().to(self.device)
        # 计算邻接矩阵分数
        # adj_scores = self.policy_net(adj_matrix, edge_index)
        s = torch.tensor(state, dtype=torch.float32).to(self.device)
        adj_scores = self.policy_net(s)
        adj_act = torch.where(available_actions == 0, torch.tensor(float('-inf')).to(self.device), adj_scores)
        # 计算概率分布并采样动作
        probs = torch.softmax(adj_act.flatten(), dim=0)
        select_index = torch.argmax(probs).item()
        self.log_probs.append(probs[select_index])
        x, y = divmod(select_index, adj_act.size(1))
        selected_action = (x, y, available_actions[x][y])
        return selected_action

    def update_policy(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards).to(self.device)  # Move to GPU
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        for log_prob, reward in zip(self.log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    def train(self, episodes):
        progress_bar = tqdm(range(episodes), desc=f"RL Optimization", dynamic_ncols=True)
        for episode in progress_bar:
            state = self.env.reset()
            total_reward = 0

            while True:
                available_actions, have_actions = self.env.get_available_actions()
                if not have_actions:
                    break
                action = self.select_action(state, available_actions)
                next_state, reward, done = self.env.step(action)
                self.rewards.append(reward)
                total_reward += reward

                if done:
                    break
                state = next_state

            self.update_policy()
            progress_bar.set_postfix(Episode=episode + 1,
                                     Total_Reward = total_reward.item(),
                                     loss = "",
                                     refresh=True)
            # print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    def get_final_edge(self):
        state = self.env.reset()
        tree_edges = []
        while True:
            available_actions, have_actions = self.env.get_available_actions()
            if not have_actions:
                break
            action = self.select_action(state, available_actions)
            tree_edges.append((action[0], action[1]))  # Record the edge (u, v)
            _, _, done = self.env.step(action)
            if done:
                break
        return tree_edges

def rl_based_solve(graph, mask, num_episodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = graph.g
    mask_adj = mask_to_adj(graph=graph, mask=mask)
    num_nodes = len(g.nodes)
    center_node = graph.center_node
    agent = RLAgent(graph, center=center_node, input_dim=num_nodes, hidden_dim=16, lr=0.01, mask=mask_adj)
    agent.train(episodes=num_episodes)

    final_edge = agent.get_final_edge()
    final_tree = torch.zeros((num_nodes, num_nodes)).to(device)
    for u, v in final_edge:
        final_tree[u, v] = 1
    return final_tree

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
