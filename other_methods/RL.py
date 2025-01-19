from sympy.strategies.branch import do_one
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from utils.prepro import calculate_node_features, Graph
from utils.encode import mask_to_adj
from utils.prepro import Graph
from utils.loss import custom_loss


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=bool)

    # def add(self, state, action, reward, next_state, done):
    #     """存储经验"""
    #     self.states[self.position] = state
    #     self.actions[self.position] = action
    #     self.rewards[self.position] = reward
    #     self.next_states[self.position] = next_state
    #     self.dones[self.position] = done
    #     self.position = (self.position + 1) % self.capacity
    #     self.size = min(self.size + 1, self.capacity)

    def add(self, states, actions, rewards, next_states, dones):
        """存单条或多条经验"""
        batch_size = len(states)
        # 计算插入的结束位置
        end_position = self.position + batch_size
        if end_position <= self.capacity:
            # 没有超出容量，直接插入
            self.states[self.position:end_position] = states
            self.actions[self.position:end_position] = actions
            self.rewards[self.position:end_position] = rewards
            self.next_states[self.position:end_position] = next_states
            self.dones[self.position:end_position] = dones
        else:
            # 超出容量，分成两段插入（循环缓冲区）
            overflow = end_position - self.capacity
            self.states[self.position:] = states[:batch_size - overflow]
            self.actions[self.position:] = actions[:batch_size - overflow]
            self.rewards[self.position:] = rewards[:batch_size - overflow]
            self.next_states[self.position:] = next_states[:batch_size - overflow]
            self.dones[self.position:] = dones[:batch_size - overflow]

            self.states[:overflow] = states[batch_size - overflow:]
            self.actions[:overflow] = actions[batch_size - overflow:]
            self.rewards[:overflow] = rewards[batch_size - overflow:]
            self.next_states[:overflow] = next_states[batch_size - overflow:]
            self.dones[:overflow] = dones[batch_size - overflow:]

        # 更新位置和有效数据大小
        self.position = end_position % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """随机采样"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (self.states[indices], self.actions[indices],
                self.rewards[indices], self.next_states[indices], self.dones[indices])

    def __len__(self):
        return self.size



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
        state是独热编码，这样可以直接和ava*pred乘一下得到一个向量，然后做softmax选对应的点（还是一个向量）
        然后拿值就是左右乘俩向量 1xnxnxnxnx1=1x1
        woc，太聪明了
    """
    def __init__(self, graph: Graph, mask):
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
        self.available_adj = torch.empty(0)
        self.mask = mask
        self.start_node = 0
        self.reset()

    def reset(self):
        start_id = torch.multinomial(self.node_need, num_samples=1, replacement=False)
        self.start_node = start_id
        one_hot = torch.zeros_like(self.node_need).to(self.device)
        one_hot[start_id] = 1
        self.state = one_hot
        self.available_adj = (self.adj != 0).to(dtype=torch.float32)
        self.available_adj[:start_id] = 0
        return self.state

    def step(self, action):
        state_id = torch.argmax(self.state)
        n = len(self.graph.nodes)
        mask = self.available_adj
        if self.mask is not None:
            mask = mask * self.mask
        ava = self.state.t() @ mask
        next_node_pre = action * ava
        count = (next_node_pre > 0).sum()
        if count <= 0:
            return self.state, -2*self.x[self.start_node], True
        next_node_pre[next_node_pre <= 0] = -torch.inf
        next_node_prob = torch.softmax(next_node_pre, dim=0).detach().cpu()
        next_state_id = torch.multinomial(next_node_prob, num_samples=1, replacement=False)
        s_ = torch.zeros_like(next_node_prob).to(self.device)
        s_[next_state_id] = 1
        self.available_adj[:, next_state_id] = 0
        reward = -self.construction[state_id][next_state_id]
        if next_state_id == self.center:
            done = True
            reward += self.x[self.start_node]
        else:
            done = False
        self.state = s_
        return s_, reward, done

    def get_loss(self, pred_adj, loss_args):
        spt, mst = custom_loss(P=pred_adj, g=self.ClassGrpah, loss_args=loss_args)
        spt = spt.cpu().detach().numpy()
        mst = mst.cpu().detach().numpy()
        return spt, mst


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim=32, fc2_dim=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)

        return x


# Define the reinforcement learning agent
class RLAgent:
    def __init__(self, graph, state_dim, action_dim, lr=0.01, mask=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.env = GraphEnv(graph, mask)
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=5000, state_dim=state_dim, action_dim=action_dim)
        self.batch_size = 32
        self.gamma = 0.9
        self.state_dim = state_dim

    def learn(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards).to(self.device)  # Move to GPU
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, episodes):
        progress_bar = tqdm(range(episodes), desc=f"RL Optimization", dynamic_ncols=True)
        for episode in progress_bar:
            state = self.env.reset()
            total_reward = 0
            state_list = []
            reward_list = []
            done_list = []
            next_state_list = []
            action_list = []

            while True:
                action = self.policy_net(state)
                next_state, reward, done = self.env.step(action)

                state_list.append(state.detach().cpu().numpy())
                next_state_list.append(next_state.detach().cpu().numpy())
                action_list.append(action.detach().cpu().numpy())
                reward_list.append(reward.detach().cpu().numpy())
                done_list.append([done])

                total_reward += reward
                if done:
                    break
                state = next_state

            pred_adj = self.policy_net(torch.eye(self.state_dim).to(self.device))
            loss_args = {'loss_iterations': 20, 'lamda': 0.1, 'unreached_weight': 0, "use_unreached": False}
            spt, mst = self.env.get_loss(pred_adj=pred_adj, loss_args=loss_args)
            reward_list -= mst / len(reward_list)
            # print(reward_list)
            # self.buffer.add(state_list, action_list, reward_list, next_state_list, done_list)
            self.learn(mst)
            progress_bar.set_postfix(Episode=episode + 1,
                                     Total_Reward = total_reward.item(),
                                     loss = "",
                                     refresh=True)

    def get_an_tree(self):
        state = self.env.reset()
        tree_edges = []
        while True:
            action_prob = 0
            action = self.policy_net(state)
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
    agent = RLAgent(graph, state_dim=num_nodes, action_dim=num_nodes, lr=0.01, mask=mask_adj)
    agent.train(episodes=num_episodes)

    final_edge = agent.get_an_tree()
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
