import torch
import torch.optim as optim
from utils.loss import custom_loss_1, custom_loss_2, test_loss
from utils.encode import mask_generation, param_to_adj,param_to_adj_work
from utils.prepro import generate_random_graph, generate_real_graph
from tqdm import tqdm
from datetime import datetime
import os
import json
import csv

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 1000
    p = 0.01
    center_node = 0
    subgraph_node = 3   #5,7,9,10
    seed = 44
    train_iterations_1 = 300
    train_iterations_2 = 300
    lr = 0.1
    random_graph = False
    loss_args={'loss_iterations': 20, 'lamda': 0.5, 'not_reached_weight': 10}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if random_graph:
        folder_name = f"random"+f"{seed}_{timestamp}"
    else:
        folder_name = f"real_"+f"{subgraph_node}_{timestamp}"

    
    # 生成图
    print('initializing graph...')
    if random_graph:
        Graph = generate_random_graph(n = n, p = p, seed = seed, center_node=center_node, device=device)
    else:
        Graph = generate_real_graph(subgraph_node=subgraph_node,center_node=center_node, device=device)
    print('number of nodes:',len(Graph.g.nodes()))
    print('number of edges:',len(Graph.g.edges()))

    output_dir = os.path.join("output", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    params = {
        "n": n,
        "p": p,
        "center_node": center_node,
        "subgraph_node": subgraph_node,
        "seed": seed,
        "train_iterations_1": train_iterations_1,
        "train_iterations_2": train_iterations_2,
        "lr": lr,
        "random_graph": random_graph,
        'loss_iterations': loss_args['loss_iterations'],
        'lamda': loss_args['lamda'],
        'not_reached_weight': loss_args['not_reached_weight'],
        'number of nodes':len(Graph.g.nodes()),
        'number of edges,':len(Graph.g.edges())
    }
    params_file = os.path.join(output_dir, "parameters.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

    #初始化待优化参数
    cen_attr = torch.zeros(1).to(device).requires_grad_()
    edge_attr=Graph.edge_attr.to(device).requires_grad_()
    # 定义优化器Adam
    optimizer = optim.Adam([edge_attr, cen_attr], lr=lr)
    # 掩码，去除一些边，使得边仅会由距中心节点较远的点指向较近的点
    mask = mask_generation(node_features=Graph.x, edge_index=Graph.edge_index).to(device)

    def dynamic_pruning_and_rewiring(param, pruning_ratio=0.5):
        with torch.no_grad():
            # 计算权重的绝对值
            grad_norm = param.grad.abs()
            # 确定剪枝阈值
            threshold = torch.quantile(grad_norm, pruning_ratio)
            # 创建剪枝掩码
            mask = grad_norm > threshold
            param.grad *= mask.float()

    '''
    第一阶段优化，
    先将待优化参数解码为对应矩阵，
    再使用custom_loss_1损失函数
    '''

    csv_file = os.path.join(output_dir, "training_log_1.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "SPTC", "MSTC", "loss"])
        writer.writeheader()

    progress_bar=tqdm(range(train_iterations_1), desc=f"First Optimization", dynamic_ncols=True)
    for epoch in progress_bar:
        optimizer.zero_grad()
        pred_adj = param_to_adj(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr])
        SPT, MST = custom_loss_1(P=pred_adj, Graph=Graph,loss_args=loss_args)
        loss = MST + SPT
        progress_bar.set_postfix(SPTC=f"{SPT:.4f}",
                                 MSTC=f"{MST:.4f}",
                                 loss=f"{loss:.4f}",
                                 refresh=True)
        loss.backward()
        dynamic_pruning_and_rewiring(edge_attr)
        optimizer.step()
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "SPTC", "MSTC","loss"])
            writer.writerow({
                "epoch": epoch,
                "SPTC": f"{SPT:.4f}",
                "MSTC": f"{MST:.4f}",
                "loss": f"{loss:.4f}"
            })

    '''
    第二阶段优化，
    先将待优化参数解码为对应矩阵，
    再使用custom_loss_2损失函数
    '''
    csv_file = os.path.join(output_dir, "training_log_2.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "SPTC", "MSTC", "not_reached","loss"])
        writer.writeheader()

    progress_bar = tqdm(range(train_iterations_2), desc=f"Second Optimization", dynamic_ncols=True)
    for epoch in progress_bar:
        optimizer.zero_grad()
        pred_adj = param_to_adj(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr])
        SPT, MST, not_reached = custom_loss_2(P=pred_adj, g=Graph,loss_args=loss_args)
        loss = MST + SPT + not_reached
        progress_bar.set_postfix(SPTC=f"{SPT:.4f}",
                                 MSTC=f"{MST:.4f}",
                                 Not_Reached=f"{not_reached:.4f}",
                                 loss=f"{loss:.4f}",
                                 refresh=True)
        loss.backward()
        dynamic_pruning_and_rewiring(edge_attr)
        optimizer.step()
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "SPTC", "MSTC", "not_reached", "loss"])
            writer.writerow({
                "epoch": epoch,
                "SPTC": f"{SPT:.4f}",
                "MSTC": f"{MST:.4f}",
                "not_reached":f"{not_reached:.4f}",
                "loss": f"{loss:.4f}"
            })


    pred_adj = param_to_adj_work(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr])

    col_sums = torch.count_nonzero(pred_adj, dim=0)
    print(col_sums.sum())
    SPT, MST, not_reached = test_loss(P=pred_adj, g=Graph)


    # 获取邻接矩阵的大小
    num_nodes = pred_adj.size()[0]

    def build_graph_optimized(adj_matrix):
        num_nodes = adj_matrix.size(0)

        # 使用非零元素获取所有边的起点和终点索引
        edges = adj_matrix.nonzero()

        # 去除自环（i != j） cqwsdc dswc cvdswd
        edges = edges[edges[:, 0] != edges[:, 1]]  # 排除(i, i)的情况

        # 构建图的邻接表
        graph = {i: [] for i in range(num_nodes)}
        for edge in edges:
            graph[edge[0].item()].append(edge[1].item())

        return graph

    # 深度优先搜索（DFS）遍历图，找到可达节点
    def dfs(graph, node, visited):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(graph, neighbor, visited)

    # 构建图
    graph = build_graph_optimized(pred_adj)

    # 用于记录从中心节点可达的所有节点
    visited = set()

    # 从中心节点开始DFS遍历
    dfs(graph, center_node, visited)

    # 检查每个节点与中心节点的连通性
    for node in range(pred_adj.size(0)):
        if node not in visited:
            print(f"节点 {node} 无法到达中心节点 {center_node}")
            for j in range(num_nodes):
                if pred_adj[j,node]==1:
                    print(j)

    loss = MST + SPT + not_reached
    print('_________________________________________________________________________________')
    print('SPT loss:', SPT.item(), ' MST loss:', MST.item(), ' not reached', not_reached.item(),
          ' total loss:', loss.item())

    # plot_graph(pred_adj)

if __name__ == '__main__':
    main()