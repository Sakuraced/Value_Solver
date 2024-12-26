import torch
import torch.optim as optim
from utils.loss import custom_loss_1, custom_loss_2, test_loss, new_test_loss
from utils.encode import mask_generation, param_to_adj,param_to_adj_work
from utils.prepro import generate_random_graph, generate_real_graph
from tqdm import tqdm
from datetime import datetime
import os
import json
import csv
import time
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 1000
    p = 0.01
    center_node = 0
    subgraph_node = 8
    seed = 44
    train_iterations_1 = 150
    train_iterations_2 = 150
    train_iterations_3 = 150
    lr = 0.1
    random_graph = False
    alpha = 1.0
    lor = True
    lora_rank = 2
    d = torch.tensor(lora_rank).to(device)
    not_reached_penalty = False #in first optimization
    if not_reached_penalty:
        train_iterations_2 += train_iterations_1
        train_iterations_1 = 0
    loss_args={'loss_iterations': 20, 'lamda': 0.05, 'not_reached_weight': 10}

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
    

    #初始化待优化参数
    cen_attr = torch.zeros(1).to(device).requires_grad_()
    edge_attr=Graph.edge_attr.to(device).requires_grad_()
    lora_Q=torch.randn(len(Graph.g.nodes()), lora_rank).to(device).requires_grad_()
    lora_K=torch.zeros(lora_rank, len(Graph.g.nodes())).to(device).requires_grad_()
    # 定义优化器Adam
    optimizer = optim.Adam([edge_attr, cen_attr,lora_Q,lora_K], lr=lr)
    # 掩码，去除一些边，使得边仅会由距中心节点较远的点指向较近的点
    mask = mask_generation(node_features=Graph.x, edge_index=Graph.edge_index).to(device)

    def dynamic_pruning_and_rewiring(param, pruning_ratio=0.5):
        with torch.no_grad():
            grad_norm = param.grad.abs()
            threshold = torch.quantile(grad_norm, pruning_ratio)
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

    start_time = time.time()
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
    end_time = time.time()
    s1_time = -start_time + end_time
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

    start_time = time.time()
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
    end_time = time.time()
    s2_time = -start_time + end_time
    '''
    第三阶段优化，
    先将待优化参数解码为对应矩阵+lora矩阵，
    再使用custom_loss_2损失函数
    '''
    csv_file = os.path.join(output_dir, "training_log_3.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "SPTC", "MSTC", "not_reached","loss"])
        writer.writeheader()

    progress_bar = tqdm(range(train_iterations_3), desc=f"Third Optimization", dynamic_ncols=True)

    start_time = time.time()
    for epoch in progress_bar:
        optimizer.zero_grad()
        lora_P=torch.mm(lora_Q,lora_K) / torch.sqrt(d) * alpha
        if not lor:
            lora_P=None
        pred_adj = param_to_adj(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr],lora=lora_P)
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
    end_time = time.time()
    s3_time = -start_time + end_time

    lora_P=torch.mm(lora_Q,lora_K) / torch.sqrt(d) * alpha
    if not lor:
        lora_P=None 
    # for i in range(len(pred_adj)):
    #     for j in range(len(pred_adj)):
    #         print(f'{pred_adj[i,j]:.1f}',end=' ')
    #     print()
    pred_adj = param_to_adj_work(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr], lora=lora_P)

    col_sums = torch.count_nonzero(pred_adj, dim=0)
    colsum=col_sums.sum()
    for i in range(1, n):
        if pred_adj[i,i]==1:
            colsum-=1
    print(colsum)
    SPT, MST, not_reached = test_loss(P=pred_adj, g=Graph, loss_args=loss_args)

    # num_nodes = pred_adj.size()[0]

    # def build_graph_optimized(adj_matrix):
    #     num_nodes = adj_matrix.size(0)

    #     edges = adj_matrix.nonzero()
    #     edges = edges[edges[:, 0] != edges[:, 1]] 
    #     graph = {i: [] for i in range(num_nodes)}
    #     for edge in edges:
    #         graph[edge[0].item()].append(edge[1].item())

    #     return graph

    # def dfs(graph, node, visited):
    #     visited.add(node)
    #     for neighbor in graph[node]:
    #         if neighbor not in visited:
    #             dfs(graph, neighbor, visited)

    # graph = build_graph_optimized(pred_adj)
    # visited = set()
    # dfs(graph, center_node, visited)
    # for node in range(pred_adj.size(0)):
    #     if node not in visited:
    #         print(f"节点 {node} 无法到达中心节点 {center_node}")
    #         for j in range(num_nodes):
    #             if pred_adj[j,node]==1:
    #                 print(j)

    loss = MST + SPT + not_reached
    print('_________________________________________________________________________________')
    print('SPT loss:', SPT.item(), ' MST loss:', MST.item(), ' not reached', not_reached.item(),
          ' total loss:', loss.item())
    


    # plot_graph(pred_adj)
    params = {
        "n": n,
        "p": p,
        "center_node": center_node,
        "subgraph_node": subgraph_node,
        "seed": seed,
        "Whether random_graph": random_graph,
        "train_iterations_1": train_iterations_1,
        "train_iterations_2": train_iterations_2,
        "train_iterations_3": train_iterations_3,
        "whether use not reached penalty in first stage": not_reached_penalty,
        "whether use lora in third stage": lor,
        "lora_rank":lora_rank,
        "lora_alpha":1.0,
        "lr": lr,
        'loss_iterations': loss_args['loss_iterations'],
        'lamda': loss_args['lamda'],
        'not_reached_weight': loss_args['not_reached_weight'],
        'number of nodes':len(Graph.g.nodes()),
        'number of edges,':len(Graph.g.edges()),
        'SPT loss':SPT.item(),
        'MST loss':MST.item(),
        'not reached':not_reached.item(),
        'total loss':loss.item(),
        'first optimization time': s1_time,
        'seconed optimization time': s2_time,
        'third optimization time': s3_time,
        'total time':s1_time+s2_time+s3_time
    }
    params_file = os.path.join(output_dir, "parameters.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

if __name__ == '__main__':
    main()