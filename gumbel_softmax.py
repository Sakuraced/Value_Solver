import torch
import torch.optim as optim
from utils.loss import *
from utils.encode import mask_generation, param_to_adj_gumbel, param_to_adj_work, param_to_adj_direct
from utils.prepro import generate_random_graph, generate_real_graph
from tqdm import tqdm
from datetime import datetime
import os
import json
import csv
import time
def main(method_type="GS", subgraph_node = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 1000
    p = 0.01
    center_node = 0
    if subgraph_node is None:
        subgraph_node = 4
    seed = 44
    train_iterations = [150, 150, 150]
    lr = 0.1
    random_graph = False
    alpha = 1.0
    lora_rank = 6
    d = torch.tensor(lora_rank).to(device)
    use_lora = [False, False, False]
    use_penalty = [False, False, False]
    loss_args={'loss_iterations': 200, 'lamda': 0.1, 'unreached_weight': 10, "use_unreached":False}
    pic_args={'self_loop':False}

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
        Graph = generate_real_graph(subgraph_node=subgraph_node,center_node=center_node, device=device, args=pic_args)
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
    三段优化，
    先将待优化参数解码为对应矩阵， 
    再使用custom_loss损失函数
    '''
    time_list = []
    for step_num in range(3):
        csv_file = os.path.join(output_dir, f"training_log_{step_num+1}.csv")
        field_list = ["epoch", "SPTC", "MSTC", "unreached","loss"]
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=field_list)
            writer.writeheader()

        progress_bar=tqdm(range(train_iterations[step_num]), desc=f"Optimization Step {step_num+1}", dynamic_ncols=True)
        start_time = time.time()
        for epoch in progress_bar:
            lora_P = None
            optimizer.zero_grad()
            if method_type == "GS":
                pred_adj = param_to_adj_gumbel(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr],lora=lora_P)
            elif method_type == "DG":
                pred_adj = param_to_adj_direct(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr],lora=lora_P)
            else:
                print("WRONG METHOD NAME")
                exit(0)
            loss_args["use_unreached"] = False
            SPT, MST = custom_loss_1(P=pred_adj, Graph=Graph, loss_args=loss_args)
            loss = MST + SPT

            unreached_str = f"{unreached:.4f}" if use_penalty[step_num] else "None"
            progress_bar.set_postfix(SPTC=f"{SPT:.4f}",
                                     MSTC=f"{MST:.4f}",
                                     Unreached=unreached_str,
                                     loss=f"{loss:.4f}",
                                     refresh=True)
            loss.backward()
            dynamic_pruning_and_rewiring(edge_attr)
            optimizer.step()
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=field_list)
                writer.writerow({
                    "epoch": epoch,
                    "SPTC": f"{SPT:.4f}",
                    "MSTC": f"{MST:.4f}",
                    "unreached":unreached_str,
                    "loss": f"{loss:.4f}"
                })
        end_time = time.time()
        time_list.append(end_time - start_time)

    #test
    if use_lora[2]:
        lora_P = torch.mm(lora_Q, lora_K) / torch.sqrt(d) * alpha
    else:
        lora_P=None
    pred_adj = param_to_adj_work(graph=Graph, param_mask=mask, param=[cen_attr, edge_attr], lora=lora_P)
    col_sums = torch.count_nonzero(pred_adj, dim=0)
    colsum=col_sums.sum()
    for i in range(1, len(Graph.g.nodes())):
        if pred_adj[i,i]==1:
            colsum-=1
    print(colsum)
    SPT, MST, unreached = test_loss(P=pred_adj, g=Graph, loss_args=loss_args)
    loss = MST + SPT + unreached
    print('_________________________________________________________________________________')
    print('SPT loss:', SPT.item(), ' MST loss:', MST.item(), ' not reached', unreached.item(),
          ' total loss:', loss.item())



    # plot_graph(pred_adj)
    params = {
        "n": n,
        "p": p,
        "center_node": center_node,
        "subgraph_node": subgraph_node,
        "seed": seed,
        "Whether random_graph": random_graph,
        "train_iterations_1": train_iterations[0],
        "train_iterations_2": train_iterations[1],
        "train_iterations_3": train_iterations[2],
        "whether use unreached penalty in each stage": use_penalty,
        "whether use lora in each stage": use_lora,
        "lora_rank":lora_rank,
        "lora_alpha":1.0,
        "lr": lr,
        'loss_iterations': loss_args['loss_iterations'],
        'lamda': loss_args['lamda'],
        'unreached_weight': loss_args['unreached_weight'],
        'number of nodes':len(Graph.g.nodes()),
        'number of edges,':len(Graph.g.edges()),
        'SPT loss':SPT.item(),
        'MST loss':MST.item(),
        'not reached':unreached.item(),
        'total loss':loss.item(),
        'first optimization time': time_list[0],
        'seconed optimization time': time_list[1],
        'third optimization time': time_list[2],
        'total time':sum(time_list),
    }
    params_file = os.path.join(output_dir, "parameters.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

if __name__ == '__main__':
    main()