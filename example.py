import torch
from openpyxl.styles.builtins import output

from utils.loss import custom_loss_2, test_loss, new_test_loss
import torch
from utils.encode import mask_generation, matrix_to_adj
from utils.prepro import generate_random_graph, generate_real_graph
from other_methods.GA import genetic_algorithm
from other_methods.SA import simulated_annealing
from other_methods.BA import artificial_bee_colony
import time
import math
from other_methods.RL import rl_based_solve
import statistics
import os

 # 样本方差
def main():
    loss_args={'loss_iterations': 20, 'lamda': 0.1, 'unreached_weight': 10}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device= 'cpu'
    n = 1000
    p = 0.01
    center_node = 0
    seed = 44
    random_graph = False
    subgraph_node = 4
    test_epoch = 20
    pic_args = {'self_loop': False}
    method = "GA"  # use GA or SA
    output_folder = f'./output/{method}/{subgraph_node}'

    print('initializing graph...')
    if random_graph:
        Graph = generate_random_graph(n = n, p = p, seed = seed, center_node=center_node, device=device)
    else:
        Graph = generate_real_graph(subgraph_node=subgraph_node,center_node=center_node, device=device, args=pic_args)
    print('number of nodes:',len(Graph.g.nodes()))
    print('number of edges:',len(Graph.g.edges()))
    mask = mask_generation(node_features=Graph.x, edge_index=Graph.edge_index).to(device)
    SPT_list = []
    MST_list = []
    not_reached_list = []
    total_loss_list = []
    time_list = []
    
    
    for i in range(test_epoch):
        '''
        如下代码，请给出一棵覆盖所有节点的Graph的子树的邻接矩阵pred_adj
        同时最小化如下损失函数, 为了test_loss正常计算，请自行保证生成满足一棵树
        '''
        start = time.time()
        # begin your code
        if method == "GA":
            pred_adj = genetic_algorithm(graph=Graph,mask=mask) #torch(n,n)
        elif method == 'SA':
            pred_adj = simulated_annealing(graph=Graph, mask=mask)
        elif method == 'BA':
            pred_adj = artificial_bee_colony(graph=Graph, mask=mask)
        elif method == "RL":
            pred_adj = rl_based_solve(graph=Graph, num_episodes=100, mask=mask)
        else:
            print("ERROR: Method Not Implemented")
        # end your code
        end = time.time()
        time_length = end - start
        SPT, MST, not_reached = test_loss(P=pred_adj, g=Graph,loss_args=loss_args)
    
        loss = MST + SPT + not_reached
        SPT_list.append(SPT.item())
        MST_list .append(MST.item())
        not_reached_list.append(not_reached.item())
        total_loss_list.append(loss.item())
        time_list.append(time_length)
        print('_________________________________________________________________________________')
        result_i = (
            f"test_epoch: {i}\n"
            f"SPT loss: {SPT.item()}\n"
            f"MST loss: {MST.item()}\n"
            f"not reached: {not_reached.item()}\n"
            f"total loss: {loss.item()}\n"
            f"time: {time_length}"
        )
        print(result_i)
        path_i = f"{output_folder}/epoch{i}"
        if not os.path.exists(path_i):
            os.makedirs(path_i)
        with open(f"{path_i}/result.txt", "w", encoding="utf-8") as f:
            f.write(result_i)

        # print('test_epoch:',i,'SPT loss:', SPT.item(), ' MST loss:', MST.item(), ' not reached', not_reached.item(),
        #   ' total loss:', loss.item())
    print('_________________________________________________________________________________')
    result = (
        f"Avg SPT loss: {sum(SPT_list)/test_epoch}\n"
        f"Avg MST loss: {sum(MST_list)/test_epoch}\n"
        f"Avg not reached: {sum(not_reached_list)/test_epoch}\n"
        f"Avg total loss: {sum(total_loss_list)/test_epoch}\n"
        f"MAX loss: {max(total_loss_list)}\n"
        f"MIN loss: {min(total_loss_list)}\n"
        f"total loss cv: {math.sqrt(statistics.variance(total_loss_list))/(sum(total_loss_list)/test_epoch)}\n"
        f"avg time: {statistics.mean(time_list)}\n"
    )

    print(result)
    # print(f'Avg SPT loss:', sum(SPT_list)/test_epoch,
    #       f'Avg MST loss:', sum(MST_list)/test_epoch,
    #       f'Avg not reached', sum(not_reached_list)/test_epoch,
    #       f'Avg total loss:', sum(total_loss_list)/test_epoch,
    #       f'MAX loss:', max(total_loss_list),
    #       f'MIN loss:', min(total_loss_list),
    #       f'total loss cv:', math.sqrt(statistics.variance(total_loss_list))/(sum(total_loss_list)/test_epoch) )
    with open(f"{output_folder}/result.txt", "w", encoding="utf-8") as f:
        f.write(result)
if __name__ == '__main__':
    main()