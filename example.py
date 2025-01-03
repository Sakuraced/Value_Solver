import torch
from utils.loss import custom_loss_2, test_loss, new_test_loss
import torch
from utils.encode import mask_generation, matrix_to_adj
from utils.prepro import generate_random_graph, generate_real_graph
from other_methods.GA import genetic_algorithm
from other_methods.SA import simulated_annealing
# from other_methods.RL import rl_based_solve
import statistics

 # 样本方差
def main():
    loss_args={'loss_iterations': 20, 'lamda': 0.1, 'not_reached_weight': 10}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 1000
    p = 0.01
    center_node = 0
    seed = 44
    random_graph = False
    subgraph_node = 1
    test_epoch = 20
    pic_args = {'self_loop': True}

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
    
    
    for i in range(test_epoch):
        '''
        如下代码，请给出一棵覆盖所有节点的Graph的子树的邻接矩阵pred_adj
        同时最小化如下损失函数, 为了test_loss正常计算，请自行保证生成满足一棵树
        '''
        # begin your code

        pred_adj = simulated_annealing(graph=Graph,mask=mask) #torch(n,n)
        # pred_adj = rl_based_solve(graph=Graph, num_episodes=100, mask=mask)
        # end your code


        SPT, MST, not_reached = test_loss(P=pred_adj, g=Graph,loss_args=loss_args)
    
        loss = MST + SPT + not_reached
        SPT_list.append(SPT.item())
        MST_list .append(MST.item())
        not_reached_list.append(not_reached.item())
        total_loss_list.append(loss.item())
        print('_________________________________________________________________________________')
        print('test_epoch:',i,'SPT loss:', SPT.item(), ' MST loss:', MST.item(), ' not reached', not_reached.item(),
          ' total loss:', loss.item())
    print('_________________________________________________________________________________')
    print('Avg SPT loss:', sum(SPT_list)/test_epoch, 'Avg MST loss:', sum(MST_list)/test_epoch, 'Avg not reached', sum(not_reached_list)/test_epoch,
          'Avg total loss:', sum(total_loss_list)/test_epoch, 'total loss variance:', statistics.variance(total_loss_list) )
if __name__ == '__main__':
    main()