import torch
import torch.optim as optim
from utils.loss import custom_loss_1, custom_loss_2, test_loss
from utils.encode import mask_generation, param_to_adj,param_to_adj_work,matrix_to_adj
from utils.prepro import generate_random_graph, generate_real_graph
from tqdm import tqdm
from datetime import datetime





# 模拟退火算法
def simulated_annealing(graph, mask, loss_fn=custom_loss_2,max_iters=100, initial_temp=1.0, temp_decay=0.99):
    # 1. 随机初始化一个 n x n 的矩阵
    n=graph.x.size()[0]
    device = graph.device
    loss_args = {'loss_iterations': 20, 'lamda': 0.1, 'unreached_weight': 10}

    X = torch.randn(n, n, requires_grad=False).to(device)  # 随机矩阵，默认不计算梯度
    
    best_X = X.clone()
    best_loss = 1e12
    
    # 2. 开始模拟退火
    temp = initial_temp
    
    for i in range(max_iters):
        # 3. 随机扰动当前解（可以是矩阵中的一个小的随机变化）
        perturbation = torch.randn_like(X) * temp  # 温度控制扰动的幅度
        new_X = X + perturbation
        
        # 4. 计算新的损失
        new_X = matrix_to_adj(matrix=new_X,graph=graph,param_mask=mask)
        a,b,c = loss_fn(new_X,g=graph,loss_args=loss_args)
        new_loss=a+b+c
        # 5. 判断是否接受新的解
        if new_loss < best_loss or torch.rand(1).item() < torch.exp((best_loss - new_loss) / temp):
            X = new_X
            best_loss = new_loss
            best_X = X.clone()
        if i%100==0:
            print(i,':',best_loss.item())
        # 6. 降低温度
        temp *= temp_decay
    
    return best_X


