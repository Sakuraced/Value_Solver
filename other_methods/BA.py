import torch
from utils.encode import mask_generation, param_to_adj,param_to_adj_work,matrix_to_adj
from utils.loss import custom_loss_1, custom_loss_2, test_loss


def artificial_bee_colony(graph, mask, loss_fn=custom_loss_2, max_iters=100, n_bees=20, n_iterations_per_bee=10, neighborhood_size=5):
    """
    人工蜂群算法
    graph: 图数据
    mask: 参数掩码
    loss_fn: 损失函数
    max_iters: 最大迭代次数
    n_bees: 蜂群数量
    n_iterations_per_bee: 每只蜜蜂的局部搜索次数
    neighborhood_size: 邻域大小，用于扰动的幅度
    """
    n = graph.x.size()[0]
    device = graph.device
    loss_args = {'loss_iterations': 20, 'lamda': 0.1, 'not_reached_weight': 10}
    
    # 1. 初始化蜜蜂位置（即种群解）
    bees = [torch.randn(n, n, requires_grad=False).to(device) for _ in range(n_bees)]
    best_bee = bees[0]
    best_loss = float('inf')
    
    # 2. 计算每只蜜蜂的适应度
    fitness = [evaluate_loss(bee, graph, mask, loss_fn, loss_args) for bee in bees]

    # 3. 开始迭代
    for i in range(max_iters):
        # 采蜜蜂阶段
        for j in range(n_bees):
            current_bee = bees[j].clone()
            current_loss = fitness[j]
            
            # 局部扰动：采蜜蜂在当前位置周围进行局部搜索
            for _ in range(n_iterations_per_bee):
                perturbation = torch.randn_like(current_bee) * neighborhood_size  # 控制扰动的幅度
                new_bee = current_bee + perturbation
                new_bee = matrix_to_adj(matrix=new_bee, graph=graph, param_mask=mask)  # 将解转换为图的邻接矩阵
                
                new_loss = evaluate_loss(new_bee, graph, mask, loss_fn, loss_args)

                # 贪心策略：选择适应度较好的解
                if new_loss < current_loss:
                    current_bee = new_bee
                    current_loss = new_loss

            # 更新蜜蜂的位置和适应度
            bees[j] = current_bee
            fitness[j] = current_loss

        # 观察蜂阶段：观察蜂根据适应度选择蜜源，并尝试在选择的蜜源附近生成新解
        total_fitness = sum(fitness)
        probabilities = [fit / total_fitness for fit in fitness]  # 计算每只蜜蜂的选择概率

        for j in range(n_bees):
            if torch.rand(1).item() < probabilities[j]:  # 观察蜂选择该蜜蜂
                current_bee = bees[j].clone()
                current_loss = fitness[j]
                
                # 局部扰动：观察蜂在蜜源附近进行扰动
                for _ in range(n_iterations_per_bee):
                    perturbation = torch.randn_like(current_bee) * neighborhood_size
                    new_bee = current_bee + perturbation
                    new_bee = matrix_to_adj(matrix=new_bee, graph=graph, param_mask=mask)
                    
                    new_loss = evaluate_loss(new_bee, graph, mask, loss_fn, loss_args)
                    
                    # 贪心策略：观察蜂选择适应度较好的解
                    if new_loss < current_loss:
                        current_bee = new_bee
                        current_loss = new_loss

                # 更新观察蜂的位置和适应度
                bees[j] = current_bee
                fitness[j] = current_loss

        # 侦察蜂阶段：如果某个解在多次迭代后没有改进，则重新随机生成
        for j in range(n_bees):
            if fitness[j] > best_loss:  # 如果适应度不如最优解
                bees[j] = torch.randn(n, n, requires_grad=False).to(device)  # 随机重新生成解
                fitness[j] = evaluate_loss(bees[j], graph, mask, loss_fn, loss_args)
        
        # 4. 更新最优解
        for j in range(n_bees):
            if fitness[j] < best_loss:
                best_loss = fitness[j]
                best_bee = bees[j]

        # 输出当前最优解
        if i % 10 == 0:
            print(f"Iteration {i}: Best loss = {best_loss.item()}")

    return best_bee


def evaluate_loss(bee, graph, mask, loss_fn, loss_args):
    """
    评估某个解的损失
    """
    bee_adj = matrix_to_adj(matrix=bee, graph=graph, param_mask=mask)
    a, b, c = loss_fn(bee_adj, g=graph, loss_args=loss_args)
    return a + b + c
