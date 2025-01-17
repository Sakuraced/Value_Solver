# 遗传算法优化

import torch
import torch.optim as optim
from utils.loss import custom_loss_1, custom_loss_2, test_loss
from utils.encode import mask_generation, param_to_adj,param_to_adj_work,matrix_to_adj
from utils.prepro import generate_random_graph, generate_real_graph
from tqdm import tqdm
from datetime import datetime

def genetic_algorithm(graph, mask, population_size=50, loss_fn = custom_loss_2,generations=50, mutation_rate=0.1, crossover_rate=0.8):
    # 初始化参数
    n=graph.x.size()[0]
    device = graph.device
    loss_args = {'loss_iterations': 20, 'lamda': 0.1, 'unreached_weight': 10}
    
    # 1. 初始化种群：随机生成若干个 n x n 的矩阵
    population = [torch.randn(n, n).to(device) for _ in range(population_size)]
    
    # 2. 评估适应度
    def evaluate_population(pop):
        fitness = []
        for individual in pop:
            adj_matrix = matrix_to_adj(matrix=individual, graph=graph, param_mask=mask)
            a, b, c = loss_fn(adj_matrix, g=graph, loss_args=loss_args)
            fitness.append((a + b + c).item())
        return torch.tensor(fitness)
    
    # 3. 选择父代：按适应度排序并选出较优解
    def select_parents(pop, fitness):
        probabilities = 1 / (fitness + 1e-8)  # 适应度越低，概率越高
        probabilities /= probabilities.sum()
        parent_indices = torch.multinomial(probabilities, population_size, replacement=True)
        return [pop[idx] for idx in parent_indices]
    
    # 4. 交叉操作：生成新个体
    def crossover(parent1, parent2):
        if torch.rand(1).item() > crossover_rate:
            return parent1.clone(), parent2.clone()  # 保持不变
        mask = torch.rand_like(parent1) < 0.5
        child1 = parent1.clone()
        child2 = parent2.clone()
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        return child1, child2
    
    # 5. 变异操作：随机扰动
    def mutate(individual):
        if torch.rand(1).item() > mutation_rate:
            return individual
        mutation = torch.randn_like(individual) * 0.1  # 小幅随机扰动
        return individual + mutation
    
    # 开始迭代进化
    best_loss = float('inf')
    best_individual = None
    
    for generation in range(generations):
        # 评估当前种群的适应度
        fitness = evaluate_population(population)
        gen_best_loss = fitness.min().item()
        gen_best_individual = population[fitness.argmin()]
        
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_individual = gen_best_individual.clone()
        
        print(f"Generation {generation + 1}, Best Loss: {gen_best_loss}")
        
        # 选择父代
        parents = select_parents(population, fitness)
        
        # 交叉和变异，生成新种群
        new_population = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
        
        # 更新种群
        population = new_population[:population_size]
    best_individual = matrix_to_adj(matrix=best_individual, graph=graph, param_mask=mask)
    return best_individual


