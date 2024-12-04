import torch
import torch.nn.functional as F

def diag_softmax(A):
    n=len(A)
    upper_triangle_indices = torch.triu_indices(n, n, 1)

    # 提取上对称的元素对
    a_ij = A[upper_triangle_indices[0], upper_triangle_indices[1]]
    a_ji = A[upper_triangle_indices[1], upper_triangle_indices[0]]

    # 对这些对称的元素执行 Softmax
    softmax_result = gumbel_softmax(torch.stack([a_ij, a_ji]))
    # 创建一个新的矩阵 B，用来存储 Softmax 结果
    B = A.clone()

    # 将 Softmax 结果放回对称的位置
    B[upper_triangle_indices[0], upper_triangle_indices[1]] = softmax_result[0]
    B[upper_triangle_indices[1], upper_triangle_indices[0]] = softmax_result[1]
    return B

def gumbel_softmax(logits, tau=1, Normalize=False,hard=False):
    """
    实现批量的 Gumbel-Softmax，支持多个向量同时进行操作。
    如果某列全为0，则这列输出也全为0。

    参数：
    logits: [batch_size, num_classes] 形式的输入
    tau: 温度参数
    hard: 是否返回 one-hot 向量
    """
    # 转置矩阵，逐列处理
    logits = logits.T
    # 从标准Gumbel分布中采样，并与logits形状相同
    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))

    # 计算 Gumbel-Softmax 样本
    y = logits + gumbels
    y_soft = F.softmax(y / tau, dim=-1)
    if hard:
        # 将输出变为one-hot向量，但保持梯度
        _, max_indices = y_soft.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, max_indices, 1.0)
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    if Normalize:
        y=torch.mul(y,logits)
    # 再次转置回原来的形状
    return y.T

def masked_softmax(E, B,gumbel=True,hard=False):
    mask = B == 0
    non_mask = ~mask
    non_mask_elements = torch.where(non_mask, E, torch.tensor(-float('inf')).to(E.device))
    softmax_non_mask_elements = F.softmax(non_mask_elements, dim=0)
    E = torch.where(non_mask, softmax_non_mask_elements, torch.zeros_like(E).to(E.device))
    return E