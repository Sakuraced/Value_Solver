import torch
from utils.loss import custom_loss_2
from utils.prepro import generate_random_graph


def main():
    # 参数设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 200
    p = 0.1
    center_node = 0
    seed = 44

    # 生成测试图
    Graph = generate_random_graph(n=n, p=p, center_node=center_node, seed=seed,device=device)



    '''
    如下代码，请给出一棵覆盖所有节点的Graph的子树的邻接矩阵pred_adj
    同时最小化如下损失函数
    '''
    pred_adj = torch.zeros((n, n)).to(device)
    '''
    请在以上部分输入代码
    '''
    SPT, MST, not_reached = custom_loss_2(P=pred_adj, g=Graph, device=device)
    loss = MST + SPT + not_reached
    print('_________________________________________________________________________________')
    print('SPT loss:', SPT.item(), ' MST loss:', MST.item(), ' not reached', not_reached.item(),
          ' total loss:', loss.item())

if __name__ == '__main__':
    main()