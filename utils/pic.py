import matplotlib.pyplot as plt
import networkx as nx
def plot_graph(pred_adj):
    G = nx.DiGraph()
    num_nodes = pred_adj.size(0)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = pred_adj[i, j].item()
            if weight != 0:
                G.add_edge(i, j, weight=weight)
    pos = nx.planar_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=250, node_color="lightblue", font_size=6, font_weight="bold",
            arrows=True)
    plt.title("Hierarchical Tree Layout")
    plt.show()