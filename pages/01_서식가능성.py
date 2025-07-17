import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random


def main():
    st.title("BBB Tight Junction Network Simulation under Variable Gravity")

    # Sidebar parameters
    st.sidebar.header("Simulation Parameters")
    grid_size = st.sidebar.slider("Grid size (nodes per side)", 5, 50, 20)
    g = st.sidebar.slider("Gravity (g)", 0.0, 1.0, 1.0, step=0.01)
    p0 = st.sidebar.slider("Base edge loss probability (p0)", 0.0, 1.0, 0.1, step=0.01)
    alpha = st.sidebar.slider("Gravity dependency α", 0.0, 2.0, 1.0, step=0.01)

    # Compute edge loss probability
    p_loss = p0 + alpha * (1.0 - g)
    st.sidebar.markdown(f"**Edge loss probability**: {p_loss:.2f}")

    # Build original grid graph
    G0 = nx.grid_2d_graph(grid_size, grid_size)
    mapping = { (i, j): i * grid_size + j for i, j in G0.nodes() }
    G = nx.relabel_nodes(G0, mapping)
    rev_map = {mapping[node]: node for node in mapping}

    # Damage the network by removing edges randomly
    G_damaged = G.copy()
    for u, v in list(G.edges()):
        if random.random() < p_loss:
            G_damaged.remove_edge(u, v)

    # Compute largest connected component size
    components = list(nx.connected_components(G_damaged))
    largest_cc = max(components, key=len) if components else set()
    size_lcc = len(largest_cc)
    total_nodes = G.number_of_nodes()

    st.write(f"Largest connected component: {size_lcc} / {total_nodes} nodes")

    # Prepare node positions for plotting
    pos = {node: rev_map[node] for node in G.nodes()}

    # Plot original and damaged networks side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Original Network (1g)")
    nx.draw(G, pos, ax=axes[0], node_size=20, with_labels=False)

    axes[1].set_title(f"Damaged Network ({g}g)")
    node_colors = ["blue" if node in largest_cc else "lightgrey" for node in G.nodes()]
    nx.draw(G_damaged, pos, ax=axes[1], node_size=20, node_color=node_colors, with_labels=False)

    st.pyplot(fig)


if __name__ == "__main__":
    main()
