import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Load TSV data correctly using existing header
df = pd.read_csv('dch.tsv', sep='\t', header=0)

# Drop potential invalid rows (if header row got into data mistakenly)
df = df[df['Disease(MESH)'] != 'Disease(MESH)']

# Rename columns for convenience
df.columns = ['Disease', 'Chemical']

# Confirm corrected data
print(df.head())

# Initialize Bipartite Graph
G = nx.Graph()

# Add nodes with node type
diseases = df['Disease'].unique()
chemicals = df['Chemical'].unique()

G.add_nodes_from(diseases, bipartite='disease')
G.add_nodes_from(chemicals, bipartite='chemical')

# Add edges (Disease-Chemical pairs)
edge_list = list(df.itertuples(index=False, name=None))
G.add_edges_from(edge_list)

# Graph summary
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Due to large graph, skip visualization or visualize a smaller subgraph if needed
subgraph_nodes = list(diseases[:10]) + list(chemicals[:10])
subgraph = G.subgraph(subgraph_nodes)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(subgraph, k=0.15)
nx.draw(subgraph, pos, with_labels=True, node_size=100, font_size=8, alpha=0.8)
plt.title("Sample Disease-Chemical Bipartite Subgraph")

# Save plot instead of showing (avoids non-interactive error)
plt.savefig('sample_subgraph.png')

# Save graph using pickle
with open('disease_chemical_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)

print("Graph and sample visualization saved successfully.")
