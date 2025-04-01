import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
import networkx as nx

# Load embeddings
embeddings = torch.load('node_embeddings.pt')
embeddings_np = embeddings.numpy()

# Load original graph
with open('disease_chemical_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Node labels
node_labels = list(G.nodes())

# Compute similarity matrices (example with first 100 nodes for efficiency)
subset_embeddings = embeddings_np[:100]  # adjust this based on your resources
subset_labels = node_labels[:100]

# Cosine Similarity
cos_sim_matrix = cosine_similarity(subset_embeddings)

# Euclidean Distance
euc_dist_matrix = euclidean_distances(subset_embeddings)

# Convert to DataFrame for better readability
cos_sim_df = pd.DataFrame(cos_sim_matrix, index=subset_labels, columns=subset_labels)
euc_dist_df = pd.DataFrame(euc_dist_matrix, index=subset_labels, columns=subset_labels)

# Save results to CSV
cos_sim_df.to_csv('cosine_similarity.csv')
euc_dist_df.to_csv('euclidean_distance.csv')

print("Similarity matrices saved successfully.")
