import pandas as pd

# Load drug-drug similarity matrix
drug_similarity_df = pd.read_csv('drug_drug_similarity.csv', index_col=0)

# Set your similarity threshold
similarity_threshold = 0.5  # Adjust this threshold as needed

# Create edges based on the threshold
edges = []

for drug_a in drug_similarity_df.index:
    for drug_b in drug_similarity_df.columns:
        similarity = drug_similarity_df.loc[drug_a, drug_b]

        # Avoid self-loops and duplicate edges (ensure symmetry)
        if drug_a != drug_b and similarity >= similarity_threshold:
            edge = tuple(sorted((drug_a, drug_b)))  # sorted ensures no duplicates
            edges.append(edge)

# Remove duplicate edges by converting to a set
unique_edges = set(edges)

# Create DataFrame of edges
edges_df = pd.DataFrame(unique_edges, columns=['Drug1', 'Drug2'])

# Save edges to CSV
edges_df.to_csv('drug_drug_edges.csv', index=False)

print(f"Drug-drug edges successfully created and saved to 'drug_drug_edges.csv'. Total edges: {len(edges_df)}.")
