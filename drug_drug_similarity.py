import pandas as pd
import numpy as np

# Load original dataset
disease_chemical_df = pd.read_csv('dch.tsv', sep='\t')

# Load disease-disease similarity matrix (cosine_similarity.csv or euclidean_distance.csv)
disease_similarity_df = pd.read_csv('cosine_similarity.csv', index_col=0)

# Create disease-to-drug mapping
disease_to_drugs = disease_chemical_df.groupby('Disease(MESH)')['Chemical'].apply(set).to_dict()

# Get unique chemicals
chemicals = disease_chemical_df['Chemical'].unique()
chem_idx = {chem: idx for idx, chem in enumerate(chemicals)}

# Initialize drug-drug similarity matrix
drug_similarity_matrix = np.zeros((len(chemicals), len(chemicals)))

# Compute transitive drug similarity
for disease_a in disease_similarity_df.index:
    for disease_b in disease_similarity_df.columns:
        similarity = disease_similarity_df.loc[disease_a, disease_b]

        drugs_a = disease_to_drugs.get(disease_a, set())
        drugs_b = disease_to_drugs.get(disease_b, set())

        # Update similarity scores for drug pairs associated with these diseases
        for drug_a in drugs_a:
            for drug_b in drugs_b:
                idx_a, idx_b = chem_idx[drug_a], chem_idx[drug_b]
                drug_similarity_matrix[idx_a, idx_b] += similarity

# Normalize the similarity matrix
max_sim = np.max(drug_similarity_matrix)
if max_sim > 0:
    drug_similarity_matrix /= max_sim

# Create DataFrame for better readability
drug_similarity_df = pd.DataFrame(drug_similarity_matrix, index=chemicals, columns=chemicals)

# Save the result
drug_similarity_df.to_csv('drug_drug_similarity.csv')

print("Drug-drug similarity matrix successfully created and saved as 'drug_drug_similarity.csv'.")
