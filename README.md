# Disease-Chemical Graph Analysis

This project constructs and analyzes a bipartite graph connecting diseases (MESH IDs) and chemicals (DrugBank IDs). It utilizes graph embeddings generated via Graph Autoencoder techniques and computes node similarities. This analysis is essential for uncovering potential associations between diseases and drugs, facilitating drug repurposing and disease treatment insights.

## Motivation
Understanding the relationships between diseases and chemicals is crucial for drug discovery and repositioning. Traditional approaches lack structural insights, which graph-based methods can effectively capture. Graph embeddings offer powerful representations of complex interactions, enabling precise similarity analyses and insightful predictions.

## Project Structure
```
.
├── dch.tsv                         # Original dataset containing disease-chemical pairs
├── step1_data_prep.py              # Graph construction and preprocessing
├── step2_graph_embedding.py        # Embedding the graph using Graph Autoencoder
├── step3_similarity.py             # Computing similarities between node embeddings
├── drug_drug_similarity.py         # Computing drug-drug similarity via transitive relationships
├── drug_drug_edges.py              # Creating drug-drug edges based on similarity thresholding
├── disease_chemical_graph.gpickle  # Saved NetworkX Graph object
├── graph_data.pt                   # PyTorch Geometric graph data object
├── node_embeddings.pt              # Generated embeddings from Graph Autoencoder
├── cosine_similarity.csv           # Pairwise cosine similarity matrix (disease-disease)
├── euclidean_distance.csv          # Pairwise Euclidean distance matrix (disease-disease)
├── drug_drug_similarity.csv        # Computed pairwise drug-drug similarity matrix
├── drug_drug_edges.csv             # Generated drug-drug edges based on similarity threshold
├── sample_subgraph.png             # Visualization of a small subset of the graph
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Detailed Steps

### Step 1: Environment Setup
This step ensures a reproducible Python environment with all necessary libraries.

```bash
python3 -m venv graph_env
source graph_env/bin/activate
pip3 install -r requirements.txt
```

### Step 2: Data Preparation and Graph Construction
This script loads the data (`dch.tsv`), constructs a bipartite graph, and saves the resulting NetworkX graph object.

```bash
python3 step1_data_prep.py
```

**Why required:**
- Transforms raw data into a structured graph.
- Enables visualization and graph-based analyses.

#### Expected Output:
```
Number of nodes: 7199
Number of edges: 466657
Graph and sample visualization saved successfully.
```

### Sample Subgraph Visualization (`sample_subgraph.png`)
- Visualizes a smaller subset of the constructed graph for quick qualitative assessment.
- Useful for verifying correct graph construction and preliminary understanding of node interactions.

### Step 3: Graph Embeddings with Graph Autoencoder
Generates embeddings capturing the structural information of the graph.

```bash
python3 step2_graph_embedding.py
```

**Why required:**
- Provides powerful node representations for downstream tasks.
- Captures meaningful structural and relational features of nodes.

#### Example Output:
```
Epoch: 10, Loss: 1.1276
Epoch: 20, Loss: 1.0443
Epoch: 30, Loss: 0.9897
Epoch: 40, Loss: 0.9749
Epoch: 50, Loss: 0.9780
Epoch: 60, Loss: 0.9731
Epoch: 70, Loss: 0.9719
Epoch: 80, Loss: 0.9730
Epoch: 90, Loss: 0.9718
Epoch: 100, Loss: 0.9708
Node embeddings saved successfully.
```

### Step 4: Disease-Disease Similarity Computation
Calculates similarity matrices using Cosine and Euclidean metrics for diseases, identifying structurally similar nodes.

```bash
python3 step3_similarity.py
```

**Why required:**
- Identifies potential novel drug-disease associations.
- Facilitates drug repurposing through inferred similarities.

#### Expected Output:
```
Similarity matrices saved successfully.
```

### Step 5: Drug-Drug Similarity Computation
Computes drug-drug similarity via transitive relationships based on disease-disease similarity matrices.

```bash
python3 drug_drug_similarity.py
```

**Why required:**
- Discovers drug similarities indirectly, providing deeper insights for drug repositioning and combination therapies.

#### Expected Output:
```
Drug-drug similarity matrix successfully created and saved as 'drug_drug_similarity.csv'.
```

### Step 6: Creating Drug-Drug Edges
Generates explicit edges between drugs based on a predefined similarity threshold, facilitating clear visualization and direct interpretation of drug relationships.

```bash
python drug_drug_edges.py
```

**Why required:**
- Simplifies the analysis of drug similarity relationships.
- Provides structured data ready for visualization and further analyses.

#### Example Output:
```
Drug-drug edges successfully created and saved to 'drug_drug_edges.csv'. Total edges: 16710.
```

## Outputs
- `cosine_similarity.csv`: Node similarity based on cosine distance, where higher values indicate greater similarity.
- `euclidean_distance.csv`: Node similarity based on Euclidean distance, where smaller values indicate greater similarity.
- `drug_drug_similarity.csv`: Derived pairwise drug-drug similarity based on transitive disease relationships.
- `drug_drug_edges.csv`: Drug-drug edges based on similarity thresholding.

## Dependencies
Included in `requirements.txt`:
```
pandas
networkx
matplotlib
scikit-learn
torch
torchvision
torchaudio
torch-geometric
torch-scatter
torch-sparse
torch-cluster
numpy
```

## Applications
- Exploring disease-drug relationship networks.
- Identifying drug repurposing opportunities.
- Structural similarity analyses in biomedical networks.

## Author
Avik Pramanick
