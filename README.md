# Disease-Chemical Graph Analysis

This project involves constructing a bipartite graph connecting diseases (MESH IDs) and chemicals (DrugBank IDs), generating node embeddings through Graph Autoencoder techniques, and calculating node similarities.

## Project Structure
```
.
├── dch.tsv                        # Original dataset
├── step1_data_prep.py             # Graph construction and preprocessing
├── step2_graph_embedding.py       # Graph embedding with Graph Autoencoder
├── step3_similarity.py            # Node similarity computations
├── disease_chemical_graph.gpickle # Saved NetworkX Graph
├── graph_data.pt                  # PyTorch Geometric graph data
├── node_embeddings.pt             # Generated node embeddings
├── cosine_similarity.csv          # Cosine similarity matrix
├── euclidean_distance.csv         # Euclidean distance matrix
├── sample_subgraph.png            # Sample visualization of the graph
└── README.md                      # This file
```

## Steps to Run

### Step 1: Environment Setup
```bash
python3 -m venv graph_env
source graph_env/bin/activate
pip install pandas networkx matplotlib scikit-learn torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-cluster
```

### Step 2: Data Preparation and Graph Construction
```bash
python step1_data_prep.py
```
#### Expected Output:
```
Number of nodes: 7199
Number of edges: 466657
Graph and sample visualization saved successfully.
```

### Step 3: Graph Embeddings with Graph Autoencoder
```bash
python step2_graph_embedding.py
```
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

### Step 4: Similarity Computation
```bash
python step3_similarity.py
```
#### Expected Output:
```
Similarity matrices saved successfully.
```

## Outputs
- `cosine_similarity.csv`: Pairwise cosine similarity between nodes.
- `euclidean_distance.csv`: Pairwise Euclidean distances between nodes.

## Applications
- Disease-drug relationship analysis
- Drug repurposing research
- Graph-based similarity analysis

## Dependencies
- Python (3.8 or higher recommended)
- pandas
- networkx
- matplotlib
- torch, PyTorch Geometric
- scikit-learn

## Author
Avik Pramanick

