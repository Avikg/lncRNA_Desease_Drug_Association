import pickle
import torch
from torch_geometric.utils import from_networkx
import networkx as nx

# Load saved NetworkX graph
with open('disease_chemical_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Convert graph to PyG data object
data = from_networkx(G)

# Initialize node features randomly (as no features provided)
num_nodes = G.number_of_nodes()
embedding_dim = 64  # Embedding dimension size
data.x = torch.rand((num_nodes, embedding_dim), dtype=torch.float)

# Check data object
print(data)

# Save the PyG data object
torch.save(data, 'graph_data.pt')

print("PyG data object created and saved as graph_data.pt")




from torch_geometric.nn import GAE, GCNConv
import torch.optim as optim

# Define GCN Encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Model Setup
out_channels = 32  # Final embedding size
model = GAE(GCNEncoder(embedding_dim, out_channels))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the Graph Autoencoder
model.train()
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Final embeddings
model.eval()
with torch.no_grad():
    embeddings = model.encode(data.x, data.edge_index)

# Save embeddings
torch.save(embeddings, 'node_embeddings.pt')
print("Node embeddings saved successfully.")
