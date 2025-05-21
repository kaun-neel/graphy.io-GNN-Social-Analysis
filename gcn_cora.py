import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- 1. Load the Cora Dataset ---
dataset = Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]  # Get the single graph object.

print(f"Dataset: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of node features: {dataset.num_features}")
print(f"Number of classes (communities): {dataset.num_classes}")
print("-" * 50)
print(f"Graph data: {data}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Node features shape: {data.x.shape}")
print(f"Edge index shape: {data.edge_index.shape}")
print(f"Labels shape: {data.y.shape}")
print(f"Training nodes: {data.train_mask.sum().item()}")
print(f"Validation nodes: {data.val_mask.sum().item()}")
print(f"Test nodes: {data.test_mask.sum().item()}")
print("-" * 50)


# --- 2. Define the GCN Model ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):       
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        return x


# --- 3. Instantiate the Model & Prepare for Training ---

# Determine device (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model parameters
num_node_features = dataset.num_features
num_classes = dataset.num_classes
hidden_channels = 16  

# Create the model instance
model = GCN(in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=num_classes)

# Move the model and data to the selected device
model = model.to(device)
data = data.to(device)

# Print the model structure
print("-" * 50)
print("Model Architecture:")
print(model)
print("-" * 50)

# --- 4. Perform a Single Forward Pass ---
model.eval() # Sets self.training to False

# No gradients needed for just a forward pass test
with torch.no_grad():
    out_logits = model(data.x, data.edge_index)

print(f"Output logits shape: {out_logits.shape}") # Expected: [num_nodes, num_classes] -> [2708, 7]
print("Sample output logits (first 5 nodes):")
print(out_logits[:5])
print("-" * 50)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out_logits = model(data.x,data.edge_index)
    loss = criterion(out_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out_logits = model(data.x, data.edge_index)
    pred = out_logits.argmax(dim=1)

    val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    val_acc = int(val_correct) / int(data.val_mask.sum())

    test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(test_correct) / int(data.test_mask.sum())

    return val_acc, test_acc

# --- Main Training Loop ---
num_epochs = 200
best_val_acc = 0.0
best_model_state = None #To store the weights of the best model

print("-" * 50)
print("Starting training...")
for epoch in range(1, num_epochs + 1):
    loss = train()
    
    if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
        val_acc, current_test_acc = test() # Get current test accuracy along with val_acc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc (current): {current_test_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy() # Save the model's weights
            print(f"*** New best validation accuracy: {best_val_acc:.4f} (at epoch {epoch}) ***")
            
    elif epoch % 1 == 0: # Log loss more frequently
         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

print("-" * 50)
print("Training finished!")

# Load the best model state before final evaluation
if best_model_state:
    model.load_state_dict(best_model_state)
    print("Loaded best model weights based on validation accuracy.")

# Final evaluation on the test set using the BEST model
final_val_acc, final_test_acc = test()
print(f'Final Validation Accuracy (best model): {final_val_acc:.4f}') # Should be your best_val_acc
print(f'Final Test Accuracy (best model): {final_test_acc:.4f}')
print("-" * 50)


#Node Embeddings
model.eval()
with torch.no_grad():
    h = model.conv1(data.x, data.edge_index)
    h = F.relu(h)

embeddings_to_visualize = h.cpu().numpy()
true_labels = data.y.cpu().numpy()

print("-" * 50)
print("Preparing for t-SNE visualization...")
print(f"Shape of embeddings for t-SNE: {embeddings_to_visualize.shape}") # Should be [2708, hidden_channels]
print(f"Shape of true labels: {true_labels.shape}")

#Applying t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt # Import for plotting

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=300) # n_iter can be increased for better convergence
embeddings_2d = tsne.fit_transform(embeddings_to_visualize)

print(f"Shape of 2D embeddings after t-SNE: {embeddings_2d.shape}") # Should be [2708, 2]
print("t-SNE transformation complete.")
print("-" * 50)

#Plotting the 2D Embeddings
plt.figure(figsize=(10, 8))
# Scatter plot:
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='jet', s=10)
# Create a legend with unique class labels
handles, _ = scatter.legend_elements(prop='colors')
legend_labels = [f'Community {i}' for i in range(dataset.num_classes)]
plt.legend(handles, legend_labels, title="Communities")

plt.title('t-SNE visualization of GCN Node Embeddings (Cora Dataset)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('gcn_cora_tsne_embeddings.png')
plt.show()

print("Visualization displayed.")
print("-" * 50)