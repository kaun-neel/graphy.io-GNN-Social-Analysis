import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv 
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- 1. Load the Cora Dataset ---
dataset = Planetoid(root='./data/Cora_SAGE', name='Cora', transform=T.NormalizeFeatures()) # Use a different root to avoid conflicts if needed
data = dataset[0]

# --- 2. Define the GraphSAGE Model ---
class GraphSAGE_Net(torch.nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, aggr='mean'): # Added aggr parameter
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- 3. Instantiate the Model & Prepare for Training (Mostly the same) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_node_features = dataset.num_features
num_classes = dataset.num_classes
hidden_channels = 64  # Start with the same as GCN for comparison

# Create the GraphSAGE model instance
model = GraphSAGE_Net(in_channels=num_node_features,
                      hidden_channels=hidden_channels,
                      out_channels=num_classes,
                      aggr='mean').to(device) # <<<< Pass aggr argument
data = data.to(device)

print("-" * 50)
print("Model Architecture (GraphSAGE):")
print(model)
print("-" * 50)

# --- Optimizer and Loss Function (can remain the same) ---
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# --- Training Function (train()) - Identical to GCN's ---
def train():
    model.train()
    optimizer.zero_grad()
    out_logits = model(data.x, data.edge_index)
    loss = criterion(out_logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# --- Evaluation (Test) Function (test()) - Identical to GCN's ---
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

# --- Main Training Loop (Mostly the same, ensure you use the "save best model" logic) ---
num_epochs = 200
best_val_acc = 0.0
best_model_state = None
best_test_acc_at_best_val = 0.0 # To store test acc when val acc was best

print("-" * 50)
print("Starting GraphSAGE training...")
for epoch in range(1, num_epochs + 1):
    loss = train()
    
    if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs: # Log every 10 epochs
        val_acc, current_test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc (current): {current_test_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc_at_best_val = current_test_acc # Store test acc at this point
            best_model_state = model.state_dict().copy()
            print(f"*** New best validation accuracy: {best_val_acc:.4f} (Test Acc: {best_test_acc_at_best_val:.4f} at epoch {epoch}) ***")
            
    elif epoch % 1 == 0: # Log loss more frequently
         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


print("-" * 50)
print("GraphSAGE Training finished!")

if best_model_state:
    model.load_state_dict(best_model_state)
    print("Loaded best GraphSAGE model weights based on validation accuracy.")

# Final evaluation on the test set using the BEST model
final_val_acc, final_test_acc = test() # This will re-evaluate the best model
print(f'Final Validation Accuracy (best GraphSAGE model): {final_val_acc:.4f}')
print(f'Final Test Accuracy (best GraphSAGE model): {final_test_acc:.4f}') # This should be best_test_acc_at_best_val
print("-" * 50)

# Ensure the best model is loaded
if best_model_state:
    model.load_state_dict(best_model_state)
    print("Loaded best model weights for visualization.")
else:
    print("Warning: Using the model state from the last epoch, not necessarily the best.")

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    h = model.conv1(data.x, data.edge_index)
    h = F.relu(h)
    node_embeddings = h.cpu().numpy() # Move to CPU and convert to NumPy array
    true_labels = data.y.cpu().numpy()   # Also get true labels as NumPy array

print(f"Shape of node embeddings: {node_embeddings.shape}") # Should be [num_nodes, hidden_channels]
print(f"Shape of true labels: {true_labels.shape}")     # Should be [num_nodes]
print("Running t-SNE... (this might take a moment)")
# n_iter is the number of iterations.
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(node_embeddings)

print(f"Shape of 2D embeddings: {embeddings_2d.shape}") # Should be [num_nodes, 2]
# Get the number of unique classes/communities
num_classes = dataset.num_classes # Or len(np.unique(true_labels))

plt.figure(figsize=(10, 8))

# Create a scatter plot, coloring points by their true label
# Use a color map that has enough distinct colors for your classes
# 'viridis', 'plasma', 'tab10', 'tab20' are good options
colors = plt.cm.get_cmap('tab10', num_classes) # Using tab10 colormap

for i in range(num_classes):
    # Select indices of nodes belonging to class i
    idxs = np.where(true_labels == i)[0]
    # Plot these points with a specific color and label
    plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1],
                color=colors(i), label=f'Community {i}', s=20) # s is marker size

plt.title('t-SNE visualization of Node Embeddings (Colored by True Community)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()
plt.savefig('graphSAGE_cora_embeddings.png')