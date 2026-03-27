import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv 
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


dataset = Planetoid(root='./data/Cora_SAGE', name='Cora', transform=T.NormalizeFeatures()) 
data = dataset[0]


class GraphSAGE_Net(torch.nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, aggr='mean'): 
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_node_features = dataset.num_features
num_classes = dataset.num_classes
hidden_channels = 64  

#Creating the GraphSAGE model instance
model = GraphSAGE_Net(in_channels=num_node_features,
                      hidden_channels=hidden_channels,
                      out_channels=num_classes,
                      aggr='mean').to(device) 
data = data.to(device)

print("-" * 50)
print("Model Architecture (GraphSAGE):")
print(model)
print("-" * 50)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out_logits = model(data.x, data.edge_index)
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

num_epochs = 200
best_val_acc = 0.0
best_model_state = None
best_test_acc_at_best_val = 0.0 

print("-" * 50)
print("Starting GraphSAGE training...")
for epoch in range(1, num_epochs + 1):
    loss = train()
    
    if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
        val_acc, current_test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc (current): {current_test_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc_at_best_val = current_test_acc 
            best_model_state = model.state_dict().copy()
            print(f"*** New best validation accuracy: {best_val_acc:.4f} (Test Acc: {best_test_acc_at_best_val:.4f} at epoch {epoch}) ***")
            
    elif epoch % 1 == 0:
         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


print("-" * 50)
print("GraphSAGE Training finished!")

if best_model_state:
    model.load_state_dict(best_model_state)
    print("Loaded best GraphSAGE model weights based on validation accuracy.")

final_val_acc, final_test_acc = test()
print(f'Final Validation Accuracy (best GraphSAGE model): {final_val_acc:.4f}')
print(f'Final Test Accuracy (best GraphSAGE model): {final_test_acc:.4f}') 
print("-" * 50)

if best_model_state:
    model.load_state_dict(best_model_state)
    print("Loaded best model weights for visualization.")
else:
    print("Warning: Using the model state from the last epoch, not necessarily the best.")

model.eval()
with torch.no_grad():
    h = model.conv1(data.x, data.edge_index)
    h = F.relu(h)
    node_embeddings = h.cpu().numpy() 
    true_labels = data.y.cpu().numpy() 

print(f"Shape of node embeddings: {node_embeddings.shape}") 
print(f"Shape of true labels: {true_labels.shape}") 
print("Running t-SNE... (this might take a moment)")

tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(node_embeddings)

print(f"Shape of 2D embeddings: {embeddings_2d.shape}") 
num_classes = dataset.num_classes

plt.figure(figsize=(10, 8))

colors = plt.cm.get_cmap('tab10', num_classes) 

for i in range(num_classes):
    idxs = np.where(true_labels == i)[0]
    plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1],
                color=colors(i), label=f'Community {i}', s=20) 

plt.title('t-SNE visualization of Node Embeddings (Colored by True Community)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()
plt.savefig('graphSAGE_cora_embeddings.png')
