import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 

# --- 1. Load the Cora Dataset (remains the same) ---
dataset = Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device) # Move data to device once

# --- 2. Define the GCN Model (remains the same) ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate): # Add dropout_rate here
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate # Store dropout rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training) # Use stored dropout_rate
        x = self.conv2(x, edge_index)
        return x

class GCN_3Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, out_channels) # New layer
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training) 
        x = self.conv3(x, edge_index)
        return x

# --- Function to run a full training and evaluation cycle ---
def run_experiment(hidden_channels_config, lr_config, dropout_rate_config, num_epochs=200):
    print(f"\n--- Running Experiment: HC={hidden_channels_config}, LR={lr_config}, Dropout={dropout_rate_config} ---")
    model = GCN(in_channels=dataset.num_features,
                hidden_channels=hidden_channels_config,
                out_channels=dataset.num_classes,
                dropout_rate=dropout_rate_config).to(device) # Pass dropout_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_config, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc_for_run = 0.0
    best_model_state_for_run = None
    test_acc_at_best_val = 0.0

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        out_logits = model(data.x, data.edge_index)
        loss = criterion(out_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # --- Test (Evaluate) ---
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs: # Evaluate periodically
            model.eval()
            with torch.no_grad():
                eval_logits = model(data.x, data.edge_index)
            eval_pred = eval_logits.argmax(dim=1)
            
            current_val_correct = (eval_pred[data.val_mask] == data.y[data.val_mask]).sum()
            current_val_acc = int(current_val_correct) / int(data.val_mask.sum())

            current_test_correct = (eval_pred[data.test_mask] == data.y[data.test_mask]).sum()
            current_test_acc = int(current_test_correct) / int(data.test_mask.sum())
            
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val Acc: {current_val_acc:.4f}, Test Acc: {current_test_acc:.4f}')

            if current_val_acc > best_val_acc_for_run:
                best_val_acc_for_run = current_val_acc
                test_acc_at_best_val = current_test_acc # Store test acc when val acc is best
                best_model_state_for_run = model.state_dict().copy()
                
    
    print(f"Finished run. Best Val Acc for this run: {best_val_acc_for_run:.4f}, Corresponding Test Acc: {test_acc_at_best_val:.4f}")
    return best_val_acc_for_run, test_acc_at_best_val, best_model_state_for_run, model # Return the last model too

# --- Hyperparameter Grid ---
hidden_channels_options = [16, 32, 64]
learning_rate_options = [0.01, 0.005, 0.001]
dropout_rate_options = [0.3, 0.5, 0.7] 
best_overall_val_acc = 0.0
best_hyperparameters = {}
best_model_state_overall = None
final_test_acc_for_best_model = 0.0
best_model_for_viz = None # To store the best model object for later visualization

# --- Main Loop for Hyperparameter Tuning ---
print("=" * 50)
print("Starting Hyperparameter Tuning...")
print("=" * 50)

for hc in hidden_channels_options:
    for lr in learning_rate_options:
        for dr in dropout_rate_options:
            val_acc, test_acc, model_state, last_model_obj = run_experiment(
                hidden_channels_config=hc,
                lr_config=lr,
                dropout_rate_config=dr,
                num_epochs=200
            )
            if val_acc > best_overall_val_acc:
                best_overall_val_acc = val_acc
                final_test_acc_for_best_model = test_acc 
                best_hyperparameters = {'hidden_channels': hc, 'lr': lr, 'dropout_rate': dr}
                best_model_state_overall = model_state
                best_model_for_viz = GCN(dataset.num_features, hc, dataset.num_classes, dr).to(device)
                best_model_for_viz.load_state_dict(best_model_state_overall)


print("=" * 50)
print("Hyperparameter Tuning Finished!")
print(f"Best Validation Accuracy Achieved: {best_overall_val_acc:.4f}")
print(f"Corresponding Test Accuracy: {final_test_acc_for_best_model:.4f}")
print(f"Best Hyperparameters: {best_hyperparameters}")
print("=" * 50)



if best_model_for_viz and best_overall_val_acc > 0: 
    print("\nVisualizing embeddings of the best model...")
    best_model_for_viz.eval()
    with torch.no_grad():
        h = best_model_for_viz.conv1(data.x, data.edge_index)
        h = F.relu(h)
    embeddings_to_visualize = h.cpu().numpy()
    true_labels = data.y.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings_to_visualize)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='jet', s=10)
    handles, _ = scatter.legend_elements(prop='colors')
    legend_labels = [f'Community {i}' for i in range(dataset.num_classes)]
    plt.legend(handles, legend_labels, title="Communities")
    plt.title(f't-SNE of Best GCN Model (Val Acc: {best_overall_val_acc:.4f}, Test Acc: {final_test_acc_for_best_model:.4f})\nParams: {best_hyperparameters}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    plt.savefig('hyper_cora_embeddings.png')