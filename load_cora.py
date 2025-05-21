from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# Load the Cora dataset
dataset = Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0] 

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}') #Communities

print(f'\nGraph: {data}:')
print('===========================================================================================================')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print(f'Node features shape: {data.x.shape}')
print(f'Edge index shape: {data.edge_index.shape}')
print(f'Labels (communities) shape: {data.y.shape}')