    ---------------------------------------------------
   |                STARTING POINT: IDEA               |
   |          "Social Network Analysis with GNNs"      |
    ---------------------------------------------------
                      |
                      V
    ---------------------------------------------------
   |         PHASE 1: LAYING THE FOUNDATION           |
   |              (The "Base Camp")                   |
    ---------------------------------------------------
   | 1. DEFINE SCOPE:                                  |
   |    - Choose ONE Task (Community, Link, Influence)|
   |    - Choose ONE GNN (GCN or GraphSAGE)           |
   | 2. SETUP ENVIRONMENT:                             |
   |    - Python & Virtual Env                        |
   |    - PyTorch, PyG, NetworkX, etc.                |
   | 3. VERSION CONTROL:                               |
   |    - Initialize Git Repo                         |
   |    - Create .gitignore                           |
    ---------------------------------------------------
                      |
                      V
    ---------------------------------------------------
   |          PHASE 2: THE DATA EXPEDITION             |
   |            (Sourcing Your "Gold")                |
    ---------------------------------------------------
   | 1. ACQUIRE DATASET:                               |
   |    - SNAP, Kaggle, PyG Datasets (e.g., Cora)     |
   |    - (Avoid direct Twitter API for now - too hard)|
   | 2. EXPLORE & UNDERSTAND:                          |
   |    - Format (Edge list? Features?)               |
   |    - Node ID Mapping                             |
   | 3. PREPROCESS & TRANSFORM:                        |
   |    - Load into NetworkX (for EDA)                |
   |    - Create PyTorch Geometric `Data` object:     |
   |      - `edge_index` (Graph Structure)            |
   |      - `x` (Node Features - create if none)      |
   | 4. EXPLORATORY DATA ANALYSIS (EDA):               |
   |    - Basic stats (nodes, edges, degree)          |
   |    - Visualize small subgraphs (NetworkX)        |
    ---------------------------------------------------
                      |
                      V
    ---------------------------------------------------
   |      PHASE 3: MODEL CONSTRUCTION SITE             |
   | (Building Your "Analysis Engine" - CHOOSE ONE PATH)|
    ---------------------------------------------------
   |          GNN CORE (GCN/GraphSAGE Layers)          |
   |                     /  |  \\                       |
   |                    /   |   \\                      |
   |  PATH A: COMMUNITY | PATH B: LINK   | PATH C: INFLUENCER |
   |    DETECTION       |  PREDICTION    |   IDENTIFICATION   |
   | - Node Classif.    | - Edge Level   | - Node Classif. or |
   | - Labels: Ground   | - Split Edges  |   Regression       |
   |   truth or algo-   |   (train/val/  | - Targets: Centrality|
   |   generated (Louvain)|   test)      |   metrics (PageRank)|
   | - Output: Community| - Negative     |   or other proxies |
   |   assignments      |   Sampling     | - Output: Influence|
   | - Loss: CrossEntropy| - Decoder      |   score/class      |
   |                    |   (dot product/| - Loss: CrossEntropy |
   |                    |    MLP)        |   or MSE           |
   |                    | - Loss: BCE    |                    |
    ---------------------------------------------------
                      | (Path Merges for Training)
                      V
    ---------------------------------------------------
   |       PHASE 4: TRAINING & EVALUATION ARENA        |
   |           (Testing Your Engine's Mettle)          |
    ---------------------------------------------------
   | 1. IMPLEMENT TRAINING LOOP:                       |
   |    - Forward pass, Loss calculation, Backward pass|
   |    - Optimizer (Adam)                            |
   | 2. IMPLEMENT EVALUATION:                          |
   |    - Define Metrics (Accuracy, AUC, F1, NMI, etc.)|
   |    - Validation set checks                       |
   | 3. RUN & ITERATE:                                 |
   |    - Train the model                             |
   |    - Monitor loss & validation metrics           |
   |    - Basic Hyperparameter Tuning (LR, layers)    |
   | 4. TEST SET PERFORMANCE:                          |
   |    - Evaluate on unseen test data                |
    ---------------------------------------------------
                      |
                      V
    ---------------------------------------------------
   |     PHASE 5: INSIGHTS & VISUALIZATION POST        |
   |           (Unveiling the "Treasure")             |
    ---------------------------------------------------
   | 1. ANALYZE RESULTS:                               |
   |    - (Task-Specific) Examine predictions         |
   |    - E.g., Top influencers, sample communities,  |
   |      strongest recommended links                 |
   | 2. VISUALIZE:                                     |
   |    - Graph with predicted communities (NetworkX) |
   |    - Node Embeddings (t-SNE/UMAP)                |
   | 3. INTERPRET & DOCUMENT:                          |
   |    - What did you find? What works/doesn't?      |
   |    - Jupyter Notebook for storytelling           |
    ---------------------------------------------------
                      |
                      V
    ---------------------------------------------------
   |      PHASE 6: EXPANSION & NEXT HORIZONS (Optional)|
   |             (Charting New Territories)            |
    ---------------------------------------------------
   | - Try the OTHER GNN (GCN if SAGE, SAGE if GCN)  |
   | - Implement ANOTHER TASK from Phase 3           |
   | - More Sophisticated Features/Models (e.g., GAT) |
   | - Explore Scalability (NeighborSampler)         |
   | - Compare with Baselines (Non-GNN methods)      |
   | - Simple UI/Dashboard (Streamlit/Flask)         |
    ---------------------------------------------------
                      |
                      V
    ---------------------------------------------------
   |                 PROJECT COMPLETE                  |
   |    "Graphy.io System v1.0 - Report & Code"    |
    ---------------------------------------------------

## LEGEND / KEY TOOLS:

- GNN: Graph Neural Network (GCN/GraphSAGE)
- PyG: PyTorch Geometric
- NX: NetworkX
- Metrics: Accuracy, F1, AUC, NMI, MSE, etc.
- Git: Version Control

## Results
The GCN model achieved approximately 80.2% test accuracy on the Cora dataset.
Here's a t-SNE visualization of the learned node embeddings:
![t-SNE Visualization of Cora Embeddings](images/gcn_cora_gcn.png)
