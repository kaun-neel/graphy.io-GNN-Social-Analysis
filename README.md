<div align="center">

```
   ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██╗   ██╗   ██╗ ██████╗
  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║╚██╗ ██╔╝   ██║██╔═══██╗
  ██║  ███╗██████╔╝███████║██████╔╝███████║ ╚████╔╝    ██║██║   ██║
  ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║  ╚██╔╝     ██║██║   ██║
  ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║   ██║  ██╗ ██║╚██████╔╝
   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝
```

**Social Network Analysis with Graph Neural Networks**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-3C2179?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-00e5ff?style=flat-square)

*Uncovering hidden patterns in complex relational data using GCN & GraphSAGE*

</div>

---

## 📖 Abstract

In the interconnected world of social media, understanding the structure and dynamics of networks is crucial. Traditional machine learning models often fall short when dealing with graph-structured data.

Graphy.io demonstrates the power of **Graph Neural Networks (GNNs)** on social graphs — representing users as **nodes** and relationships as **edges** — to tackle tasks intractable for standard ML. By applying models like GCN and GraphSAGE to Twitter's follower network or citation graphs like Cora, we unlock a new class of relational intelligence.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Community Detection** | Identify densely connected clusters of users within the network |
| 🔗 **Link Prediction** | Recommend new connections between users likely to interact |
| ⭐ **Influencer Identification** | Pinpoint the most central nodes based on network topology |
| 🏷️ **Node Classification** | Categorize nodes based on connections and attributes |
| 📊 **t-SNE Visualization** | Visualize high-dimensional embeddings reduced to 2D |
| 📈 **Rigorous Evaluation** | Benchmark with Accuracy, F1, AUC, NMI, and MSE |

---

## 🛠️ Tech Stack

```
Language   →  Python 3.8+
DL Framework →  PyTorch
Graph ML   →  PyTorch Geometric (PyG)
Graph Ops  →  NetworkX
ML Toolkit →  scikit-learn
Numerics   →  NumPy · Pandas
Plotting   →  Matplotlib · Seaborn
Versioning →  Git & GitHub
```

---

## 🧠 Models & Notation

### GCN — Graph Convolutional Network
Aggregates neighbourhood features via spectral convolutions. Strong baseline for **transductive** node classification tasks.

### GraphSAGE — Graph Sample and Aggregate
Inductive framework that samples and aggregates from local neighbourhoods. Scales to **unseen nodes** at inference time.

**Evaluation Metrics:** `Accuracy` · `F1-Score` · `AUC` · `NMI` (Normalized Mutual Information) · `MSE`

---

## ⚙️ Installation & Setup

**Prerequisites:** Python 3.8+ · pip · venv

```bash
# 1. Clone the repository
git clone https://github.com/kaun-neel/graphy.io.git
cd graphy.io

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install PyTorch for your system / CUDA version
#    → https://pytorch.org/get-started/locally/

# 4. Install PyTorch Geometric
pip install torch_geometric

# 5. Install remaining dependencies
pip install -r requirements.txt
# numpy · pandas · networkx · scikit-learn · matplotlib · seaborn
```

---

## 📈 Results

> **Dataset:** Cora — 2,708 nodes · 5,429 edges · 7 topic categories  
> **Task:** Node classification (predict paper topic)  
> **Model:** Graph Convolutional Network (GCN)

```
Test Accuracy  ████████████████████████░░░░░  80.2%
```

The t-SNE plot of learned embeddings shows **clear clustering by paper category**, confirming the model has successfully learned to group topically similar papers together in the embedding space.

![t-SNE Visualization](images/gcn_cora_tsne_embeddings.png)

---

## 🚀 Project Goals

- [x] Implement GCN for node classification on citation networks
- [x] Evaluate with standard metrics (Accuracy, F1, AUC, NMI)
- [x] Visualize learned embeddings with t-SNE
- [ ] Implement GraphSAGE for inductive learning
- [ ] Scale to larger social network datasets (Twitter, Reddit)
- [ ] Hyperparameter tuning with Optuna / Ray Tune
- [ ] Deploy as a REST API with FastAPI or Flask

---

## 💡 Future Work

1. **GraphSAGE Experiments** — Implement and benchmark inductive learning against GCN
2. **Larger Graphs** — Apply to Twitter or Reddit subsets for real-world scale
3. **Hyperparameter Tuning** — Automate search with Optuna or Ray Tune
4. **Web Service** — Serve predictions (e.g., connection recommendations) via FastAPI

---

## 🤝 Contributing

Contributions are greatly appreciated! Here's the workflow:

```bash
# 1. Fork the project, then:
git checkout -b feature/YourFeatureName

# 2. Make your changes and commit
git commit -m 'Add some AmazingFeature'

# 3. Push and open a Pull Request
git push origin feature/YourFeatureName
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

**Indraneel Bose**

[![GitHub](https://img.shields.io/badge/GitHub-@kaun--neel-181717?style=flat-square&logo=github)](https://github.com/kaun-neel)
[![Email](https://img.shields.io/badge/Email-indraneelbose89191@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:indraneelbose89191@gmail.com)
[![Project](https://img.shields.io/badge/Project-graphy.io-00e5ff?style=flat-square)](https://github.com/kaun-neel/graphy.io)

</div>
