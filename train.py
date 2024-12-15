import os
import sys
import time
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# 数据过滤
def filter_data(X, highly_genes=500):
    X = np.ceil(X).astype(float)
    adata = sc.AnnData(X)
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=4, min_disp=0.5, n_top_genes=highly_genes, subset=True
    )
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)
    return genes_idx, cells_idx

# 使用 PyG 构建图
def make_graph_pyg(X, Y=None, threshold=0, dense_dim=50, normalize_weights="log_per_cell"):
    num_genes = X.shape[1]
    num_cells = X.shape[0]
    if normalize_weights == "log_per_cell":
        X1 = np.log1p(X)
        X1 /= np.sum(X1, axis=1, keepdims=True) + 1e-6
    elif normalize_weights == "per_cell":
        X1 = X / (np.sum(X, axis=1, keepdims=True) + 1e-6)
    else:
        X1 = X
    row_idx, gene_idx = np.nonzero(X > threshold)
    non_zeros = X1[(row_idx, gene_idx)]
    cell_idx = row_idx + num_genes
    gene_feat = PCA(dense_dim).fit_transform(X1.T)
    cell_feat = X1 @ gene_feat
    x = np.vstack([gene_feat, cell_feat]).astype(np.float32)
    y = np.hstack([[-1] * num_genes, Y]).astype(int) if Y is not None else [-1] * (num_genes + num_cells)
    edge_index = np.vstack([gene_idx, cell_idx])
    return edge_index, x, y

# 模型训练
def train(model, optimizer, n_epochs, dataloader, n_clusters, save=False, save_path="./output", plot=False, cluster=["KMeans"], cluster_params={}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(save_path, exist_ok=True)  # 创建保存路径

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            adj_logits, embeddings = model(batch.x, batch.edge_index)
            edge_logits = adj_logits[batch.edge_index[0], batch.edge_index[1]]
            adj = torch.ones(batch.edge_index.size(1), device=device)
            loss = F.binary_cross_entropy_with_logits(edge_logits, adj)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}")

        # 保存模型权重
        if save:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch + 1}.pth"))
            print(f"Model saved for epoch {epoch + 1}.")

    # 评估模型
    scores = evaluate(model, dataloader, n_clusters, plot=plot, cluster=cluster, cluster_params=cluster_params)

    # 保存结果
    if save:
        with open(os.path.join(save_path, "training_scores.pkl"), "wb") as f:
            pickle.dump(scores, f)
        print(f"Training scores saved at {save_path}.")

    return scores

# 模型评估
# 模型评估
def evaluate(model, dataloader, n_clusters, plot=False, cluster=["KMeans"], cluster_params={}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    embeddings = []
    labels = []

    for batch in dataloader:
        batch = batch.to(device)
        _, emb = model(batch.x, batch.edge_index)
        embeddings.append(emb.cpu().detach().numpy())
        labels.extend(batch.y.cpu().detach().numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    scores = {}

    # KMeans 聚类
    if "KMeans" in cluster:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pred = kmeans.fit_predict(embeddings)
        scores["kmeans_ari"] = adjusted_rand_score(labels, pred)
        scores["kmeans_nmi"] = normalized_mutual_info_score(labels, pred)
        if plot:
            pca = PCA(n_components=2).fit_transform(embeddings)
            plt.scatter(pca[:, 0], pca[:, 1], c=pred, cmap="viridis")
            plt.title("KMeans Clustering")
            plt.show()

    # Leiden 聚类
    if "Leiden" in cluster:
        resolution = cluster_params.get("Leiden", {}).get("resolution", 1.0)
        adata = sc.AnnData(embeddings)
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
        sc.tl.leiden(adata, resolution=resolution, flavor="igraph", directed=False)
        leiden_pred = adata.obs["leiden"].astype("int").values

        scores["leiden_pred"] = leiden_pred
        scores["leiden_ari"] = adjusted_rand_score(labels, leiden_pred)
        scores["leiden_nmi"] = normalized_mutual_info_score(labels, leiden_pred)

    return scores
# 主程序
if __name__ == "__main__":
    path = "/Users/jacksonzhang/Desktop/program/deep_learning/Camp.h5"
    data = sc.read_h5ad(path)
    X_all = data.X
    y_all = data.obs.values[:, 0]
    Y = np.array(y_all)
    X = np.array(X_all.todense() if hasattr(X_all, "todense") else X_all)
    genes_idx, cells_idx = filter_data(X, highly_genes=3000)
    X = X[cells_idx][:, genes_idx]
    Y = Y[cells_idx]
    edge_index, node_feats, labels = make_graph_pyg(X, Y)
    graph_data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
    )
    train_ids = (graph_data.y != -1).nonzero(as_tuple=False).view(-1)
    sampler_loader = NeighborLoader(graph_data, input_nodes=train_ids, num_neighbors=[-1], batch_size=128, shuffle=True)
    from models import GCNAE
    model = GCNAE(
        in_feats=50,
        n_hidden=200,
        n_layers=1,
        activation=F.relu,
        dropout=0.1,
        hidden=[300],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    try:
        scores = train(
            model, optimizer, n_epochs=10, dataloader=sampler_loader,
            n_clusters=len(np.unique(Y)), save=True, save_path="./output",
            plot=False, cluster=["Leiden"], cluster_params={"Leiden": {"resolution": 0.5}}
        )
        print("Training complete. Scores:", scores)
    except Exception as e:
        print(f"Error during training: {e}")