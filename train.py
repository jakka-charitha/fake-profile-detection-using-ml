import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("combined_users.csv")

feature_columns = [
    'fav_number', 'statuses_count', 'followers_count', 'friends_count',
    'favourites_count', 'listed_count', 'geo_enabled', 'utc_offset', 'protected', 'verified'
]

X = df[feature_columns].fillna(0)
y = torch.tensor(df['label'].values, dtype=torch.long)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x = torch.tensor(X_scaled, dtype=torch.float)

similarity = cosine_similarity(X_scaled)
threshold = 0.85

edges = np.array(np.where(similarity > threshold))
edges = edges[:, edges[0] != edges[1]]
edge_index = torch.tensor(edges, dtype=torch.long)

num_nodes = x.size(0)
indices = np.arange(num_nodes)

train_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=y, random_state=42
)
train_idx, val_idx = train_test_split(
    train_idx, test_size=0.25, stratify=y[train_idx], random_state=42
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data = Data(x=x, edge_index=edge_index, y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GraphSAGE(x.size(1)).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data)
    pred = out[mask].argmax(dim=1)
    true = data.y[mask]
    return pred.cpu(), true.cpu(), out.cpu()

epochs = 100
for epoch in range(1, epochs + 1):
    loss = train()
    if epoch % 10 == 0:
        pred, true, _ = evaluate(data.val_mask)
        acc = (pred == true).sum().item() / len(true)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

pred_test, true_test, logits = evaluate(data.test_mask)

print("\n=== Test Classification Report ===")
print(classification_report(
    true_test,
    pred_test,
    target_names=["Real Account", "Fake Account"]
))

def visualize_subgraph(data, num_nodes=150):
    edge_index = data.edge_index.cpu().numpy()
    labels = data.y.cpu().numpy()

    selected_nodes = random.sample(range(data.num_nodes), num_nodes)
    selected_nodes = set(selected_nodes)

    G = nx.Graph()

    for n in selected_nodes:
        G.add_node(n, label=labels[n])

    for src, dst in edge_index.T:
        if src in selected_nodes and dst in selected_nodes:
            G.add_edge(src, dst)

    colors = ['red' if G.nodes[n]['label'] == 1 else 'blue' for n in G.nodes]

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos,
            node_color=colors,
            node_size=40,
            edge_color='gray',
            alpha=0.6)

    plt.title("GraphSAGE Fake Account Detection")
    plt.show()

visualize_subgraph(data)

@torch.no_grad()
def visualize_embeddings(data, model):
    model.eval()
    embeddings = model.conv1(data.x, data.edge_index).cpu().numpy()
    labels = data.y.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        emb_2d[labels == 0, 0],
        emb_2d[labels == 0, 1],
        c='blue', label='Real', alpha=0.6
    )
    plt.scatter(
        emb_2d[labels == 1, 0],
        emb_2d[labels == 1, 1],
        c='red', label='Fake', alpha=0.6
    )

    plt.legend()
    plt.title("GraphSAGE Learned Node Embeddings (t-SNE)")
    plt.show()

visualize_embeddings(data, model)
import joblib
torch.save(model.state_dict(), "graphsage_fake_detector.pth")
joblib.dump(scaler, "feature_scaler.pkl")
