from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load("feature_scaler.pkl")

feature_columns = [
    'fav_number', 'statuses_count', 'followers_count', 'friends_count',
    'favourites_count', 'listed_count', 'geo_enabled',  'utc_offset', 'protected', 'verified'
]

base_df = pd.read_csv("combined_users.csv")
X_base = scaler.transform(base_df[feature_columns].fillna(0))
x_base = torch.tensor(X_base, dtype=torch.float)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(x_base.shape[1]).to(device)
model.load_state_dict(torch.load("graphsage_fake_detector.pth", map_location=device))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        user_data = {f: float(request.form[f]) for f in feature_columns}
        df_new = pd.DataFrame([user_data])
        scaled = scaler.transform(df_new)
        x_new = torch.tensor(scaled, dtype=torch.float)

        similarity = cosine_similarity(scaled, X_base)[0]
        threshold = 0.85
        connected = np.where(similarity > threshold)[0]

        src, dst = [], []
        new_idx = x_base.shape[0]

        for n in connected:
            src.extend([new_idx, n])
            dst.extend([n, new_idx])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        x_all = torch.cat([x_base, x_new], dim=0)

        data = Data(x=x_all, edge_index=edge_index).to(device)

        with torch.no_grad():
            out = model(data)
            probs = F.softmax(out[new_idx], dim=0).cpu().numpy()

        confidence_plot = plot_confidence(probs[0]*100, probs[1]*100)

        labels = base_df["label"].values
        edge_index_full = torch.tensor(
            np.where(cosine_similarity(X_base) > 0.85),
            dtype=torch.long
        )

        embedding_plot = plot_embeddings(
            model,
            x_base,
            edge_index_full,
            labels
        )

        result = {
            "label": "FAKE ACCOUNT" if np.argmax(probs) == 1 else "REAL ACCOUNT",
            "real": probs[0] * 100,
            "fake": probs[1] * 100,
            "confidence_plot": confidence_plot,
            "embedding_plot": embedding_plot
        }

    return render_template("index.html", result=result)

def plot_confidence(real, fake):
    plt.figure(figsize=(4, 4))
    plt.bar(["Real", "Fake"], [real, fake])
    plt.title("Prediction Confidence (%)")
    plt.ylabel("Confidence")
    plt.ylim(0, 100)
    plt.tight_layout()
    path = "static/plots/confidence.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_embeddings(model, x_base, edge_index, labels):
    model.eval()
    with torch.no_grad():
        emb = model.conv1(x_base.to(device), edge_index.to(device))
        emb = emb.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(6, 5))
    plt.scatter(emb_2d[labels == 0, 0], emb_2d[labels == 0, 1], alpha=0.6, label="Real")
    plt.scatter(emb_2d[labels == 1, 0], emb_2d[labels == 1, 1], alpha=0.6, label="Fake")
    plt.legend()
    plt.title("GraphSAGE Node Embeddings (t-SNE)")
    plt.tight_layout()

    path = "static/plots/embeddings.png"
    plt.savefig(path)
    plt.close()
    return path

if __name__ == "__main__":
    app.run(debug=True)
