import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = joblib.load("feature_scaler.pkl")

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

base_df = pd.read_csv("combined_users.csv")

feature_columns = [
    'fav_number', 'statuses_count', 'followers_count', 'friends_count',
    'favourites_count', 'listed_count', 'geo_enabled',  'utc_offset', 'protected', 'verified'
]

X_base = scaler.transform(base_df[feature_columns].fillna(0))
x_base = torch.tensor(X_base, dtype=torch.float)

single_user = {
    "fav_number": 73095,
    "statuses_count": 80,
    "followers_count": 33,
    "friends_count": 852,
    "favourites_count": 0,
    "listed_count": 0,
    "geo_enabled": 0,
    "utc_offset": 0,
    "protected": 0,
    "verified": 0
}

single_df = pd.DataFrame([single_user])
single_scaled = scaler.transform(single_df)
x_new = torch.tensor(single_scaled, dtype=torch.float)

similarity = cosine_similarity(single_scaled, X_base)[0]
threshold = 0.85

connected_nodes = np.where(similarity > threshold)[0]

edges_src = []
edges_dst = []

new_node_index = x_base.shape[0]

for n in connected_nodes:
    edges_src.append(new_node_index)
    edges_dst.append(n)
    edges_src.append(n)
    edges_dst.append(new_node_index)

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

x_all = torch.cat([x_base, x_new], dim=0)

data = Data(
    x=x_all,
    edge_index=edge_index
).to(device)

model = GraphSAGE(x_all.shape[1]).to(device)
model.load_state_dict(torch.load("graphsage_fake_detector.pth", map_location=device))
model.eval()

with torch.no_grad():
    out = model(data)
    probs = F.softmax(out[new_node_index], dim=0)
    pred = torch.argmax(probs).item()

label_map = {0: "REAL ACCOUNT", 1: "FAKE ACCOUNT"}

print("Prediction:", label_map[pred])
print("Confidence:")
print(f"Real : {probs[0].item():.4f}")
print(f"Fake : {probs[1].item():.4f}")
