###
# This script trains a link prediction model
# that predicts whether two concepts occur in the same sentence.
###

import argparse
import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score

from utils.conceptnet_gnn import ConceptnetGNN
from models.edge_prediction_gnn_model import EdgePredictionModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--edge_file", type=str, help="Where the pairs of co-occurring words are stored"
)

args = parser.parse_args()

conceptnet = ConceptnetGNN("conceptnet.graph")
conceptnet.add_edges(args.edge_file)

# split edges into train and validation
transform = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0,
    disjoint_train_ratio=0.3,
    add_negative_train_samples=False,
    edge_types=("concept", "samesentence", "concept"),
    rev_edge_types=("concept", "samesentence", "concept"),
)
train_data, val_data, test_data = transform(conceptnet.conceptnet_data)

# training

# Define seed edges
edge_label_index = train_data["concept", "samesentence", "concept"].edge_label_index
edge_label = train_data["concept", "samesentence", "concept"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[10] * 2,
    edge_label_index=(("concept", "samesentence", "concept"), edge_label_index),
    edge_label=edge_label,
    neg_sampling="binary",
    batch_size=128,
    shuffle=True,
)

model = EdgePredictionModel(64, conceptnet.conceptnet_data)

# train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 4):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["concept", "samesentence", "concept"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

# evaluation

# Define the validation seed edges:
edge_label_index = val_data["concept", "samesentence", "concept"].edge_label_index
edge_label = val_data["concept", "samesentence", "concept"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[10] * 2,
    edge_label_index=(("concept", "samesentence", "concept"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=False,
)

# evaluate the model
preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")

torch.save(model, "edge_prediction_model.pt")
