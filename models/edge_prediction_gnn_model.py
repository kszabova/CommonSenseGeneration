# the model is adapted from a tutorial by PyG authors
# at https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70

import torch

import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_concepts: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_concepts1 = x_concepts[edge_label_index[0]]
        edge_feat_concepts2 = x_concepts[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_concepts1 * edge_feat_concepts2).sum(dim=-1)


class EdgePredictionModel(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # We learn node embedding since we don't have any features
        self.node_embedding = torch.nn.Embedding(
            data["concept"].num_nodes, hidden_channels
        )
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "concept": self.node_embedding(data["concept"].node_id),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["concept"],
            data["concept", "samesentence", "concept"].edge_label_index,
        )
        return pred
