###
# Utils for working with conceptnet in pytorch geometric
###

import torch

from .conceptnet import Conceptnet


class ConceptnetGNN:
    def __init__(self, conceptnet_path):
        self.conceptnet_path = conceptnet_path

        self.conceptnet = self.get_conceptnet()
        self.conceptnet_data = self.conceptnet.to_pyg_data()

    def get_conceptnet(self):
        conceptnet = Conceptnet(graph_filename=self.conceptnet_path)
        return conceptnet

    def add_edges(self, edges_path):
        edges_set = set()
        with open(edges_path, "r") as file:
            for line in file:
                concept1, concept2 = line.split()
                edges_set.add((concept1, concept2))
        edges_from = []
        edges_to = []
        for edge in edges_set:
            idx1 = self.conceptnet.resources.concept2id.get(edge[0])
            idx2 = self.conceptnet.resources.concept2id.get(edge[1])
            if idx1 is not None and idx2 is not None:
                edges_from.append(idx1)
                edges_to.append(idx2)
        edges_from_torch = torch.tensor(edges_from)
        edges_to_torch = torch.tensor(edges_to)
        edges = torch.stack([edges_from_torch, edges_to_torch], dim=0)
        self.conceptnet_data["concept", "samesentence", "concept"].edge_index = edges
