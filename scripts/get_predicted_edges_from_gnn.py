import argparse
import tqdm
import json

import torch

from datasets import load_dataset

from utils.conceptnet_gnn import ConceptnetGNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Where the trained link prediction model is stored",
    default="trained_models/edge_prediction_model.pt",
)
parser.add_argument(
    "--out_file",
    type=str,
    help="Where to store the predicted edges",
    default="data/predicted_edges.json",
)
args = parser.parse_args()

# load conceptnet
conceptnet = ConceptnetGNN("conceptnet.graph")
conceptnet.conceptnet_data[
    "concept", "samesentence", "concept"
].edge_index = torch.empty([2, 0], dtype=torch.long)

# load dataset
dataset = load_dataset("common_gen")

# save all concepts from the dataset
concepts = set()
for split in ["train", "validation", "test"]:
    for example in dataset[split]:
        concepts.update(set(example["concepts"]))

# find 128 concepts starting from each concept in common gen
# save to dictionary in the form of {concept_idx: [reachable concept_idxs]}
concept_dict = {}
max_reachable = 128
for concept in tqdm.tqdm(concepts):
    concept_idx = conceptnet.conceptnet.resources.concept2id.get(concept)
    if not concept_idx:
        continue
    if not conceptnet.conceptnet.graph.has_node(concept_idx):
        continue
    reachable_indices = []
    queue = [concept_idx]
    while len(reachable_indices) < max_reachable and queue:
        cur_idx = queue.pop(0)
        neighbors = list(conceptnet.conceptnet.graph.neighbors(cur_idx))
        reachable_indices.extend(neighbors)
        queue.extend(neighbors)
    concept_dict[concept_idx] = reachable_indices[:max_reachable]

# create edge label index
edges_start = []
edges_end = []
for concept_idx, reachable_indices in concept_dict.items():
    edges_start.extend([concept_idx] * len(reachable_indices))
    edges_end.extend(reachable_indices)
edge_label_index = torch.tensor([edges_start, edges_end])
conceptnet.conceptnet_data[
    "concept", "samesentence", "concept"
].edge_label_index = edge_label_index

# load trained model
model = torch.load(args.model, map_location=torch.device("cpu"))
model.eval()

predicted_words = {}
with torch.no_grad():
    preds = model(conceptnet.conceptnet_data)
    for start, end, pred in zip(edges_start, edges_end, preds):
        start_word = conceptnet.conceptnet.resources.id2concept[start]
        end_word = conceptnet.conceptnet.resources.id2concept[end]
        if pred.item() > 0.8 and pred.item() < 1.2:
            predicted_words.setdefault(start_word, []).append(end_word)

# save predicted edges
with open(args.out_file, "w") as f:
    f.write(json.dumps(predicted_words, indent=4))
