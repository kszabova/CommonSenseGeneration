import networkx as nx
import json

from tqdm import tqdm
from datasets import load_dataset

concept_path = "../data/concept.txt"
relation_path = "../data/relation.txt"
cpnet_graph_path = "../data/conceptnet.graph"

PATTERNS = {
    "antonym": "{c1} is the opposite of {c2}",
    "atlocation": "{c1} can be found at {c2}",
    "capableof": "{c1} is capable of {c2}",
    "causes": "{c1} causes {c2}",
    "createdby": "{c1} is created by {c2}",
    "isa": "{c1} is a {c2}",
    "desires": "{c1} wants {c2}",
    "hassubevent": "Something that might happen while {c1} is {c2}",
    "partof": "{c1} is a part of {c2}",
    "hascontext": "{c1} happens in the context of {c2}",
    "hasproperty": "{c1} has the property {c2}",
    "madeof": "{c1} is made of {c2}",
    "notcapableof": "{c1} is not capable of {c2}",
    "notdesires": "{c1} does not desire {c2}",
    "receivesaction": "{c1} can be {c2}",
    "relatedto": "{c1} is related to {c2}",
    "usedfor": "{c1} is used for {c2}",
}

concept2id = None
relation2id = None
id2relation = None
id2concept = None
concepts = None


def load_resources():
    global concept2id, relation2id, id2relation, id2concept, concepts
    concept2id = {}
    id2concept = {}
    with open(concept_path, "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(relation_path, "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")

    concepts = set()

    commongen = load_dataset("common_gen")
    splits = [commongen["train"], commongen["validation"], commongen["test"]]
    for split in splits:
        for example in split:
            concepts.add(";".join(example["concepts"]))
    print("concepts done")


def convert_relation_to_sentence(c1_id, c2_id, relation_id):
    relation = id2relation[relation_id]
    c1, c2 = id2concept[c1_id], id2concept[c2_id]
    pattern = PATTERNS[relation]
    return pattern.format(c1=c1, c2=c2)


def get_sentences_from_path(path, G):
    sentences = []
    for edge in zip(path[:-1], path[1:]):
        relations = G.get_edge_data(*edge).values()
        for relation_data in relations:
            relation = relation_data["rel"]
            sentence = convert_relation_to_sentence(*edge, relation)
            sentences.append(sentence)
    return sentences


if __name__ == "__main__":
    load_resources()

    subgraphs = {}
    G = nx.read_gpickle(cpnet_graph_path)
    for concept_key in tqdm(concepts, desc="processing concepts"):
        concept_tuple = concept_key.split(";")
        subgraphs.setdefault(concept_key, {})
        for i, c1 in enumerate(concept_tuple[:-1]):
            for c2 in concept_tuple[i + 1 :]:
                pair_key = f"{c1} {c2}"
                subgraphs[concept_key].setdefault(pair_key, [])
                c1_id, c2_id = concept2id.get(c1), concept2id.get(c2)
                if not c1_id or not c2_id:
                    continue
                if not c1_id in G or not c2_id in G:
                    continue
                if nx.has_path(G, c1_id, c2_id):
                    shortest_path = nx.shortest_path(G, c1_id, c2_id)
                    sentences = get_sentences_from_path(shortest_path, G)
                    subgraphs[concept_key][pair_key].extend(sentences)
                if nx.has_path(G, c2_id, c1_id):
                    shortest_path = nx.shortest_path(G, c2_id, c1_id)
                    sentences = get_sentences_from_path(shortest_path, G)
                    subgraphs[concept_key][pair_key].extend(sentences)

    with open("../data/conceptnet_subgraphs.json", "w") as f:
        f.write(json.dumps(subgraphs, indent=4))

