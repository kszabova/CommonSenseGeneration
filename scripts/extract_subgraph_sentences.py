import json

from tqdm import tqdm
from datasets import load_dataset
from argparse import ArgumentParser

from utils.conceptnet import Conceptnet

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

concept_set = None


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--graph_filename", type=str, help="Location of the stored conceptnet graph."
    )
    parser.add_argument("--tgt_path", type=str, help="Where the output will be stored.")
    return parser


def load_concept_set():
    global concept_set

    concept_set = set()

    commongen = load_dataset("common_gen")
    splits = [commongen["train"], commongen["validation"], commongen["test"]]
    for split in splits:
        for example in split:
            concepts = [concept.replace(" ", "_") for concept in example["concepts"]]
            concept_set.add(" ".join(concepts))
    print("Finished loading concept set")


def query_edge(conceptnet, *edge):
    query = {"mode": "edge", "start": edge[0], "end": edge[1]}
    return conceptnet.query_local(**query)


def query_shortest_path(conceptnet, c1, c2):
    query = {"mode": "shortest_path", "start": c1, "end": c2}
    return conceptnet.query_local(**query)


def convert_relation_to_sentence(c1_id, c2_id, relation_id, conceptnet):
    relation = conceptnet.query_resource("id2relation", relation_id)
    c1 = conceptnet.query_resource("id2concept", c1_id)
    c2 = conceptnet.query_resource("id2concept", c2_id)
    pattern = PATTERNS[relation]
    return pattern.format(c1=c1, c2=c2)


def get_sentences_from_path(path, conceptnet):
    sentences = []
    for edge in zip(path[:-1], path[1:]):
        relations = query_edge(conceptnet, *edge)
        max_relation = max(relations, key=lambda dict: dict["weight"])["rel"]
        sentence = convert_relation_to_sentence(*edge, max_relation, conceptnet)
        sentences.append(sentence)
    return sentences


def main():
    parser = get_args()
    args = parser.parse_args()

    load_concept_set()
    conceptnet = Conceptnet(graph_filename=args.graph_filename)

    subgraphs = {}
    for concept_key in tqdm(concept_set, desc="processing concepts"):
        concept_tuple = concept_key.split()
        subgraphs.setdefault(concept_key, {})
        for i, c1 in enumerate(concept_tuple[:-1]):
            for c2 in concept_tuple[i + 1 :]:
                pair_key = f"{c1} {c2}"
                subgraphs[concept_key].setdefault(pair_key, [])
                c1_to_c2, c2_to_c1 = query_shortest_path(conceptnet, c1, c2)
                sentences_c1_to_c2 = (
                    get_sentences_from_path(c1_to_c2, conceptnet) if c1_to_c2 else []
                )
                sentences_c2_to_c1 = (
                    get_sentences_from_path(c2_to_c1, conceptnet) if c2_to_c1 else []
                )
                subgraphs[concept_key][pair_key].extend(
                    [sentences_c1_to_c2, sentences_c2_to_c1]
                )

    with open(args.tgt_path, "w") as f:
        f.write(json.dumps(subgraphs, indent=4))


if __name__ == "__main__":
    main()

