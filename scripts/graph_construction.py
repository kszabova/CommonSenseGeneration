###
# Adapted after https://github.com/cdjhz/multigen
###

import math
from tqdm import tqdm
import nltk

from argparse import ArgumentParser

from utils.conceptnet import Conceptnet

nltk.download("stopwords")
nltk_stopwords = nltk.corpus.stopwords.words("english")
nltk_stopwords += [
    "like",
    "gone",
    "did",
    "going",
    "would",
    "could",
    "get",
    "in",
    "up",
    "may",
    "wanter",
]

conceptnet_en_path = "./data/conceptnet-english.csv"
conceptnet_graph_path = "./data/conceptnet.graph"


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--skip_relatedto", action="store_true", help="Skip the relation /r/RelatedTo."
    )
    parser.add_argument(
        "--graph_file_path", type=str, help="Where to store the .graph file."
    )
    return parser


def add_edge(conceptnet, start, end, rel, weight):
    kwargs = {
        "mode": "add_edge",
        "start": start,
        "end": end,
        "rel": rel,
        "weight": weight,
    }
    conceptnet.create_local(**kwargs)


def save_to_file(conceptnet, filepath):
    kwargs = {"mode": "save", "filepath": filepath}
    conceptnet.create_local(**kwargs)


def main():
    global blacklist

    parser = get_parser()
    args = parser.parse_args()
    conceptnet = Conceptnet()
    with open(conceptnet_en_path, "r", encoding="utf8") as f:
        lines = f.readlines()

        def not_save(cpt):
            for t in cpt.split("_"):
                if t in nltk_stopwords:
                    return True
            return False

        for line in tqdm(lines, desc="saving to graph"):
            ls = line.strip().split("\t")
            rel = conceptnet.query_resource("relation2id", ls[0])
            subj = conceptnet.query_resource("concept2id", ls[1])
            obj = conceptnet.query_resource("concept2id", ls[2])
            weight = float(ls[3])
            if ls[0] == "hascontext":
                continue
            if not_save(ls[1]) or not_save(ls[2]):
                continue
            if ls[0] == "relatedto" or ls[0] == "antonym":
                if args.skip_relatedto:
                    continue
                weight -= 0.3
            if subj == obj:  # delete loops
                continue
            weight = 1 + float(math.exp(1 - weight))
            add_edge(conceptnet, subj, obj, rel, weight)

    save_to_file(conceptnet, args.graph_file_path)


if __name__ == "__main__":
    main()

