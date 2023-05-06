###
# This script is used to find all edges of type ("concept", "samesentence", "concept")
# for training a GNN "recommender" system.
#

import argparse
import tqdm

import spacy

from datasets import load_dataset


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_file", type=str, help="Where to store the file with saved edges."
    )
    return parser


def get_references_from_dataset(dataset, splits):
    dataset = load_dataset(dataset)
    references = []
    for split in splits:
        for example in dataset[split]:
            references.append(example["target"])
    return references


def get_lemmas_from_doc(doc):
    allowed_pos = ["NOUN", "PROPN", "ADJ", "VERB"]
    candidates = set()
    for token in doc:
        if token.pos_ in allowed_pos:
            candidates.add(token.lemma_)
    return list(candidates)


def get_pairs_from_lemmas(lemmas):
    pair_indices = [
        (i, j) for i in range(len(lemmas)) for j in range(i + 1, len(lemmas))
    ]
    pairs = set()
    for idx1, idx2 in pair_indices:
        pairs.add((lemmas[idx1], lemmas[idx2]))
        pairs.add((lemmas[idx2], lemmas[idx1]))
    return pairs


def main():
    parser = get_argparser()
    args = parser.parse_args()

    sentences = get_references_from_dataset("common_gen", ["train"])
    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(sentences)
    edges = set()
    for doc in tqdm.tqdm(docs, total=len(sentences)):
        lemmas = get_lemmas_from_doc(doc)
        pairs = get_pairs_from_lemmas(lemmas)
        edges.update(pairs)
    with open(args.out_file, "w") as file:
        for pair in edges:
            file.write(f"{pair[0]} {pair[1]}\n")


if __name__ == "__main__":
    main()

