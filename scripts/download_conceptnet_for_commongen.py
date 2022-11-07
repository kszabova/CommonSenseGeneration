import argparse
import csv

from datasets import load_dataset

from ..utils.conceptnet import Conceptnet


def write_csv_line(relation, type, csv_writer):
    if type == "sentence_only":
        csv_writer.writerow([relation["sentence"]])
    if type == "complete":
        csv_writer.writerow([relation["start"], relation["end"], relation["sentence"]])


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", help="Process words from 'train'"
    )
    parser.add_argument("--val", action="store_true", help="Process words from 'val'")
    parser.add_argument("--test", action="store_true", help="Process words from 'test'")
    parser.add_argument(
        "--filename", type=str, required=True, help="File where to store the results"
    )
    parser.add_argument(
        "--type",
        choices=["sentence_only", "complete"],
        required=True,
        help="How to write results",
    )
    parser.add_argument("--log", action="store_true", help="Log progress")
    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    concepts = set()

    commongen = load_dataset("common_gen")
    splits = [commongen["train"], commongen["validation"], commongen["test"]]
    processed_splits = [args.train, args.val, args.test]
    for split, is_processed in zip(splits, processed_splits):
        if is_processed:
            for example in split:
                for concept in example["concepts"]:
                    concepts.add(concept)

    conceptnet = Conceptnet()
    with open(args.filename, "w") as file:
        writer = csv.writer(file)
        for i, concept in enumerate(concepts):
            if args.log and i % 100 == 0:
                print(f"Processed {i}/{len(concepts)} concepts")
            relations = conceptnet.query_api(concept)
            for relation in relations:
                write_csv_line(relation, args.type, writer)

