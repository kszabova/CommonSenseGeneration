import argparse
import csv
import json


def _add_to_dict(dict, key1, key2, sentence, type):
    if type == "one_keyword":
        # symmetric relation
        dict.setdefault(key1, {"sentences": []})["sentences"].append(sentence)
        dict.setdefault(key2, {"sentences": []})["sentences"].append(sentence)
    elif type == "both_keywords":
        # symmetric relation
        dict.setdefault(key1, {}).setdefault(key2, {"sentences": []})[
            "sentences"
        ].append(sentence)
        dict.setdefault(key2, {}).setdefault(key1, {"sentences": []})[
            "sentences"
        ].append(sentence)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, help="File containing the downloaded conceptnet relations"
    )
    parser.add_argument("--dest", type=str, help="File to store the results")
    parser.add_argument(
        "--type",
        choices=["one_keyword", "both_keywords"],
        help="Whether the dictionary should contain just one key or both",
    )

    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    result = {}
    with open(args.csv, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            _add_to_dict(result, *row, args.type)

    with open(args.dest, "w") as file:
        file.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
