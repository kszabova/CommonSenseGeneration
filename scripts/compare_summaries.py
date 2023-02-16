import json
import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        nargs="+",
        required=True,
        help="Source files to compare",
    )
    parser.add_argument(
        "-d",
        "--dest",
        type=str,
        required=True,
        help="Destination file to write summary to",
    )
    parser.add_argument(
        "--sort_by_concepts", action="store_true", help="Sort by number of concepts"
    )
    return parser


def get_json_data(file):
    with open(file, "r") as f:
        return json.load(f)


def get_all_keys(summaries):
    keys = set()
    for summary in summaries:
        keys.update(summary.keys())
    return keys


def get_comparison(summaries, sort=False):
    generated_sentences = []
    keys = get_all_keys(summaries)
    if sort:
        keys = sorted(keys, key=lambda k: len(k.split()))
    for key in keys:
        key_object = {"concepts": key}
        for i, summary in enumerate(summaries):
            key_object[f"file{i}"] = summary.get(key, [])
        generated_sentences.append(key_object)
    return generated_sentences


def main():
    parser = get_argparser()
    args = parser.parse_args()

    sources = args.sources
    summaries = [get_json_data(source) for source in sources]
    generated_sentences = get_comparison(summaries, args.sort_by_concepts)
    summary = {
        "generated_sentences": generated_sentences,
    }
    for i, source in enumerate(sources):
        summary[f"file{i}"] = source
    with open(args.dest, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
