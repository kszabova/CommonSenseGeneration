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
        help="Source files to summarize",
    )
    parser.add_argument(
        "-d",
        "--dest",
        type=str,
        required=True,
        help="Destination file to write summary to",
    )
    return parser


def get_json_data(source):
    with open(source, "r") as f:
        return json.load(f).get("generated_sentences", {})


def get_all_keys(objects):
    keys = set()
    for obj in objects:
        keys.update(obj.keys())
    return keys


def get_summary(objects):
    summary = {}
    keys = get_all_keys(objects)
    for key in keys:
        summary[key] = []
        for obj in objects:
            summary[key].extend(obj.get(key, {}).get("predictions", []))
    return summary


def main():
    parser = get_argparser()
    args = parser.parse_args()

    sources = args.sources
    objects = [get_json_data(source) for source in sources]
    summary = get_summary(objects)

    with open(args.dest, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
