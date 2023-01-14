import datasets
import argparse
import random


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="common_gen")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dest_file", type=str, default="data/classifier_data.data")
    return parser


def get_data(dataset, split):
    data = datasets.load_dataset(dataset, split=split)
    data = data.remove_columns(["concept_set_idx", "concepts"])
    data = data.rename_column("target", "text")
    data = data.add_column("class", [1] * len(data))
    return data


def gather_vocab(data):
    vocab = set()
    for item in data:
        vocab.update(item["text"].split())
    return vocab


def random_word_replacement(input, vocab):
    random_word = random.choice(list(vocab))
    input_split = input.split()
    random_index = random.randint(0, len(input_split) - 1)
    input_split[random_index] = random_word
    return " ".join(input_split)


def random_word_deletion(input):
    input_split = input.split()
    random_index = random.randint(0, len(input_split) - 1)
    input_split.pop(random_index)
    return " ".join(input_split)


def get_dataset_with_replacement(data, vocab):
    new_data = {"text": [], "class": [0] * len(data)}
    for item in data:
        new_data["text"].append(random_word_replacement(item["text"], vocab))
    return datasets.Dataset.from_dict(new_data)


def get_dataset_with_deletion(data):
    new_data = {"text": [], "class": [0] * len(data)}
    for item in data:
        new_data["text"].append(random_word_deletion(item["text"]))
    return datasets.Dataset.from_dict(new_data)


def main():
    parser = get_argparser()
    args = parser.parse_args()

    data_orig = get_data(args.dataset, args.split)
    vocab = gather_vocab(data_orig)
    data_with_replacement = get_dataset_with_replacement(data_orig, vocab)
    data_with_deletion = get_dataset_with_deletion(data_orig)
    data = datasets.concatenate_datasets(
        [data_orig, data_with_replacement, data_with_deletion]
    )
    data = data.train_test_split(test_size=0.1)
    data.save_to_disk(args.dest_file)


if __name__ == "__main__":
    main()
