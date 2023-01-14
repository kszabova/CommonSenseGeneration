import datasets
import argparse
import random


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="common_gen")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dest_file", type=str, default="data/error_fixer_data.data")
    return parser


def get_data(dataset, split):
    data = datasets.load_dataset(dataset, split=split)
    data = data.remove_columns(["concept_set_idx", "concepts"])
    data = data.rename_column("target", "input")
    data = data.map(lambda example: {"output": example["input"]})
    return data


def gather_vocab(data):
    vocab = set()
    for item in data:
        vocab.update(item["input"].split())
    return vocab


def random_word_replacement(input, vocab, max_replacements=1):
    input_split = input.split()
    num_replacements = random.randint(1, min(len(input_split), max_replacements))

    random_words = random.sample(list(vocab), num_replacements)
    random_indices = random.sample(range(len(input_split)), num_replacements)
    for idx, word in zip(random_indices, random_words):
        input_split[idx] = word
    return " ".join(input_split)


def random_word_deletion(input, max_deletions=1):
    input_split = input.split()
    num_deletions = random.randint(1, min(len(input_split), max_deletions))

    random_indices = random.sample(range(len(input_split)), num_deletions)
    for idx in sorted(random_indices, reverse=True):
        del input_split[idx]
    return " ".join(input_split)


def get_dataset_with_replacement(data, vocab):
    new_data = {"input": [], "output": []}
    for item in data:
        new_data["input"].append(random_word_replacement(item["input"], vocab, 5))
        new_data["output"].append(item["input"])
    return datasets.Dataset.from_dict(new_data)


def get_dataset_with_deletion(data):
    new_data = {"input": [], "output": []}
    for item in data:
        new_data["input"].append(random_word_deletion(item["input"], 5))
        new_data["output"].append(item["input"])
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
