import argparse

from utils.data_mangling import *


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="common_gen")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dest_file", type=str, default="data/error_fixer_data.data")
    parser.add_argument("--include_concept_completeness", action="store_true")
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    nlp = None
    # nlp must be instantiated if we want a column determining
    # whether the "output" has all input concepts present
    if args.include_concept_completeness:
        nlp = get_nlp_object()

    data_orig = get_data(args.dataset, args.split, args.include_concept_completeness)
    vocab = gather_vocab(data_orig)
    data_with_replacement = get_dataset_with_replacement(
        data_orig, vocab, args.include_concept_completeness, nlp=nlp
    )
    data_with_deletion = get_dataset_with_deletion(
        data_orig, args.include_concept_completeness, nlp=nlp
    )
    data = datasets.concatenate_datasets(
        [data_orig, data_with_replacement, data_with_deletion]
    )
    data = data.train_test_split(test_size=0.1)
    data.save_to_disk(args.dest_file)


if __name__ == "__main__":
    main()
