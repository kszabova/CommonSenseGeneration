import random
import datasets
import spacy


def get_data(dataset, split, keep_concepts=False, *, nlp=None):
    data = datasets.load_dataset(dataset, split=split)
    if not keep_concepts:
        data = data.remove_columns(["concept_set_idx", "concepts"])
    else:
        data = data.remove_columns(["concept_set_idx"])
    data = data.rename_column("target", "input")
    data = data.map(lambda example: {"output": example["input"]})
    if keep_concepts:
        if not nlp:
            raise ValueError("Must pass in nlp object if keeping concepts")
        data = data.map(
            lambda example: {
                "contains_all_concepts": _count_concepts_in_output(
                    example["input"], example["concepts"], nlp
                )
            }
        )
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


def get_dataset_with_replacement(
    data, vocab, include_concept_completeness=False, *, nlp=None
):
    new_data = {"input": [], "output": []}
    if include_concept_completeness:
        if not nlp:
            raise ValueError(
                "Must pass in nlp object if including concept completeness"
            )
        new_data["contains_all_concepts"] = []
        new_data["concepts"] = []
    for item in data:
        replacement_sentence = random_word_replacement(item["input"], vocab, 5)
        new_data["input"].append(replacement_sentence)
        new_data["output"].append(item["input"])
        if include_concept_completeness:
            new_data["contains_all_concepts"].append(
                _count_concepts_in_output(replacement_sentence, item["concepts"], nlp)
            )
            new_data["concepts"].append(item["concepts"])
    return datasets.Dataset.from_dict(new_data)


def get_dataset_with_deletion(data, include_concept_count=False, *, nlp=None):
    new_data = {"input": [], "output": []}
    if include_concept_count:
        if not nlp:
            raise ValueError(
                "Must pass in nlp object if including concept completeness"
            )
        new_data["contains_all_concepts"] = []
        new_data["concepts"] = []
    for item in data:
        deletion_sentence = random_word_deletion(item["input"], 5)
        new_data["input"].append(deletion_sentence)
        new_data["output"].append(item["input"])
        if include_concept_count:
            new_data["contains_all_concepts"].append(
                _count_concepts_in_output(deletion_sentence, item["concepts"], nlp)
            )
            new_data["concepts"].append(item["concepts"])
    return datasets.Dataset.from_dict(new_data)


def get_nlp_object():
    return spacy.load("en_core_web_sm")


def _count_concepts_in_output(output, concepts, nlp):
    concept_doc = nlp(" ".join(concepts))
    concept_lemmas = set([token.lemma for token in concept_doc])
    output_doc = nlp(output)
    for token in output_doc:
        if token.lemma in concept_lemmas:
            concept_lemmas.remove(token.lemma)
    # return True if all concepts were found in the output
    return int(len(concept_lemmas) == 0)
