import json
import itertools

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from argparse import ArgumentParser


def get_argparser():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--data_path", type=str, help="Path to data")
    return parser


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model


def load_data(data_path):
    with open(data_path) as f:
        data = json.load(f)
    return data["generated_sentences"]


def create_input_ids(tokenizer, concepts, sentence):
    concepts_string = " ".join(concepts)
    input_string = f"{concepts_string} {tokenizer.sep_token} {sentence}"
    return tokenizer(input_string, padding="max_length", return_tensors="pt")


def generate_example(tokenizer, model, concepts, sentence):
    tokenizer_output = create_input_ids(tokenizer, concepts, sentence)
    output_ids = model.generate(**tokenizer_output)
    output_string = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_string


def main():
    parser = get_argparser()
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    data = load_data(args.data_path)

    for example in data.values():
        concepts = example["concepts"]
        sentence = example["base_predictions"][0]
        print(f"Concepts: {concepts}")
        print(f"Input sentence: {sentence}")
        for concept_permutation in list(itertools.permutations(concepts)):
            output_string = generate_example(
                tokenizer, model, concept_permutation, sentence
            )
            print(output_string)
        # wait
        if input("Press ENTER to continue or type 'exit'\n") == "exit":
            break


if __name__ == "__main__":
    main()
