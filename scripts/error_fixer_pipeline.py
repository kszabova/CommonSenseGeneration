import argparse
import datasets
import evaluate
import json
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.model_pipeline import ModelPipeline

TOKENIZER = AutoTokenizer.from_pretrained("facebook/bart-base")


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths", nargs="+", type=str, help="Path to saved models"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset to use as input to the first model"
    )
    parser.add_argument(
        "--data_partitions", nargs="+", type=str, help="Partitions of data to use"
    )
    parser.add_argument("--output_path", type=str, help="Path to output file")
    return parser


def load_models(model_paths, device):
    models = []
    for model_path in model_paths:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        models.append(model)
    return models


def load_data(dataset_name, partitions):
    data = datasets.load_dataset(dataset_name)
    return [data[partition] for partition in partitions]


def _concat_concepts(input):
    joined_concepts = " ".join([str(concept) for concept in input["concepts"]])
    return TOKENIZER(joined_concepts, return_tensors="pt")["input_ids"]


def _add_output_from_base_model(model_output, previous_output):
    output_decoded = TOKENIZER.decode(model_output[0], skip_special_tokens=True)
    return previous_output.copy() | {"base_model_output": output_decoded}


def _concat_concepts_and_output(previous_output):
    concatted_concepts_and_output = f"{' '.join(previous_output['concepts'])} {TOKENIZER.sep_token} {previous_output['base_model_output']}"
    return TOKENIZER(concatted_concepts_and_output, return_tensors="pt")["input_ids"]


def _add_output_from_error_fixer(model_output, previous_output):
    output_decoded = TOKENIZER.decode(model_output[0], skip_special_tokens=True)
    changed_input = previous_output["base_model_output"] != output_decoded
    return previous_output.copy() | {
        "error_fixer_output": output_decoded,
        "changed_input": changed_input,
    }


def _prepare_refs_for_sacrebleu(self, references):
    lengths = set([len(ref_list) for ref_list in references])
    max_length = max(lengths)
    sacrebleu_references = []
    for ref_list in references:
        cur_length = len(ref_list)
        sacrebleu_references.append(
            ref_list + ["" for _ in range(max_length - cur_length)]
        )
    return sacrebleu_references


def main():
    parser = get_argparser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = load_models(args.model_paths, device)
    data = load_data(args.dataset, args.data_partitions)

    generated_sentences = {}

    for partition in data:
        pipeline = ModelPipeline(
            models,
            partition,
            [_concat_concepts, _concat_concepts_and_output],
            [_add_output_from_base_model, _add_output_from_error_fixer],
        )
        for output in pipeline.run_pipeline():
            conceptset_generated_sentences = generated_sentences.setdefault(
                output["concept_set_idx"],
                {
                    "references": [],
                    "predictions": [output["error_fixer_output"]],
                    "changed_output": [output["changed_input"]],
                },
            )
            conceptset_generated_sentences["references"].append(output["target"])

        # compute BLEU
        predictions = []
        references = []
        for value in generated_sentences.values():
            predictions.append(value["predictions"][0])
            references.append(value["references"])

        sacrebleu = evaluate.load("sacrebleu")
        sacrebleu_references = _prepare_refs_for_sacrebleu(references)
        sacrebleu_results = sacrebleu.compute(
            predictions=predictions, references=sacrebleu_references
        )

        with open(args.output_path, "w") as f:
            f.write(
                json.dump(
                    {
                        "generated_sentences": generated_sentences,
                        "sacrebleu": sacrebleu_results,
                    },
                    indent=4,
                )
            )


if __name__ == "__main__":
    main()
