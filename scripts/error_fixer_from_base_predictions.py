import json
import torch
import evaluate

from argparse import ArgumentParser

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from metrics import ConceptRecall


def _prepare_refs_for_sacrebleu(references):
    lengths = set([len(ref_list) for ref_list in references])
    max_length = max(lengths)
    sacrebleu_references = []
    for ref_list in references:
        cur_length = len(ref_list)
        sacrebleu_references.append(
            ref_list + ["" for _ in range(max_length - cur_length)]
        )
    return sacrebleu_references


parser = ArgumentParser()
parser.add_argument("--base_predictions", type=str, required=True)
parser.add_argument("--error_fixer_model", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

TOKENIZER = AutoTokenizer.from_pretrained("facebook/bart-base")

with open(args.base_predictions) as f:
    base_predictions = json.load(f)

error_fixer_model = AutoModelForSeq2SeqLM.from_pretrained(args.error_fixer_model)

device = "cuda" if torch.cuda.is_available() else "cpu"
generated_sentences = {}
for value in tqdm(base_predictions["generated_sentences"].values()):
    concatted_concepts_and_output = (
        f"{' '.join(value['concepts'])} {TOKENIZER.cls_token} {value['predictions'][0]}"
    )
    input_ids = TOKENIZER(concatted_concepts_and_output, return_tensors="pt")[
        "input_ids"
    ]
    output = error_fixer_model.generate(input_ids.to(device))
    decoded = TOKENIZER.decode(output[0], skip_special_tokens=True)
    generated_sentences[" ".join(value["concepts"])] = {
        "base_predictions": value["predictions"],
        "references": value["references"],
        "concepts": value["concepts"],
        "predictions": [decoded],
        "changed_output": [value["predictions"][0] != decoded],
    }

# compute BLEU
predictions = []
references = []
concepts = []
for value in generated_sentences.values():
    predictions.append(value["predictions"][0])
    references.append(value["references"])
    concepts.append(value["concepts"])

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_references = _prepare_refs_for_sacrebleu(references)
sacrebleu_results = sacrebleu.compute(
    predictions=predictions, references=sacrebleu_references
)

unrolled_refs = []
unrolled_preds = []
for refs, pred in zip(references, predictions):
    unrolled_refs.extend(refs)
    unrolled_preds.extend([pred] * len(refs))

rouge = evaluate.load("rouge")
rouge_results = rouge.compute(predictions=unrolled_preds, references=unrolled_refs)

meteor = evaluate.load("meteor")
meteor_results = meteor.compute(predictions=unrolled_preds, references=unrolled_refs)

concept_recall = ConceptRecall()
concept_recall_results = concept_recall.compute(concepts, predictions)

with open(args.output, "w") as f:
    f.write(
        json.dumps(
            {
                "generated_sentences": generated_sentences,
                "sacrebleu": sacrebleu_results,
                "rouge": rouge_results,
                "meteor": meteor_results,
                "concept_recall": concept_recall_results,
            },
            indent=4,
        )
    )
