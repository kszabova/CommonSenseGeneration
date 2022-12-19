from pytorch_lightning.callbacks import Callback

import json
import os
import evaluate

import gem_metrics

from metrics import ConceptRecall, GranularBLEU


class ValidationCallback(Callback):
    def __init__(self, output_file, min_epochs) -> None:
        super().__init__()
        self.output_file = output_file
        self.min_epoch_idx = min_epochs - 1

        self.generated_sentences = {}
        # TODO get filename from config
        self.valid_file = "./data/valid_references.json"

        self.references = self._get_references()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.current_epoch >= self.min_epoch_idx:
            if batch_idx == 0:
                self.generated_sentences = {}
            inputs = outputs["examples"]["src"]
            references = outputs["examples"]["ref"]
            predictions = outputs["examples"]["pred"]
            concepts = outputs["examples"]["concepts"]
            for input, ref, pred, conc in zip(
                inputs, references, predictions, concepts
            ):
                conc_string = " ".join(conc)
                self.generated_sentences[conc_string] = self.generated_sentences.get(
                    conc_string,
                    {
                        "concepts": conc,
                        "inputs": [],
                        "references": [],
                        "predictions": [],
                    },
                )
                self.generated_sentences[conc_string]["inputs"].append(input)
                self.generated_sentences[conc_string]["references"].extend(
                    self.references.get(conc_string, [""])
                )
                self.generated_sentences[conc_string]["predictions"].append(pred)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.min_epoch_idx:
            try:
                os.remove(self.output_file)
            except OSError:
                pass
            predictions = []
            references = []
            concepts = []
            for value in self.generated_sentences.values():
                for pred in value["predictions"]:
                    predictions.append(pred)
                    references.append(value["references"])
                    concepts.append(value["concepts"])

            bleu = evaluate.load("bleu")
            bleu_results = bleu.compute(predictions=predictions, references=references)

            sacrebleu = evaluate.load("sacrebleu")
            sacrebleu_references = self._prepare_refs_for_sacrebleu(references)
            sacrebleu_results = sacrebleu.compute(
                predictions=predictions, references=sacrebleu_references
            )

            granular_bleu = GranularBLEU()
            granular_bleu_results = granular_bleu.compute(
                concepts, predictions, sacrebleu_references
            )

            unrolled_refs = []
            unrolled_preds = []
            for refs, pred in zip(references, predictions):
                unrolled_refs.extend(refs)
                unrolled_preds.extend([pred] * len(refs))

            rouge = evaluate.load("rouge")
            rouge_results = rouge.compute(
                predictions=unrolled_preds, references=unrolled_refs
            )

            meteor = evaluate.load("meteor")
            meteor_results = meteor.compute(
                predictions=unrolled_preds, references=unrolled_refs
            )

            preds_gem = gem_metrics.texts.Predictions(unrolled_preds)
            refs_gem = gem_metrics.texts.References(unrolled_refs)

            cider_results = gem_metrics.compute(
                preds_gem, refs_gem, metrics_list=["cider"]
            )

            concept_recall = ConceptRecall()
            concept_recall_results = concept_recall.compute(concepts, predictions)

            results = {}
            results["generated_sentences"] = self.generated_sentences
            results["metrics"] = {
                "bleu": bleu_results,
                "sacrebleu": sacrebleu_results,
                "granular_bleu": granular_bleu_results,
                "rouge": rouge_results,
                "meteor": meteor_results,
                "cider": cider_results,
                "concept_recall": concept_recall_results,
            }

            with open(self.output_file, "w") as output_f:
                output_f.write(json.dumps(results, indent=4))

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

    def _get_references(self):
        with open(self.valid_file, "r") as file:
            references = json.load(file)
        return references
