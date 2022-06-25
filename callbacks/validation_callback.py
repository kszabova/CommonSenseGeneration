from pytorch_lightning.callbacks import Callback

import json
import os
import evaluate

import gem_metrics


class ValidationCallback(Callback):
    def __init__(self, output_file, min_epochs) -> None:
        super().__init__()
        self.output_file = output_file
        self.min_epoch_idx = min_epochs - 1

        self.generated_sentences = {}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.current_epoch >= self.min_epoch_idx:
            if batch_idx == 0:
                self.generated_sentences = {}
            inputs = outputs["examples"]["src"]
            references = outputs["examples"]["ref"]
            predictions = outputs["examples"]["pred"]
            for input, ref, pred in zip(inputs, references, predictions):
                self.generated_sentences[input] = self.generated_sentences.get(
                    input, {"references": [], "predictions": []}
                )
                self.generated_sentences[input]["references"].append(ref)
                self.generated_sentences[input]["predictions"].append(pred)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.min_epoch_idx:
            try:
                os.remove(self.output_file)
            except OSError:
                pass
            predictions = []
            references = []
            for value in self.generated_sentences.values():
                predictions_set = set(value["predictions"])
                value["predictions"] = list(predictions_set)
                for prediction in predictions_set:
                    predictions.append(prediction)
                    references.append(list(value["references"]))

            bleu = evaluate.load("bleu")
            bleu_results = bleu.compute(predictions=predictions, references=references)

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

            cider_results = None
            try:
                preds_gem = gem_metrics.texts.Predictions(unrolled_preds)
                refs_gem = gem_metrics.texts.References(unrolled_refs)

                cider_results = gem_metrics.compute(
                    preds_gem, refs_gem, metrics_list=["cider"]
                )
            except:
                pass

            results = {}
            results["generated_sentences"] = self.generated_sentences
            results["metrics"] = {
                "bleu": bleu_results,
                "rouge": rouge_results,
                "meteor": meteor_results,
                "cider": cider_results,
            }

            with open(self.output_file, "w") as output_f:
                output_f.write(json.dumps(results, indent=4))
