import evaluate


class GranularBLEU:
    def __init__(self) -> None:
        self.data = {}
        self.bleu = evaluate.load("sacrebleu")

    def compute(self, concepts, predictions, references):
        """
        Computes sacreBLEU of the dataset split by the number of concepts.

        Parameters
        ----------
        concepts : string[][]
            List of lists of concepts.
        predictions : string[]
            List of predictions. Must be of the same length as `concepts`.
        references : string[][]
            List of references. Must be of the same length as `predictions`.
            Each element of `references` must be of the same length.

        Returns
        -------
        dict[]
            sacreBLEU split by the number of given concepts
        """
        assert len(concepts) == len(
            predictions
        ), "The number of concepts doesn't match the number of predictions"
        assert len(predictions) == len(
            references
        ), "The number  of predictions doesn't match the number of references"
        assert (
            len(set([len(ref) for ref in references])) == 1
        ), "Each set of references must be of the same length"

        for conc, pred, ref in zip(concepts, predictions, references):
            no_concepts = len(conc)
            data = self.data.setdefault(
                no_concepts, {"predictions": [], "references": []}
            )
            data["predictions"].append(pred)
            data["references"].append(ref)

        bleu_results = []
        for no_concepts, data in self.data.items():
            result = self.bleu.compute(
                predictions=data["predictions"], references=data["references"]
            )
            bleu_results.append(
                {"number_concepts": no_concepts, "score": result["score"]}
            )

        return bleu_results
