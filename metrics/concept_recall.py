import spacy


class ConceptRecall:
    def __init__(self) -> None:
        # TODO: ship the model with the code?
        self.nlp = spacy.load("en_core_web_sm")

    def compute(self, keywords, predictions):
        """
        Compute the proportion of keywords appearing in predictions.

        Parameters
        ----------
        keywords : string[][]
            List of lists of keywords.
        predictions : string[]
            List of predictions. Needs to be of the same length as keywords.

        Returns
        -------
        float
            The proportion of keywords appearing in the predictions.
        """
        assert len(keywords) == len(
            predictions
        ), f"`keywords` and `predictions` must be of equal length, got {len(keywords)} and {len(predictions)}"

        keywords_total = 0
        keywords_found = 0

        keyword_strs = [" ".join(kw_list) for kw_list in keywords]

        kw_docs = self.nlp.pipe(keyword_strs)
        pred_docs = self.nlp.pipe(predictions)

        for kw, pred in zip(kw_docs, pred_docs):
            keywords_total += len(kw)
            # get all lemmas in the prediction
            lemmata = [tok_pred.lemma for tok_pred in pred]
            # for each keyword lemma, check if it exists in the document
            for tok_kw in kw:
                if tok_kw.lemma in lemmata:
                    keywords_found += 1

        return {
            "recall": keywords_found / keywords_total,
            "keywords_total": keywords_total,
            "keywords_found": keywords_found,
        }
