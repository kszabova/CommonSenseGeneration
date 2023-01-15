from transformers import BartForConditionalGeneration, BartTokenizer


class Postprocess:
    """
    This class is used to postprocess the output of the model.
    """

    def __init__(self, config):
        self.type = config.postprocess_type
        self._init_data(self.type, config)

    def postprocess(self, inputs):
        """
        Postprocess a specific input or return it unchanged if postprocessing type is None.

        :param input_ids(str): The input to postprocess.
        """
        if self.type == "error_fixer":
            input_ids = self.tokenizer(inputs, return_tensors="pt")["input_ids"]
            output_ids = self.model.generate(input_ids)
            postprocessed = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            return postprocessed
        if self.type == "selection":
            raise NotImplementedError
        return inputs

    def _init_data(self, type, config):
        if type == "error_fixer":
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            self.model = BartForConditionalGeneration.from_pretrained(
                config.postprocess_model_path
            )

