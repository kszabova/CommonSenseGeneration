class ModelPipeline:
    def __init__(self, models, initial_data, input_fns, output_fns):
        self.models = models
        self.data = initial_data
        self.input_fns = input_fns
        self.output_fns = output_fns

    def model_step(self, model, prev_outputs, input_fn, output_fn):
        return output_fn(model.generate(input_fn(prev_outputs)), prev_outputs)

    def run_pipeline(self):
        for data in self.data:
            prev_outputs = data
            for model, input_fn, output_fn in zip(
                self.models, self.input_fns, self.output_fns
            ):
                prev_outputs = self.model_step(model, prev_outputs, input_fn, output_fn)
            yield prev_outputs
