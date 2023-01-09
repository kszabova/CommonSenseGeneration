from pytorch_lightning import Callback


class SavePretrainedModelCallback(Callback):
    def __init__(self, checkpoint_dir) -> None:
        self.checkpoint_dir = checkpoint_dir

    # def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    #     print("SAVING CKPT")
    #     pl_module.model.save_pretrained("./checkpoints/test")

    def on_fit_end(self, trainer, pl_module):
        pl_module.model.save_pretrained(self.checkpoint_dir)
