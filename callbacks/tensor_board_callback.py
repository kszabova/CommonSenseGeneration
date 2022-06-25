from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger


class TensorBoardCallback(Callback):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        self.tb_logger = TensorBoardLogger("tb_logs", name=self.model_name)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        self.tb_logger.log_metrics(outputs["log"], batch_idx)
