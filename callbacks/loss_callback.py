from pytorch_lightning.callbacks import Callback

import logging


class LossCallback(Callback):
    logger = logging.getLogger("metrics")

    def __init__(self, log_interval) -> None:
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        if batch_idx % self.log_interval == 0:
            self.logger.info(
                f"STEP {trainer.global_step} Loss: {trainer.callback_metrics['loss']}"
            )
