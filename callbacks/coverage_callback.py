from pytorch_lightning.callbacks import Callback
import logging


class CoverageCallback(Callback):
    logger = logging.getLogger("coverage")

    def __init__(self, log_coverage):
        super().__init__()
        self.log_coverage = log_coverage

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.log_coverage:
            coverage = (
                pl_module.total_pairs_found / pl_module.total_keywords
                if pl_module.total_keywords
                else 0
            )
            self.logger.info(f"{coverage * 100}%")

