from synvo_engine.train.config import TrainerConfig

from .base_trainer import BaseTrainer
from .config import TrainerConfig


class Hf_Trainer(BaseTrainer):
    def __init__(self, config: TrainerConfig) -> None:
        super().__init__(config)

    def build(self):
        return super().build()

    def run(self, **kwargs):
        return super().run(**kwargs)
