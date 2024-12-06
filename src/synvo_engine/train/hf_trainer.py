import torch
from transformers import Trainer

from .base_trainer import BaseTrainer
from .config import TrainerConfig
from .custom import LLaVATrainer


class Hf_Trainer(BaseTrainer):
    def __init__(self, config: TrainerConfig) -> None:
        super().__init__(config)

    def build(self):
        super().build()
        self.trainer = self._build_trainer()

    def _build_trainer(self):
        trainer = LLaVATrainer(
            model=self.model,
            args=self.config.trainer_args,
            data_collator=self.train_dataset.get_collator(),
            train_dataset=self.train_dataset,
        )
        return trainer

    def run(self, **kwargs):
        self.trainer.train()
        self.trainer.save_state()
        self.safe_save_model_for_hf_trainer(
            self.trainer, self.config.trainer_args.output_dir
        )

    def safe_save_model_for_hf_trainer(self, trainer: Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        trainer.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        if trainer.deepspeed:
            trainer.save_model(output_dir)
            return

        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
