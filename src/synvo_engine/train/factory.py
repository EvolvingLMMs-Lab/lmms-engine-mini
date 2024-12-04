from .accelerate_megatron_trainer import AccelerateMegatronTrainer
from .config import TrainerConfig


class TrainerFactory:
    @staticmethod
    def create_trainer(config: TrainerConfig, **kwargs):
        if config.trainer_type == "accelerate_megatron":
            return AccelerateMegatronTrainer(config, **kwargs)
        else:
            raise ValueError(f"Unknown trainer type: {config.trainer_type}")
