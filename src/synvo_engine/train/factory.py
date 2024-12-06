from .accelerate_megatron_trainer import AccelerateMegatronTrainer
from .config import TrainerConfig
from .hf_trainer import Hf_Trainer


class TrainerFactory:
    @staticmethod
    def create_trainer(config: TrainerConfig, **kwargs):
        if config.trainer_type == "accelerate_megatron":
            return AccelerateMegatronTrainer(config, **kwargs)
        elif config.trainer_type == "hf_trainer":
            return Hf_Trainer(config, **kwargs)
        else:
            raise ValueError(f"Unknown trainer type: {config.trainer_type}")
