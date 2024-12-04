from dataclasses import dataclass
from typing import Literal, Optional

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainerConfig:
    trainer_type: Literal["accelerate_megatron"]
    dataset_config: DatasetConfig
    model_config: ModelConfig
    per_device_batch_size: int = 8
    learning_rate: float = 5e-05
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 1
    checkpointing_steps: int = 1000
    report_to: Literal["wandb", "none"] = "wandb"
    output_dir: str = "./output"
    num_warmup_steps: int = 0
    run_name: str
