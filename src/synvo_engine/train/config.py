from dataclasses import dataclass
from typing import Literal, Optional

from transformers import TrainingArguments

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainerConfig:
    trainer_type: Literal["accelerate_megatron", "hf_trainer"]
    dataset_config: DatasetConfig
    model_config: ModelConfig
    trainer_args: TrainingArguments
