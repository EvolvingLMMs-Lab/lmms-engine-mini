from dataclasses import dataclass
from typing import Literal, Optional

from transformers import TrainingArguments

from ..datasets import DatasetConfig
from ..models import ModelConfig


class TrainerConfig(TrainingArguments):
    trainer_type: Literal["accelerate_megatron"]
    dataset_config: DatasetConfig
    model_config: ModelConfig
