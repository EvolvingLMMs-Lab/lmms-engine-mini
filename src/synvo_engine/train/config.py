from dataclasses import dataclass
from typing import Literal, Optional

import transformers

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: Optional[bool] = False


@dataclass
class TrainerConfig:
    trainer_type: Literal["accelerate_megatron", "hf_trainer"]
    dataset_config: DatasetConfig
    model_config: ModelConfig
    trainer_args: TrainingArguments
