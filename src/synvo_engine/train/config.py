from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import transformers
from trl import DPOConfig, GRPOConfig

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: Optional[bool] = False
    freeze_modules: Optional[List[str]] = None
    only_save_mm_adapter: Optional[bool] = False


@dataclass
class DPOArguments(DPOConfig):
    freeze_modules: Optional[List[str]] = None


@dataclass
class GRPOArguments(GRPOConfig):
    freeze_modules: Optional[List[str]] = None
    reward_funcs: Optional[List[str]] = None


TrainingArgumentType = Union[TrainingArguments, DPOArguments, GRPOArguments]


@dataclass
class TrainerConfig:
    trainer_type: Literal["accelerate_megatron", "hf_trainer"]
    dataset_config: DatasetConfig
    model_config: ModelConfig
    trainer_args: TrainingArgumentType
    trainer_args_type: Literal["sft", "dpo"] = "sft"
