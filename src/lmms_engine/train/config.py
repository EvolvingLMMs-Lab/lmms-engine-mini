from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Union

import transformers
from peft import LoraConfig as PeftLoraConfig
from trl import DPOConfig, GRPOConfig

from ..datasets import DatasetConfig
from ..models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_muon: Optional[bool] = False
    freeze_modules: Optional[List[str]] = None
    only_save_mm_adapter: Optional[bool] = False
    use_lora: Optional[bool] = False
    use_rmpad: Optional[bool] = False


@dataclass
class DPOArguments(DPOConfig):
    freeze_modules: Optional[List[str]] = None


@dataclass
class GRPOArguments(GRPOConfig):
    freeze_modules: Optional[List[str]] = None
    reward_funcs: Optional[List[str]] = None


TrainingArgumentType = Union[TrainingArguments, DPOArguments, GRPOArguments]


@dataclass
class LoraConfig(PeftLoraConfig):
    adapter_name: str = "default"


@dataclass
class TrainerConfig:
    trainer_type: Literal["accelerate_megatron", "hf_trainer"]
    dataset_config: DatasetConfig
    model_config: ModelConfig
    trainer_args: TrainingArgumentType
    trainer_args_type: Literal["sft", "dpo", "grpo"] = "sft"
    lora_configs: Optional[List[LoraConfig]] = None

    def to_dict(self):
        trainer_args_dict = self.trainer_args.to_dict()
        lora_configs = (
            [lora_config.to_dict() for lora_config in self.lora_configs]
            if self.lora_configs
            else None
        )
        final_dict = asdict(self)
        final_dict["trainer_args"] = trainer_args_dict
        final_dict["lora_configs"] = lora_configs
        return final_dict
