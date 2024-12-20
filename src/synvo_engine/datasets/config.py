from dataclasses import dataclass
from typing import Literal, Union

from .processor import ProcessorConfig


@dataclass
class DatasetConfig:
    dataset_type: Literal["vision"]
    dataset_format: Literal["json", "jsonl", "yaml", "hf_dataset"]
    dataset_path: str
    processor_config: Union[dict, ProcessorConfig]
    chat_template: Literal["qwen"] = "qwen"
