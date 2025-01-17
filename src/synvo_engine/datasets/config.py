from dataclasses import dataclass
from typing import Literal, Optional, Union

from .processor import ProcessorConfig


@dataclass
class DatasetConfig:
    dataset_type: Literal["vision", "vision_audio", "vision_preference"]
    dataset_format: Literal["json", "jsonl", "yaml", "hf_dataset"]
    dataset_path: str
    processor_config: Union[dict, ProcessorConfig]
    chat_template: Literal["qwen"] = "qwen"
    packing: Optional[bool] = False
    packing_strategy: Optional[str] = None
    packing_length: Optional[int] = 32000
