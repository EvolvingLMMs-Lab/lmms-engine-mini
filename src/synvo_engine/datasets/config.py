from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetConfig:
    dataset_type: Literal["vision"]
    dataset_format: Literal["json", "hf_dataset"]
    dataset_path: str
    processor_name: str
