from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name_or_path: str
    model_class: str
