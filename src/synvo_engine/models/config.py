from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ModelConfig:
    model_name_or_path: str
    model_class: str
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa"]] = "sdpa"
