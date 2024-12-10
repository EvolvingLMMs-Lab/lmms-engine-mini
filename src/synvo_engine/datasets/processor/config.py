from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ProcessorConfig:
    processor_name: str
    processor_modality: Literal["vision", "audio"]
    processor_type: str
    overwrite_config: Optional[dict] = None
