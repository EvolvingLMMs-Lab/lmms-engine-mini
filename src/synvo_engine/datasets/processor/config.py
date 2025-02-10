from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ProcessorConfig:
    processor_name: str
    processor_modality: Literal["vision", "audio"]
    processor_type: str
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
