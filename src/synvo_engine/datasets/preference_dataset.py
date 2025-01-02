from torch.utils.data import Dataset

from .config import DatasetConfig
from .processor import ProcessorConfig


class TextPreferenceDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.processor_config = config.processor_config
        if isinstance(self.processor_config, dict):
            self.processor_config = ProcessorConfig(**self.processor_config)
