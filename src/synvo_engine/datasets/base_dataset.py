from abc import abstractmethod
from copy import deepcopy
from typing import Dict

from datasets import Sequence, load_dataset
from PIL import Image
from torch.utils.data import Dataset

from ..utils import Logging
from ..utils.data_utils import DataUtilities
from ..utils.train import TrainUtilities
from .collator import VisionCollator
from .config import DatasetConfig
from .processor import ProcessorConfig, ProcessorFactory


class BaseDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.processor_config = config.processor_config
        if isinstance(self.processor_config, dict):
            self.processor_config = ProcessorConfig(**self.processor_config)

    def _build_from_config(self):
        if self.config.dataset_format == "json":
            self.data_list = DataUtilities.load_json(self.config.dataset_path)
        elif self.config.dataset_format == "jsonl":
            self.data_list = DataUtilities.load_jsonlines(self.config.dataset_path)
        elif self.config.dataset_format == "hf_dataset":
            self.data_list = load_dataset(self.config.dataset_path, split="train")
            self.data_list_no_image = deepcopy(self.data_list)
            self.data_list_no_image = self.data_list_no_image.remove_columns("image")
        elif self.config.dataset_format == "yaml":
            self.data_list, self.data_folder = DataUtilities.load_yaml(
                self.config.dataset_path
            )
        else:
            raise NotImplementedError

    def _build_processor(self):
        processor = ProcessorFactory.create_processor(self.processor_config)
        if self.processor_config.overwrite_config:
            for key, value in self.processor_config.overwrite_config.items():
                setattr(processor, key, value)
                Logging.info(f"Overwrite processor {key} to {value}")
        return processor

    def build(self):
        self._build_from_config()
        self.processor = self._build_processor()
        self.processor.build()

    @abstractmethod
    def load_from_json(self, data, data_folder=None):
        pass

    @abstractmethod
    def load_from_hf(self, data):
        pass

    @abstractmethod
    def get_collator(self):
        pass
