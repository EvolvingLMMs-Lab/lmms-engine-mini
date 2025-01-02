from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .config import DatasetConfig
from .processor import ProcessorConfig


class VisionPreferenceDataset(BaseDataset):
    def load_from_hf(self, data):
        raise NotImplementedError

    def load_from_json(self, data, data_folder=None):
        raise NotImplementedError

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()

    def get_collator(self):
        return super().get_collator()
