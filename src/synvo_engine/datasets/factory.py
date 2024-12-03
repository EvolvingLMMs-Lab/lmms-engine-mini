from .config import DatasetConfig
from .vision_dataset import VisionSFTDataset


class DatasetFactory:
    @staticmethod
    def create_dataset(config: DatasetConfig, **kwargs):
        if config.dataset_type == "vision":
            return VisionSFTDataset(config, **kwargs)
