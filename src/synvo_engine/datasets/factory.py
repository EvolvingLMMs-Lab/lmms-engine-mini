from .config import DatasetConfig
from .preference_dataset import VisionPreferenceDataset
from .vision_audio_dataset import VisionAudioSFTDataset
from .vision_dataset import VisionSFTDataset


class DatasetFactory:
    @staticmethod
    def create_dataset(config: DatasetConfig, **kwargs):
        if config.dataset_type == "vision":
            return VisionSFTDataset(config, **kwargs)
        elif config.dataset_type == "vision_audio":
            return VisionAudioSFTDataset(config, **kwargs)
        elif config.dataset_type == "vision_preference":
            return VisionPreferenceDataset(config, **kwargs)
        else:
            raise NotImplementedError(
                f"Dataset type '{config.dataset_type}' not found!"
            )
