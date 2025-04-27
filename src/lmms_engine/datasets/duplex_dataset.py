from .collator import DuplexCollator
from .vision_audio_dataset import VisionAudioSFTDataset


class DuplexDataset(VisionAudioSFTDataset):
    def get_collator(self):
        return DuplexCollator(self.processor)
