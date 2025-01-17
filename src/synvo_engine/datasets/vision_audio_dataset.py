import os
from typing import Dict

import librosa
import torch
from PIL import Image

from ..utils.train import TrainUtilities
from .vision_dataset import VisionSFTDataset


class VisionAudioSFTDataset(VisionSFTDataset):
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images = []
        audios = []
        messages = data["messages"]
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images.append(
                        self.load_image(
                            content["image_url"]["url"], data_folder=data_folder
                        )
                    )
                elif content["type"] == "audio_url":
                    audios.append(
                        self.load_audio(
                            content["audio_url"]["url"],
                            sr=self.processor.sampling_rate,
                            data_folder=data_folder,
                        )
                    )

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if len(images) == 0:
            images = None
        if len(audios) == 0:
            audios = None
        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            audios=audios,
            sampling_rate=self.processor.sampling_rate,
        )
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        return super().load_from_hf(data)
