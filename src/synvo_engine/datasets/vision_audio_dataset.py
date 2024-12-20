import os
from typing import Dict

import librosa
import torch
from PIL import Image

from ..utils.train import TrainUtilities
from .vision_dataset import VisionSFTDataset


class VisionAudioSFTDataset(VisionSFTDataset):
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images_list = []
        audios_list = []
        messages = data["messages"]
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "audio_url":
                    audios_list.append(content["audio_url"]["url"])

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [
                Image.open(os.path.join(data_folder, image)) for image in images_list
            ]
            audios = [
                librosa.load(
                    os.path.join(data_folder, audio), sr=self.processor.sampling_rate
                )[0]
                for audio in audios_list
            ]
        else:
            images = [Image.open(image) for image in images_list]
            audios = [
                librosa.load(audio, sr=self.processor.sampling_rate)[0]
                for audio in audios_list
            ]
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

    @property
    def modality_length(self):
        length = []
        if (
            self.config.dataset_format == "json"
            or self.config.dataset_format == "jsonl"
            or self.config.dataset_format == "yaml"
        ):
            for data in self.data_list:
                mm_data_num = 0
                for message in data["messages"]:
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            mm_data_num += 1
                        elif content["type"] == "audio_url":
                            mm_data_num += 1
                length.append(mm_data_num)
        elif self.config.dataset_format == "hf_dataset":
            for data in self.data_list_no_image:
                mm_data_num = 0
                for message in data["messages"]:
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            mm_data_num += 1
                        elif content["type"] == "audio_url":
                            mm_data_num += 1
                length.append(mm_data_num)
        else:
            raise NotImplementedError
        return length
