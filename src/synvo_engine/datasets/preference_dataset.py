import os
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.train import TrainUtilities
from .base_dataset import BaseDataset
from .config import DatasetConfig
from .processor import ProcessorConfig


class VisionPreferenceDataset(BaseDataset):
    def load_from_hf(self, data):
        raise NotImplementedError

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        # TODO Write a protocol for vision openai input
        images_list = []
        prompt_messages = data["prompt"]
        chosen_messages = data["chosen"]
        rejected_messages = data["rejected"]
        for message in prompt_messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])

        hf_messages_prompt = TrainUtilities.convert_open_to_hf(prompt_messages)
        hf_messages_chosen = TrainUtilities.convert_open_to_hf(chosen_messages)
        hf_messages_reject = TrainUtilities.convert_open_to_hf(rejected_messages)
        # TODO
        # Now assume images are all in prompt
        if data_folder is not None:
            images = [
                Image.open(os.path.join(data_folder, image)) for image in images_list
            ]
        else:
            images = [Image.open(image) for image in images_list]
        inputs = self.processor.process(images=images, hf_messages=hf_messages_prompt)
        # Cautions:
        # Only activate for Kino processor now
        # May need to refactor or added for llava
        chosen_inputs = self.processor.process(
            images=None, hf_messages=hf_messages_chosen, add_system_prompt=False
        )
        reject_inputs = self.processor.process(
            images=None, hf_messages=hf_messages_reject, add_system_prompt=False
        )
        inputs["prompt_input_ids"] = inputs.pop("input_ids")
        inputs.pop("labels")
        inputs["chosen_input_ids"] = chosen_inputs["input_ids"]
        inputs["rejected_input_ids"] = reject_inputs["input_ids"]
        return inputs

    def __getitem__(self, index):
        if (
            self.config.dataset_format == "json"
            or self.config.dataset_format == "jsonl"
        ):
            data_dict = self.load_from_json(self.data_list[index])
        elif self.config.dataset_format == "yaml":
            data_dict = self.load_from_json(
                self.data_list[index], self.data_folder[index]
            )
        elif self.config.dataset_format == "hf_dataset":
            data_dict = self.load_from_hf(self.data_list[index])
        else:
            raise NotImplementedError
        return data_dict

    def __len__(self):
        return super().__len__()

    def get_collator(self):
        return super().get_collator()
