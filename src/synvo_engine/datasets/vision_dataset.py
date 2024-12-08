import json
from copy import deepcopy
from typing import Dict

import torch
from datasets import Dataset as HFDataset
from datasets import Image as HFImageFeature
from datasets import Sequence, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from ..utils.train import TrainUtilities
from .collator import VisionCollator
from .config import DatasetConfig


class VisionSFTDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config

    def _build_from_config(self):
        self.processor = AutoProcessor.from_pretrained(self.config.processor_name)
        if self.config.dataset_format == "json":
            with open(self.config.dataset_path, "r") as f:
                self.data_list = json.load(f)
        elif self.config.dataset_format == "hf_dataset":
            self.data_list = load_dataset(self.config.dataset_path, split="train")
            self.data_list_no_image = deepcopy(self.data_list)
            self.data_list_no_image = self.data_list_no_image.remove_columns("image")
        else:
            raise NotImplementedError

    def build(self):
        self._build_from_config()

    def load_from_json(self, data) -> Dict[str, torch.Tensor]:
        # TODO Write a protocol for vision openai input
        images_list = []
        messages = data["messages"]
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        images = [Image.open(image) for image in images_list]
        prompt = self.processor.apply_chat_template(hf_messages, tokenize=False)
        inputs = dict(
            images=images,
            prompt=prompt,
        )
        labels = self.get_labels(hf_messages)["labels"]
        inputs["labels"] = labels
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        messages = data["messages"]
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if isinstance(data["image"], list):
            images = data["image"]
        else:
            images = [data["image"]]
        prompt = self.processor.apply_chat_template(hf_messages, tokenize=False)
        inputs = dict(
            images=images,
            prompt=prompt,
        )
        labels = self.get_labels(hf_messages)["labels"]
        inputs["labels"] = labels
        return inputs

    def __getitem__(self, index):
        if self.config.dataset_format == "json":
            data_dict = self.load_from_json(self.data_list[index])
        elif self.config.dataset_format == "hf_dataset":
            data_dict = self.load_from_hf(self.data_list[index])
        else:
            raise NotImplementedError
        return data_dict

    def __len__(self):
        return len(self.data_list)

    def get_collator(self):
        return VisionCollator(self.processor)

    def get_labels(self, hf_messages):
        if self.config.chat_template == "qwen":
            labels = TrainUtilities.get_qwen_template_labels(
                hf_messages, self.processor
            )
        return labels

    @property
    def modality_length(self):
        length = []
        if self.config.dataset_format == "json":
            for data in self.data_list:
                mm_data_num = 0
                for message in data["messages"]:
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            mm_data_num += 1
                length.append(mm_data_num)
        elif self.config.dataset_format == "hf_dataset":
            for data in self.data_list_no_image:
                mm_data_num = 0
                for message in data["messages"]:
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            mm_data_num += 1
                length.append(mm_data_num)
        else:
            raise NotImplementedError
        return length
