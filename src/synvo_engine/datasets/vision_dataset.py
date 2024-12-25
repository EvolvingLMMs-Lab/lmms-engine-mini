import json
import os
from copy import deepcopy
from typing import Dict

import jsonlines
import torch
import yaml
from datasets import Dataset as HFDataset
from datasets import Image as HFImageFeature
from datasets import Sequence, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from ..utils import Logging
from ..utils.data_utils import DataUtilities
from ..utils.train import TrainUtilities
from .collator import VisionCollator
from .config import DatasetConfig
from .processor import ProcessorConfig, ProcessorFactory


class VisionSFTDataset(Dataset):
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

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        # TODO Write a protocol for vision openai input
        images_list = []
        messages = data["messages"]
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [
                Image.open(os.path.join(data_folder, image)) for image in images_list
            ]
        else:
            images = [Image.open(image) for image in images_list]
        inputs = self.processor.process(images=images, hf_messages=hf_messages)
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        messages = data["messages"]
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if isinstance(data["image"], list):
            images = data["image"]
        else:
            images = [data["image"]]
        inputs = self.processor.process(images=images, hf_messages=hf_messages)
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
        return len(self.data_list)

    def get_collator(self):
        return VisionCollator(self.processor)

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
