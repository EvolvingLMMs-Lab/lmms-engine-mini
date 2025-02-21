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
from ..utils.train_utils import TrainUtilities
from .base_dataset import BaseDataset
from .collator import VisionCollator
from .config import DatasetConfig
from .processor import ProcessorConfig, ProcessorFactory


class VisionSFTDataset(BaseDataset):
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

    def get_collator(self):
        return VisionCollator(self.processor)
