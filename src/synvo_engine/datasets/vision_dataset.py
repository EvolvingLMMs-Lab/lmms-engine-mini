import json

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from ..utils.train import TrainUtilities
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

    def build(self):
        self._build_from_config()

    def load_from_json(self, data):
        # TODO Write a protocol for vision openai input
        images_list = []
        messages = data["messages"]
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        prompt = self.processor.apply_chat_template(hf_messages)
        images = [Image.open(image) for image in images_list]
        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        return inputs

    def __getitem__(self, index):
        if self.config.dataset_format == "json":
            data_dict = self.load_from_json(self.data_list[index])
        return data_dict

    def __len__(self):
        return len(self.data_list)
