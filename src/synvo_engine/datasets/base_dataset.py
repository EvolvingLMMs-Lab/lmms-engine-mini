from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List

from datasets import Sequence, load_dataset
from PIL import Image
from torch.utils.data import Dataset

from ..utils import Logging
from ..utils.data_utils import DataUtilities
from ..utils.train import TrainUtilities
from .collator import VisionCollator
from .config import DatasetConfig
from .processor import ProcessorConfig, ProcessorFactory


class BaseDataset(Dataset):
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

        self.data_lengths = (
            self._estimate_data_tokens(self.data_list)
            if self.config.dataset_format != "hf_dataset"
            else self.data_list_no_image
        )
        if self.config.packing:
            self.packing_index = self._pack_by_first_fit(self.data_lengths)
            Logging.info(
                f"Before packing : {len(self.data_list)}, After packing : {len(self.packing_index)}"
            )

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

    def _estimate_data_tokens(self, data_list):
        lengths = []
        for data in data_list:
            if "chosen" in data or "rejected" in data:
                raise ValueError("Preference learning is not supported for now.")
            messages = data["messages"]
            cur_len = 0
            for message in messages:
                content = message["content"]
                for cont in content:
                    if cont["type"] == "image_url":
                        cur_len += 2000
                    elif cont["type"] == "audio_url":
                        cur_len += 750
                    elif cont["type"] == "text":
                        cur_len += len(cont["text"].split()) * 1.25
                    else:
                        raise TypeError(
                            f"Encountered invalid content type {cont['type']}"
                        )
            lengths.append(cur_len)
        return lengths

    def _pack_by_first_fit(self, lengths: List[int]):
        max_length = max(lengths)
        Logging.info(f"Packing inputs...pack max length: {max_length}")

        result = []
        current_concatenated_length = 0
        current_list = []
        for i in range(len(lengths)):
            cur_length = lengths[i]
            if cur_length + current_concatenated_length <= max_length:
                current_concatenated_length += cur_length
                current_list.append(i)
            else:  # current_list is done, create a new one
                if len(current_list) > 0:
                    result.append(current_list)
                current_list = [i]
                current_concatenated_length = cur_length

        if len(current_list) > 0:
            result.append(current_list)

        # assert to make sure no indices were missing
        assert sum([len(indices) for indices in result]) == len(lengths)
        return result

    @property
    def modality_length(self):
        # If it is packing, we add by packing index
        if self.config.packing:
            lengths = []
            for index_group in self.packing_index:
                cur_length = 0
                for index in index_group:
                    cur_length += self.data_lengths[index]
                lengths.append(cur_length)
            return lengths
        # Otherwise, the original data lengths is sufficient
        return self.data_lengths

    def __len__(self):
        if self.config.packing:
            return len(self.packing_index)
        return len(self.data_list)

    def __getitem__(self, index):
        if self.config.packing:
            index_group = self.packing_index[index]
            data_dict_list = self.load_from_packing(index_group)
            return data_dict_list

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

    def load_from_packing(self, index_group):
        if (
            self.config.dataset_format == "json"
            or self.config.dataset_format == "jsonl"
        ):
            data_dict_list = [
                self.load_from_json(self.data_list[index]) for index in index_group
            ]
        elif self.config.dataset_format == "yaml":
            data_dict_list = [
                self.load_from_json(self.data_list[index], self.data_folder[index])
                for index in index_group
            ]
        elif self.config.dataset_format == "hf_dataset":
            data_dict_list = [
                self.load_from_hf(self.data_list[index]) for index in index_group
            ]
        else:
            raise NotImplementedError
        return data_dict_list

    @abstractmethod
    def load_from_json(self, data, data_folder=None):
        pass

    @abstractmethod
    def load_from_hf(self, data):
        pass

    @abstractmethod
    def get_collator(self):
        pass
