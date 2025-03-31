import os
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from datasets import Sequence, load_dataset
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset

from ..utils import Logging
from ..utils.data_utils import DataUtilities
from .config import DatasetConfig
from .datasetmixin import LMMsDatasetMixin
from .processor import ProcessorConfig, ProcessorFactory

LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


class BaseDataset(Dataset, LMMsDatasetMixin):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
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
            if self.config.packing_strategy is None:
                raise ValueError("Packing strategy is not specified.")
            packing_length = self.config.packing_length
            if self.config.packing_strategy == "first_fit":
                self.packing_index = self._pack_by_first_fit(
                    self.data_lengths, packing_length
                )
            elif "window" in self.config.packing_strategy:
                window_size = int(self.config.packing_strategy.split("_")[1])
                self.packing_index = self._pack_by_window(
                    self.data_lengths, packing_length, window_size
                )
            else:
                raise NotImplementedError
            Logging.info(
                f"Before packing : {len(self.data_list)}, After packing : {len(self.packing_index)}"
            )

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
                        if "precomputed_tokens" in cont:
                            cur_len += cont["precomputed_tokens"]
                        else:
                            cur_len += 2000
                    elif cont["type"] == "audio_url":
                        if "precomputed_tokens" in cont:
                            cur_len += cont["precomputed_tokens"]
                        else:
                            cur_len += 750
                    elif cont["type"] == "video_url":
                        if "precomputed_tokens" in cont:
                            cur_len += cont["precomputed_tokens"]
                        else:
                            cur_len += 5000
                    elif cont["type"] == "text":
                        cur_len += len(cont["text"].split()) * 1.5
                    else:
                        raise TypeError(
                            f"Encountered invalid content type {cont['type']}"
                        )
            lengths.append(cur_len)
        return lengths

    def _pack_by_first_fit(self, lengths: List[int], packing_length: int):
        max_length = packing_length
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

    def _pack_by_window(
        self,
        lengths: List[int],
        packing_length: int,
        window_size: int = 100,
        control_threshold: float = 1,
        max_size: int = -1,
    ):
        max_length = packing_length
        Logging.info(f"Packing inputs...pack length:{max_length}")

        result = []
        current_concatenated_length = 0
        current_list = []
        i = 0
        cur_window = {}

        next_window = {}
        for k in range(window_size):
            next_window[f"{k}"] = lengths[k]
        while i < len(lengths):
            cur_window = next_window
            next_window = {}
            for j in cur_window.keys():
                cur_length = cur_window[j]
                if (
                    cur_length + current_concatenated_length
                ) * control_threshold <= max_length and (
                    max_size == -1 or len(current_list) < max_size
                ):
                    current_concatenated_length += cur_length
                    current_list.append(int(j))
                else:
                    next_window[j] = cur_window[j]

            if current_list == []:
                if i != len(lengths) - 1:
                    current_list.append(int(next(iter(next_window))))
                    next_window.pop(next(iter(next_window)))
                    cur_window.pop(next(iter(next_window)))
                else:
                    i += 1
                    continue

            for k in range(min(len(current_list), len(lengths) - i - 1)):
                if k + i + window_size < len(lengths):
                    index = k + i + window_size
                    next_window[f"{index}"] = lengths[index]
            i += min(len(current_list), len(lengths) - i)

            result.append(current_list)

            current_concatenated_length = 0
            current_list = []

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
