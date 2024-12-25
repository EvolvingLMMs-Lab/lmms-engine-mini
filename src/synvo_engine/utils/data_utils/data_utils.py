import json
from typing import Dict, List, Tuple

import jsonlines
import yaml

from ..logging_utils import Logging


class DataUtilities:
    @staticmethod
    def load_json(path: str) -> List[Dict[str, List]]:
        with open(path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_jsonlines(path: str) -> List[Dict[str, List]]:
        data_list = []
        with jsonlines.open(path, "r") as f:
            for data in f:
                data_list.append(data)

        return data_list

    @staticmethod
    def load_yaml(path: str) -> Tuple[List[Dict[str, List]], List[str]]:
        data_list = []
        data_folder_list = []
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
            datasets = yaml_data.get("datasets")
            data_paths = [dataset.get("json_path") for dataset in datasets]
            data_folders = [dataset.get("data_folder") for dataset in datasets]
            data_types = [dataset.get("data_type") for dataset in datasets]
            for data_path, data_folder, data_type in zip(
                data_paths, data_folders, data_types
            ):
                Logging.info(f"Loading data from {data_path}")
                if data_type == "json":
                    data = DataUtilities.load_json(data_path)
                    data_list.extend(data)
                    data_folder_list.extend([data_folder] * len(data))
                elif data_type == "jsonl":
                    data = DataUtilities.load_jsonlines(data_path)
                    data_list.extend(data)
                    data_folder_list.extend([data_folder] * len(data))
                else:
                    raise NotImplementedError
                Logging.info(f"Dataset size: {len(data)}")
        return data_list, data_folder_list
