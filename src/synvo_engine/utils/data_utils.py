import json
import math
from typing import Dict, List, Tuple

import jsonlines
import yaml

from .logging_utils import Logging

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


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

    @staticmethod
    def smart_nframes(
        total_frames: int,
        video_fps: int | float,
        fps: int,
    ) -> int:
        """calculate the number of frames for video used for model inputs.

        Args:
            ele (dict): a dict contains the configuration of video.
                support either `fps` or `nframes`:
                    - nframes: the number of frames to extract for model inputs.
                    - fps: the fps to extract frames for model inputs.
                        - min_frames: the minimum number of frames of the video, only used when fps is provided.
                        - max_frames: the maximum number of frames of the video, only used when fps is provided.
            total_frames (int): the original total number of frames of the video.
            video_fps (int | float): the original fps of the video.

        Raises:
            ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

        Returns:
            int: the number of frames for video used for model inputs.
        """
        min_frames = DataUtilities.ceil_by_factor(
            ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR
        )
        max_frames = DataUtilities.floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = DataUtilities.floor_by_factor(nframes, FRAME_FACTOR)
        if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
            raise ValueError(
                f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
            )
        return nframes

    @staticmethod
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    @staticmethod
    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    @staticmethod
    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor
