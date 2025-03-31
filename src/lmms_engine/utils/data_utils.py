import json
import math
from io import BytesIO
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Literal, Tuple

import jsonlines
import numpy as np
import yaml
from librosa import resample
from tqdm import tqdm

from .logging_utils import Logging
from .train_utils import TrainUtilities

try:
    from google.cloud.storage import Client
except:
    Logging.info("Google Cloud SDK not installed. Skipping import.")

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
    def maybe_load_json_or_jsonlines(
        path: str, data_type: Literal["json", "jsonl"]
    ) -> List[Dict[str, List]]:
        if data_type == "json":
            return DataUtilities.load_json(path)
        elif data_type == "jsonl":
            return DataUtilities.load_jsonlines(path)
        else:
            raise NotImplementedError

    @staticmethod
    def wrap_func(args):
        path, data_type = args
        return DataUtilities.maybe_load_json_or_jsonlines(path, data_type)

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
            with Pool(cpu_count()) as p:
                nested_data_list = tqdm(
                    list(p.imap(DataUtilities.wrap_func, zip(data_paths, data_types))),
                    total=len(data_paths),
                    desc="Loading data ...",
                    disable=not TrainUtilities.is_rank_zero(),
                )

            for data, data_folder, data_path in zip(
                nested_data_list, data_folders, data_paths
            ):
                Logging.info(f"Data : {data_path}")
                data_list.extend(data)
                data_folder_list.extend([data_folder] * len(data))
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
        min_frames = DataUtilities.ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
        max_frames = DataUtilities.floor_by_factor(
            min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR
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

    @staticmethod
    def download_blob_to_stream(
        storage_client: Client,
        bucket_name: str,
        source_blob_name: str,
        file_obj: BytesIO,
    ) -> BytesIO:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_file(file_obj)
        return file_obj

    @staticmethod
    def resample_audio(
        audio_array: np.ndarray, original_sr: int, target_sr: int
    ) -> np.ndarray:
        audio_resample_array = resample(
            audio_array, orig_sr=original_sr, target_sr=target_sr
        )
        return audio_resample_array
