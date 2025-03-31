import os
from io import BytesIO
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from decord import VideoReader, cpu
from PIL import Image, PngImagePlugin
from torchvision import io, transforms

from ..utils import DataUtilities, Logging
from .config import DatasetConfig
from .processor import ProcessorConfig, ProcessorFactory

try:
    from google.cloud.storage import Client
except:
    Logging.info("Google Cloud SDK not installed. Skipping import.")

LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


class LMMsDatasetMixin:
    def __init__(self, config: DatasetConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        if self.config.use_gcs:
            self.storage_client = Client()
            self.bucket_name = self.config.bucket_name

    def _build_from_config(self):
        raise NotImplementedError("This method should be implemented in the subclass")

    def _build_processor(self):
        processor = ProcessorFactory.create_processor(self.processor_config)
        return processor

    def build(self):
        self._build_from_config()
        self.processor = self._build_processor()
        self.processor.build()

    def load_image(self, image_path: str, data_folder=None) -> Image.Image:
        if data_folder is not None:
            image_path = os.path.join(data_folder, image_path)

        if self.config.use_gcs:
            file_obj = BytesIO()
            file_obj = DataUtilities.download_blob_to_stream(
                self.storage_client, self.bucket_name, image_path, file_obj
            )
            file_obj.seek(0)
            image = Image.open(file_obj)
        else:
            image = Image.open(image_path)
        return image

    def load_audio(self, audio_path: str, sr: int, data_folder=None) -> np.ndarray:
        if data_folder is not None:
            audio_path = os.path.join(data_folder, audio_path)
        if self.config.use_gcs:
            file_obj = BytesIO()
            file_obj = DataUtilities.download_blob_to_stream(
                self.storage_client, self.bucket_name, audio_path, file_obj
            )
            file_obj.seek(0)
            audio, orig_sr = sf.read(file_obj)
            audio = DataUtilities.resample_audio(audio, orig_sr, sr)
        else:
            audio = librosa.load(audio_path, sr=sr)[0]
        return audio

    def load_videos(
        self, video_path: str, data_folder=None, fps: int = 1
    ) -> Tuple[np.ndarray, float]:
        if data_folder is not None:
            video_path = os.path.join(data_folder, video_path)

        if self.config.use_gcs:
            file_obj = BytesIO()
            file_obj = DataUtilities.download_blob_to_stream(
                self.storage_client, self.bucket_name, video_path, file_obj
            )
            file_obj.seek(0)
            # Forcing to use decord at this time, torchvision actually also can, but I don't want to deal with it now
            return self.load_video_decord(file_obj, fps)

        if self.config.video_backend == "decord":
            return self.load_video_decord(video_path, fps)
        elif self.config.video_backend == "torchvision":
            return self.load_video_torchvision(video_path, fps)
        else:
            raise ValueError(f"Video backend {self.config.video_backend} not supported")

    def load_video_decord(
        self,
        video_path: Union[str, List[str], BytesIO],
        fps: int,
    ) -> Tuple[np.ndarray, float]:
        if isinstance(video_path, str) or isinstance(video_path, BytesIO):
            vr = VideoReader(video_path, ctx=cpu(0))
        elif isinstance(video_path, List):
            vr = VideoReader(video_path[0], ctx=cpu(0))

        total_frames, video_fps = len(vr), vr.get_avg_fps()
        if self.config.video_sampling_strategy == "fps":
            nframes = DataUtilities.smart_nframes(
                total_frames, video_fps=video_fps, fps=fps
            )
        elif self.config.video_sampling_strategy == "frame_num":
            nframes = self.config.frame_num
        else:
            raise ValueError(
                f"Invalid video sampling strategy: {self.config.video_sampling_strategy}"
            )
        uniform_sampled_frames = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        spare_frames = torch.tensor(spare_frames).permute(
            0, 3, 1, 2
        )  # Convert to TCHW format
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        return spare_frames, sample_fps  # (frames, height, width, channels)

    def load_video_torchvision(
        self,
        video_path: str,
        fps: int,
    ) -> Tuple[np.ndarray, float]:
        # Right now by default load the whole video
        video, audio, info = io.read_video(
            video_path,
            start_pts=0.0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW",
        )
        total_frames, video_fps = video.size(0), info["video_fps"]
        nframes = DataUtilities.smart_nframes(
            total_frames=total_frames, video_fps=video_fps, fps=fps
        )
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        video = video[idx]
        return video, sample_fps
