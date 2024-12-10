from abc import ABC, abstractmethod

import torch
from loguru import logger

from ..datasets import DatasetFactory
from ..models import ModelFactory
from .config import TrainerConfig


class BaseTrainer(ABC):
    """
    This is a base trainer wrapper to wrap all other trainer or your training logic
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.train_dataset_config = config.dataset_config
        self.model_config = config.model_config
        self.config = config

    def build(self):
        self.model = self._build_model()
        self.train_dataset = self._build_train_dataset()

    def _build_model(self):
        model_class = ModelFactory.create_model(self.model_config.model_class)
        model = model_class.from_pretrained(
            self.model_config.model_name_or_path,
            attn_implementation=self.model_config.attn_implementation,
            torch_dtype=(torch.bfloat16 if self.config.trainer_args.bf16 else None),
        )
        if self.model_config.overwrite_config:
            for key, value in self.model_config.overwrite_config.items():
                setattr(model.config, key, value)
                logger.info(f"Overwrite {key} to {value}")
        return model

    def _build_train_dataset(self):
        dataset = DatasetFactory.create_dataset(self.train_dataset_config)
        dataset.build()
        return dataset

    @abstractmethod
    def run(self, **kwargs):
        pass
