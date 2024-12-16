from typing import List

from ..datasets import DatasetConfig
from ..models import ModelConfig
from ..protocol import Runnable
from ..train import TrainerConfig, TrainerFactory, TrainingArguments


class Pipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.initialize = False

    def build(self):
        if not self.initialize:
            self.components: List[Runnable] = []
            for config in self.config:
                if config["type"] == "trainer":
                    self.components.append(self._build_trainer(config["config"]))
                elif config["type"] == "pipeline":
                    self.components.append(self._build_pipeline(config["config"]))

    def run(self, **kwargs):
        if not self.initialize:
            self.build()

        for component in self.components:
            component.build()
            component.run()

    def _build_trainer(self, config: dict):
        dataset_config = config.pop("dataset_config")
        dataset_config = DatasetConfig(**dataset_config)
        model_config = config.pop("model_config")
        model_config = ModelConfig(**model_config)
        trainer_type = config.pop("trainer_type")
        trainer_args = TrainingArguments(**config)
        train_config = TrainerConfig(
            dataset_config=dataset_config,
            model_config=model_config,
            trainer_type=trainer_type,
            trainer_args=trainer_args,
        )
        trainer = TrainerFactory.create_trainer(train_config)
        return trainer

    def _build_pipeline(self, config: dict):
        pipeline = Pipeline(config)
        return pipeline
