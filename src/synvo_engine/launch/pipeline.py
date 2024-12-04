from typing import List

from ..protocol import Runnable
from ..train import TrainerConfig, TrainerFactory


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
            component.run()

    def _build_trainer(self, config: dict):
        train_config = TrainerConfig(**config)
        trainer = TrainerFactory.create_trainer(train_config)
        return trainer

    def _build_pipeline(self, config: dict):
        pipeline = Pipeline(config)
        return pipeline
