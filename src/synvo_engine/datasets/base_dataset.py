from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def load_from_hub(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_from_disk(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_from_json(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_collator(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_sampler(self, *args, **kwargs):
        pass
