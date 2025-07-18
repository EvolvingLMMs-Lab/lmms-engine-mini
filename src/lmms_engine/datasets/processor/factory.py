from typing import Union, overload

from ...protocol import Processable
from .config import ProcessorConfig

PROCESSOR_MAPPING = {}


# A decorator class to register processors
def register_processor(processor_type: str):
    def decorator(cls):
        if processor_type in PROCESSOR_MAPPING:
            raise ValueError(f"Processor type {processor_type} is already registered.")
        PROCESSOR_MAPPING[processor_type] = cls
        return cls

    return decorator


class ProcessorFactory:
    @overload
    @staticmethod
    def create_processor(processor_config: dict) -> Processable:
        ...

    @overload
    @staticmethod
    def create_processor(processor_config: ProcessorConfig) -> Processable:
        ...

    @staticmethod
    def create_processor(processor_config: Union[ProcessorConfig, dict]) -> Processable:
        if processor_config.processor_type not in PROCESSOR_MAPPING:
            raise NotImplementedError(
                f"Processor {processor_config.processor_type} not implemented"
                f"Available processors: {list(PROCESSOR_MAPPING.keys())}"
            )
        processor_cls = PROCESSOR_MAPPING[processor_config.processor_type]
        return processor_cls(processor_config)
