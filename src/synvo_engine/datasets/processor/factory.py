from typing import Union, overload

from ...protocol import Processable
from .config import ProcessorConfig
from .vision import LLaVADataProcessor, Qwen2VLDataProcessor


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
        if isinstance(processor_config, dict):
            processor_config = ProcessorConfig(**processor_config)

        if processor_config.processor_modality == "vision":
            return ProcessorFactory.create_vision_processor(processor_config)
        elif processor_config.processor_modality == "audio":
            return ProcessorFactory.create_audio_processor(processor_config)
        else:
            raise NotImplementedError(
                f"Processor type {processor_config.processor_type} not implemented"
            )
        pass

    @staticmethod
    def create_vision_processor(processor_config: ProcessorConfig) -> Processable:
        if processor_config.processor_type == "qwen2_vl":
            return Qwen2VLDataProcessor(processor_config)
        elif processor_config.processor_type == "llava":
            return LLaVADataProcessor(processor_config)
        else:
            raise NotImplementedError(
                f"Processor {processor_config.processor_name} not implemented"
            )

    @staticmethod
    def create_audio_processor(processor_config: ProcessorConfig) -> Processable:
        pass
