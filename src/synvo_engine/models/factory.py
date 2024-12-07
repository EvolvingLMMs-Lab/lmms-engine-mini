import importlib

from .llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionForConditionalGeneration,
)

MODEL_REGISTRY = {"llava_onevision": LlavaOnevisionForConditionalGeneration}


class ModelFactory:
    @staticmethod
    def create_model(model_name, **kwargs):
        if model_name not in MODEL_REGISTRY:
            try:
                transformers_module = importlib.import_module("transformers")
                cls = getattr(transformers_module, model_name)
                MODEL_REGISTRY[model_name] = cls
            except:
                raise ValueError(f"Model '{model_name}' not found!")
        return MODEL_REGISTRY[model_name]
