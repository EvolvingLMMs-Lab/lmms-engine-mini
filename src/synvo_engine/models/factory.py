import importlib

MODEL_REGISTRY = {"llava_onevision": "LlavaOnevisionForConditionalGeneration"}


class ModelFactory:
    @staticmethod
    def create_model(model_name, **kwargs):
        if model_name not in MODEL_REGISTRY:
            try:
                transformers_module = importlib.import_module("transformers")
                cls = getattr(transformers_module, model_name)
            except:
                raise ValueError(f"Model '{model_name}' not found!")
        else:
            try:
                synvo_model_module = importlib.import_module(
                    f"synvo_engine.models.{model_name}"
                )
                cls = getattr(synvo_model_module, MODEL_REGISTRY[model_name])
            except ImportError as e:
                raise ValueError(
                    f"Model '{model_name}' can not be imported! \n Error : {e}"
                )
        return cls
