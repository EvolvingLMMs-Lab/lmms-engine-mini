import importlib

MODEL_REGISTRY = {
    "llava_onevision": "LlavaOnevisionForConditionalGeneration",
    "aero": "AeroForConditionalGeneration",
    "aero_omni": "AeroOmniForConditionalGeneration",
    "qwen2_5_vl_audio": "KinoQwen2_5_VLForConditionalGeneration",
}


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
                lmms_model_module = importlib.import_module(
                    f"lmms_engine.models.{model_name}"
                )
                cls = getattr(lmms_model_module, MODEL_REGISTRY[model_name])
            except ImportError as e:
                raise ValueError(
                    f"Model '{model_name}' can not be imported! \n Error : {e}"
                )
        return cls
