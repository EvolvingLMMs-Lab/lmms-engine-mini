import importlib

MODEL_REGISTRY = {}


def register_model(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert (
                name not in MODEL_REGISTRY
            ), f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead."

            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


class ModelFactory:
    @staticmethod
    def create_model(model_name, **kwargs):
        if model_name not in MODEL_REGISTRY:
            try:
                cls = importlib.import_module("transformers", model_name)
                MODEL_REGISTRY[model_name] = cls
            except:
                raise ValueError(f"Model '{model_name}' not found!")
        return MODEL_REGISTRY[model_name](**kwargs)
