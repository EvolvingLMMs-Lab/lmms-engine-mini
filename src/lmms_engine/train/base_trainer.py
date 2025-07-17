import json
import os
import random
import shutil
from abc import ABC, abstractmethod

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from ..datasets import DatasetFactory
from ..models import ModelFactory
from ..models.kernels import CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN
from ..models.kernels import (
    _apply_liger_kernel_to_instance as _apply_liger_kernel_to_custom_instance,
)
from ..utils import Logging
from ..utils.train_utils import TrainUtilities
from .config import TrainerConfig


class BaseTrainer(ABC):
    """
    This is a base trainer wrapper to wrap all other trainer or your training logic
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.train_dataset_config = config.dataset_config
        self.model_config = config.model_config
        self.config = config
        self.lora_configs = config.lora_configs

    def build(self):
        self.model = self._build_model()
        self.train_dataset = self._build_train_dataset()
        if self.model_config.pretrain_mm_mlp_adapter is not None:
            self._load_mm_projector()
        if self.config.trainer_args.use_liger_kernel:
            self._apply_liger_kernel()
            # Set to False as we already apply the liger kernel by ourselves
            self.config.trainer_args.use_liger_kernel = False

    def _build_model(self):
        model_class = ModelFactory.create_model(self.model_config.model_class)
        model = model_class.from_pretrained(
            self.model_config.model_name_or_path,
            attn_implementation=self.model_config.attn_implementation,
            torch_dtype=(torch.bfloat16 if self.config.trainer_args.bf16 else None),
        )
        if self.config.trainer_args.use_lora:
            model = self._build_lora_on_model(model)
        if self.model_config.overwrite_config:
            for key, value in self.model_config.overwrite_config.items():
                setattr(model.config, key, value)
                Logging.info(f"Overwrite {key} to {value}")
        return model

    def _apply_liger_kernel(self):
        kwargs = {"use_rmpad": self.config.trainer_args.use_rmpad}
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            from liger_kernel.transformers.monkey_patch import (
                MODEL_TYPE_TO_APPLY_LIGER_FN,
            )
        except ImportError as e:
            Logging.error(
                "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                "Please install it with `pip install liger-kernel`"
            )

        model_type = getattr(self.model, "config", None) and getattr(
            self.model.config, "model_type", None
        )
        if model_type in CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN:
            Logging.info(f"Try to apply liger kernel on the model {model_type}")
            _apply_liger_kernel_to_custom_instance(self.model, **kwargs)
        # If the model itself is already in liger kernel,
        # we should not apply the liger kernel again
        elif model_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
            Logging.info(f"Try to apply liger kernel on the model {model_type}")
            _apply_liger_kernel_to_instance(self.model)
        else:
            # If not, we probe whether lm can apply
            Logging.info(
                f"Not found model class, Try to apply liger kernel on the language model of the model {model_type}"
            )
            try:
                model_type = getattr(
                    self.model.language_model, "config", None
                ) and getattr(self.model.language_model.config, "model_type", None)
                _apply_liger_kernel_to_instance(self.model.language_model)
                if model_type and model_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
                    Logging.info(
                        f"Successfully apply liger kernels to model type {model_type}"
                    )
                else:
                    Logging.info(
                        f"Cannot find model type {model_type} in MODEL_TYPE_TO_APPLY_LIGER_FN, skip applying liger kernels"
                    )
            except Exception as e:
                Logging.error(
                    f"Try to apply liger kernel on the language model of the model, but failed with exceptions : \n {e}"
                )

    def _load_mm_projector(self):
        pretrain_mm_mlp_adapter = self.config.model_config.pretrain_mm_mlp_adapter
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

        def get_w(weights, keyword):
            return {
                k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k
            }

        deepspeed3_enabled = hasattr(
            [p for p in self.model.multi_modal_projector.parameters()][0], "ds_id"
        )

        TrainUtilities.load_zero_partitions(
            self.model.multi_modal_projector,
            get_w(mm_projector_weights, "multi_modal_projector"),
            deepspeed3_enabled,
            pretrain_mm_mlp_adapter,
        )
        TrainUtilities.load_zero_partitions(
            self.model.audio_modal_projector,
            get_w(mm_projector_weights, "audio_modal_projector"),
            deepspeed3_enabled,
            pretrain_mm_mlp_adapter,
        )

        Logging.info(
            f"Loaded multi_modal_projector,audio_modal_projector weights from {pretrain_mm_mlp_adapter}."
        )

    def _build_train_dataset(self):
        dataset = DatasetFactory.create_dataset(self.train_dataset_config)
        dataset.build()
        return dataset

    @abstractmethod
    def run(self, **kwargs):
        pass

    def freeze_not_lora_params(self):
        from peft.tuners.lora.layer import LoraLayer

        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                for n, p in module.named_parameters():
                    if "base_layer" in n:
                        p.requires_grad = False
                    elif "lora" in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = True

    def _build_lora_on_model(self, model):
        assert self.config.trainer_args.use_lora, "You should set use_lora to True"
        if self.lora_configs is None:
            return model
        for lora_config in self.lora_configs:
            adapter_name = lora_config.adapter_name
            peft_model = get_peft_model(
                model,
                peft_config=lora_config,
                adapter_name=adapter_name,
            )
            setattr(model.config, f"{adapter_name}_lora", lora_config.to_dict())
            Logging.info(f"Set {adapter_name}_lora to {lora_config.to_dict()}")
            trainable_params, all_param = peft_model.get_nb_trainable_parameters()
            Logging.info(
                f"trainable params: {trainable_params:,d} || "
                f"all params: {all_param:,d} || "
                f"trainable%: {100 * trainable_params / all_param:.4f}"
            )
            Logging.info(f"Model structure : {model}")

        return model

    def save_config(self):
        output_dir = self.config.trainer_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/training_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        if self.config.dataset_config.dataset_format == "yaml":
            # Copy the yaml to output dir
            yaml_path = self.config.dataset_config.dataset_path
            shutil.copy(yaml_path, f"{output_dir}/dataset.yaml")

    def set_random_seed(self, random_seed: int = 42):
        # Setting random seed for all
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        Logging.info(f"Set random seed to {random_seed}")
        return random_seed
