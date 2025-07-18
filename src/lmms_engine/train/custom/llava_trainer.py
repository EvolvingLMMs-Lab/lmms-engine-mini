import importlib.metadata
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from transformers import Trainer
from transformers.trainer import logger
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
from transformers.trainer_utils import has_length
from transformers.utils import (
    is_datasets_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
)

from ...utils.train_utils import TrainUtilities
from .muon import Muon


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


TRAINER_STATE_NAME = "trainer_state.json"


class LLaVATrainer(Trainer):
    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(
                self.train_dataset, datasets.Dataset
            ):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            # Hard code here because we use our own processing class
            model_input_name = None
            # model_input_name = (
            # self.processing_class.model_input_names[0]
            # if self.processing_class is not None
            # else None
            # )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=self.train_dataset.modality_length,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # Using techniques from https://github.com/KellerJordan/Muon
            # Seems not boosting the efficiency
            if self.args.use_muon:
                muon_params = []
                adamw_params = []
                for grouped_parameters in optimizer_grouped_parameters:
                    for param in grouped_parameters["params"]:
                        if param.ndim >= 2:
                            muon_params.append(param)
                        else:
                            adamw_params.append(param)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            if self.args.use_muon:
                self.optimizer = Muon(
                    muon_params,
                    lr=0.02,
                    momentum=0.95,
                    adamw_params=adamw_params,
                    adamw_lr=optimizer_kwargs["lr"],
                    adamw_betas=optimizer_kwargs["betas"],
                    adamw_eps=optimizer_kwargs["eps"],
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs
                )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def get_memory(self):
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        mem = torch.cuda.memory_allocated()
        return peak_mem / 1e9, mem / 1e9

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "only_save_mm_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["multi_modal_projector", "audio_modal_projector"]

            weight_to_save = TrainUtilities.get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.state.global_step == 0 or getattr(self, "cur_time", None) is None:
            self.cur_time = time.perf_counter()
            self.mfu = 0.0
            self.flops = 0
        if (
            self.state.global_step % 10 == 0
            and self.flops > 0  # No flops logging for this model
        ):
            prev_time = self.cur_time
            self.cur_time = time.perf_counter()
            device = self.args.local_rank
            flops_tensor = torch.tensor(self.flops, device=device)
            torch.distributed.all_reduce(
                flops_tensor, op=torch.distributed.ReduceOp.SUM
            )
            self.mfu = (
                flops_tensor.item()
                / (self.cur_time - prev_time)
                / self.args.world_size
                / TrainUtilities.get_device_flops("B")
            )
            self.log({"mfu": round(self.mfu, 2)})
            self.flops = 0
        loss, outputs = super().compute_loss(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch,
            return_outputs=True,
        )
        self.flops += outputs.get("flops", 0)
        return (loss, outputs) if return_outputs else loss
