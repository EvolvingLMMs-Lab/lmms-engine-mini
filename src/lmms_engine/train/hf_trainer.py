import os
import pathlib

import torch
from transformers import Trainer

from ..utils import Logging
from ..utils.train_utils import TrainUtilities
from .base_trainer import BaseTrainer
from .config import TrainerConfig
from .custom import LLaVADPOTrainer, LLaVAGRPOTrainer, LLaVATrainer
from .reward_func import REWARD_FUNC_REGISTRY


class Hf_Trainer(BaseTrainer):
    def __init__(self, config: TrainerConfig) -> None:
        self.set_random_seed()
        super().__init__(config)

    def build(self):
        super().build()
        if self.config.trainer_args_type == "dpo":
            Logging.info("Init Reference Model for DPO ...")
            self.ref_model = self._build_ref_model()
        elif self.config.trainer_args_type == "grpo":
            Logging.info("Init Reference Model for GRPO...")
            self.ref_model = self._build_ref_model()
        self.trainer = self._build_trainer()

    def _build_trainer(self):
        if self.config.trainer_args_type == "sft":
            trainer = LLaVATrainer(
                model=self.model,
                args=self.config.trainer_args,
                data_collator=self.train_dataset.get_collator(),
                train_dataset=self.train_dataset,
                processing_class=self.train_dataset.processor,
            )
        elif self.config.trainer_args_type == "dpo":
            trainer = LLaVADPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=self.config.trainer_args,
                data_collator=self.train_dataset.get_collator(),
                train_dataset=self.train_dataset,
                processing_class=self.train_dataset.processor,
            )
        elif self.config.trainer_args_type == "grpo":
            reward_funcs = [
                REWARD_FUNC_REGISTRY[func]
                for func in self.config.trainer_args.reward_funcs
            ]
            trainer = LLaVAGRPOTrainer(
                model=self.model,
                args=self.config.trainer_args,
                ref_model=self.ref_model,
                train_dataset=self.train_dataset,
                reward_funcs=reward_funcs,
                data_collator=self.train_dataset.get_collator(),
            )
        else:
            raise NotImplementedError(
                f"Unknown trainer args type: {self.config.trainer_args_type}"
            )
        return trainer

    def run(self, **kwargs):
        self.save_config()
        if self.config.trainer_args.freeze_modules:
            for modules in self.config.trainer_args.freeze_modules:
                cls = getattr(self.model, modules, None)
                if cls is not None:
                    for param in cls.parameters():
                        param.requires_grad = False

        if self.config.trainer_args.use_lora:
            self.freeze_not_lora_params()
        if list(pathlib.Path(self.config.trainer_args.output_dir).glob("checkpoint-*")):
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()
        self.trainer.save_state()
        self.safe_save_model_for_hf_trainer(
            self.trainer, self.config.trainer_args.output_dir
        )

    def safe_save_model_for_hf_trainer(self, trainer: Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        trainer.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        check_only_save_mm_adapter = self.config.trainer_args.only_save_mm_adapter
        Logging.info(f"Only save projectors: {check_only_save_mm_adapter}")

        if check_only_save_mm_adapter:
            # Only save Adapter
            keys_to_match = ["multi_modal_projector", "audio_modal_projector"]

            weight_to_save = TrainUtilities.get_mm_adapter_state_maybe_zero_3(
                trainer.model.named_parameters(), keys_to_match
            )
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith("checkpoint-"):
                    mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(
                        weight_to_save,
                        os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                    )
                else:
                    torch.save(
                        weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                    )
            return
        if trainer.deepspeed:
            trainer.save_model(output_dir)
            return
        if self.config.trainer_args.fsdp2:
            # For fsdp we merge the shards into a single checkpoint after the training is done
            if trainer.processing_class is not None:
                trainer.processing_class.save_pretrained(output_dir)
            return

        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    def _build_ref_model(self):
        ref_model = self._build_model()
        parameter_names = [n for n, _ in ref_model.named_parameters()]
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        ref_model.eval()
        return ref_model
