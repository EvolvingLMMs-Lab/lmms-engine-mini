import math
import os
from typing import Union, overload

import torch
from accelerate import Accelerator
from accelerate.utils import MegatronLMDummyScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import DatasetConfig, DatasetFactory
from ..models import ModelConfig, ModelFactory
from .config import TrainerConfig


class AccelerateMegatronTrainer:
    def __init__(self, config: dict) -> None:
        self.train_dataset_config = DatasetConfig(**config["train_dataset_config"])
        self.model_config = ModelConfig(**config["model_config"])
        self.config = TrainerConfig(
            config["trainer_type"],
            dataset_config=self.dataset_config,
            model_config=self.model_config,
        )

    def build(self):
        self.model = self._build_model()
        self.train_dataset = self._build_train_dataset()
        self.accelerator = self._build_accelerator()

    def _build_model(self):
        model_class = ModelFactory.create_model(self.model_config.model_class)
        model = model_class.from_pretrained(self.model_config.model_name_or_path)
        return model

    def _build_train_dataset(self):
        return DatasetFactory.create_dataset(self.train_dataset_config)

    def _build_accelerator(self):
        accelerator_log_kwargs = {}
        accelerator_log_kwargs["log_with"] = self.config.report_to
        accelerator_log_kwargs["project_dir"] = self.config.output_dir
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            **accelerator_log_kwargs,
        )
        return accelerator

    def run(self):
        self.train_dataloader = self.get_train_dataloader()
        self.optimizer = self.get_optimizer()
        num_train_step = self.get_train_step()
        self.lr_scheduler = self._create_megatron_scheduler()

        (
            model_wrapped,
            optimizer_wrapped,
            train_dataloader_wrapped,
            lr_scheduler_wrapped,
        ) = self.prepare_model()
        num_train_step = self.get_train_step(train_dataloader_wrapped)

        self.progress_bar = tqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )
        self.completed_steps = 0
        self.starting_epoch = 0

        for epoch in range(self.starting_epoch, self.config.num_train_epochs):
            model_wrapped.train()
            self.training_loop(
                train_dataloader_wrapped,
                model_wrapped,
                optimizer_wrapped,
                lr_scheduler_wrapped,
            )

        self.accelerator.end_training()
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            self.config.output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )
        if self.accelerator.is_main_process:
            self.train_dataset.processor.save_pretrained(self.config.output_dir)

    def training_loop(self, train_dataloader, model, optimizer, lr_scheduler):
        total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            loss = self.training_step(batch, model, optimizer, lr_scheduler)
            if self.accelerator.sync_gradients:
                self.progress_bar.update(1)
                self.completed_steps += 1
            if isinstance(self.config.checkpointing_steps, int):
                if (
                    self.completed_steps % self.config.checkpointing_steps == 0
                    and self.accelerator.sync_gradients
                ):
                    output_dir = f"step_{self.completed_steps}"
                    if self.config.output_dir is not None:
                        output_dir = os.path.join(self.config.output_dir, output_dir)
                    self.accelerator.save_state(output_dir)
            if self.completed_steps >= self.config.max_train_steps:
                break

    def training_step(self, batch, model, optimizer, lr_scheduler):
        with self.accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        self.accelerator.log(
            {
                "train_loss": loss.item(),
                "step": self.completed_steps,
            },
            step=self.completed_steps,
        )

        return loss.detach().float()

    def _create_megatron_scheduler(self):
        lr_scheduler = MegatronLMDummyScheduler(
            optimizer=self.optimizer,
            total_num_steps=self.config.max_train_steps,
            warmup_num_steps=self.config.num_warmup_steps,
        )
        return lr_scheduler

    @property
    def total_batch_size(self):
        return self.accelerator.state.megatron_lm_plugin.global_batch_size

    def get_train_dataloader(self):
        collator = self.train_dataset.get_collator()
        train_dataloader = DataLoader(
            self.train_dataset,
            self.config.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        return train_dataloader

    def get_optimizer(self):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.config.learning_rate
        )
        return optimizer

    @overload
    def get_train_step(self) -> int:
        ...

    def get_train_step(self, train_dataloader: Union[None, DataLoader] = None) -> int:
        if train_dataloader is None:
            # Scheduler and math around the number of training steps.
            self.overrode_max_train_steps = False
            num_update_steps_per_epoch = math.ceil(
                len(self.train_dataloader) / self.config.gradient_accumulation_steps
            )
            if self.config.max_train_steps is None:
                self.config.max_train_steps = (
                    self.config.num_train_epochs * num_update_steps_per_epoch
                )
                self.overrode_max_train_steps = True
            return self.config.max_train_steps
        else:
            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(
                len(train_dataloader) / self.config.gradient_accumulation_steps
            )
            if self.overrode_max_train_steps:
                self.config.max_train_steps = (
                    self.config.num_train_epochs * num_update_steps_per_epoch
                )
            # Afterwards we recalculate our number of training epochs
            self.config.num_train_epochs = math.ceil(
                self.config.max_train_steps / num_update_steps_per_epoch
            )
            return self.config.max_train_steps

    def prepare_model(self):
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        return model, optimizer, train_dataloader, lr_scheduler

    def get_checkpointing_steps(self):
        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = self.config.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)
        return checkpointing_steps

    def init_trackers(self):
        experiment_config = self.config.__dict__
        self.accelerator.init_trackers("clm_no_trainer", experiment_config)
