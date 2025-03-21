import collections
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from ...protocol import Processable
from ...utils.train_utils import TrainUtilities


@dataclass
class PreferenceCollator:
    processor: Processable

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.processor.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        prompt_input_ids = inputs.pop("prompt_input_ids")
        prompt_input_ids = self.pad_sequence(
            prompt_input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        prompt_attention_mask = prompt_input_ids.ne(
            self.processor.tokenizer.pad_token_id
        )

        chosen_input_ids = inputs.pop("chosen_input_ids")
        chosen_input_ids = self.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        chosen_attention_mask = chosen_input_ids.ne(
            self.processor.tokenizer.pad_token_id
        )

        rejected_input_ids = inputs.pop("rejected_input_ids")
        rejected_input_ids = self.pad_sequence(
            rejected_input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        rejected_attention_mask = rejected_input_ids.ne(
            self.processor.tokenizer.pad_token_id
        )

        batched_inputs = {}
        for key, values in inputs.items():
            batched_inputs[key] = torch.concatenate(values, dim=0)

        batched_inputs["prompt_input_ids"] = prompt_input_ids
        batched_inputs["prompt_attention_mask"] = prompt_attention_mask
        batched_inputs["chosen_input_ids"] = chosen_input_ids
        batched_inputs["chosen_attention_mask"] = chosen_attention_mask
        batched_inputs["rejected_input_ids"] = rejected_input_ids
        batched_inputs["rejected_attention_mask"] = rejected_attention_mask

        return batched_inputs

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
