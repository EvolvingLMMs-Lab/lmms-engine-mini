from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from transformers import AutoProcessor


@dataclass
class VisionCollator:
    processor: AutoProcessor

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
        prompt = []
        images = []
        for instance in instances:
            prompt.append(instance["prompt"])
            images.append(instance["images"])

        inputs = self.processor(
            text=prompt, images=images, return_tensors="pt", do_pad=True, padding=True
        )
        labels = inputs["input_ids"].clone()
        labels[labels == self.image_token_id] = -100
        inputs["labels"] = labels

        return inputs

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
