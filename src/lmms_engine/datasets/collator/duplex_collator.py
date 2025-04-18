import collections
from typing import Dict, Sequence

import numpy as np
import torch

from .vision_collator import VisionCollator


class DuplexCollator(VisionCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")
        audio_input_ids = inputs.pop("audio_input_ids")
        codec_labels = inputs.pop("codec_labels")
        split_sizes = [len(input_ids), len(audio_input_ids)]
        # First pad input ids by padding tokens
        padded_input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        padded_input_ids = [p for p in padded_input_ids]
        merged_input_ids = padded_input_ids + audio_input_ids
        merged_labels = labels + codec_labels
        # Then pad the text and audio together by audio pad token id
        merged_input_ids = self.pad_sequence(
            merged_input_ids,
            batch_first=True,
            padding_value=self.processor.audio_pad_token_id,
        )
        merged_labels = self.pad_sequence(
            merged_labels,
            batch_first=True,
            padding_value=-100,
        )
        input_ids, audio_input_ids = merged_input_ids.split(split_sizes)
        labels, codec_labels = merged_labels.split(split_sizes)
        # Assume audio is always longer than text
        attention_mask = input_ids.ne(self.processor.audio_pad_token_id)
        batched_inputs = {}
        for key, values in inputs.items():
            batched_inputs[key] = torch.concatenate(values, dim=0)
        batched_inputs["input_ids"] = input_ids
        batched_inputs["labels"] = labels
        batched_inputs["attention_mask"] = attention_mask
        batched_inputs["audio_input_ids"] = audio_input_ids
        batched_inputs["codec_labels"] = codec_labels
        # batched_inputs["position_ids"] = position_ids

        return batched_inputs
