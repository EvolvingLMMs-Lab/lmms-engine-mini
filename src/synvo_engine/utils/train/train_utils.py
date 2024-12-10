from typing import Iterable, List, Union

import torch
from transformers import AutoProcessor


class TrainUtilities:
    @staticmethod
    def prepare_model():
        pass

    @staticmethod
    def convert_open_to_hf(messages):
        hf_messages = []
        for message in messages:
            new_message = {"role": message["role"], "content": []}
            for content in message["content"]:
                if content["type"] == "image_url":
                    new_message["content"].append({"type": "image"})
                else:
                    new_message["content"].append(
                        {"type": "text", "text": content["text"]}
                    )
            hf_messages.append(new_message)

        return hf_messages

    @staticmethod
    def sanity_check_labels(
        processor: AutoProcessor, input_ids: torch.Tensor, labels: torch.Tensor
    ):
        print(" ======== Inputs ========")
        for o in processor.batch_decode(input_ids):
            print(o)
            break
        print(" ======== Labels ========")
        labels[labels == -100] = 0
        for o in processor.batch_decode(labels):
            print(o)
            break

    @staticmethod
    def get_device_flops(unit="T"):
        def unit_convert(number, level):
            units = ["B", "K", "M", "G", "T", "P"]
            if number <= 0:
                return number
            ptr = 0
            while ptr < len(units) and units[ptr] != level:
                number /= 1000
                ptr += 1
            return number

        device_name = torch.cuda.get_device_name()
        flops = float("inf")  # INF flops for unkown gpu type
        if "H100" in device_name or "H800" in device_name:
            flops = 989e12
        elif "A100" in device_name or "A800" in device_name:
            flops = 312e12
        elif "L40" in device_name:
            flops = 181.05e12
        elif "910B" in device_name:
            flops = 354e12
        flops_unit = unit_convert(flops, unit)
        return flops_unit

    @staticmethod
    def get_qwen_template_labels(hf_messages, processor: AutoProcessor):
        image_token_index = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )
        special_tokens = processor.tokenizer.additional_special_tokens
        unmask_tokens_idx = [
            processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        for message in hf_messages:
            role = message["role"]
            encode_id = processor.apply_chat_template([message], tokenize=True)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                # Revert image so that we recognize it in labels
                # Unmask later
                target[idx] = image_token_index

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    @staticmethod
    def _expand_image_tokens_labels(
        input_ids: torch.tensor,  # [bs x seq_len] with padding
        labels: List[torch.tensor],
        image_sizes: Iterable[Union[List[int], int]],
        height: int,
        width: int,
        special_token_idx: int,
        num_frames: int = 1,
        processor: AutoProcessor = None,
    ):
        new_labels = torch.zeros_like(input_ids)
        new_labels -= 100
        for batch_idx, label in enumerate(labels):
            special_token_pos = [
                idx for idx, la in enumerate(label.tolist()) if la == special_token_idx
            ]
            prev = 0
            for idx, pos in enumerate(special_token_pos):
                image_size_list = next(image_sizes)
                original_size = (
                    image_size_list[0] if num_frames != 1 else image_size_list
                )
                if not isinstance(original_size, (list, tuple)):
                    # cast to list to avoid numerical precision errors when calculating unpadding
                    original_size = original_size.tolist()
                orig_height, orig_width = original_size
                num_image_tokens = processor._get_number_of_features(
                    orig_height, orig_width, height, width
                )
                if processor.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1

                # Before image token, original labels
                new_labels[batch_idx, prev:pos] = label[prev:pos]
                # Expand image token
                new_labels[batch_idx, pos : pos + num_image_tokens] = -100

                if idx == len(special_token_pos) - 1:
                    # After last image token, original labels
                    last_label_shape = label[pos + 1 :].shape[0]
                    # The rest are just padding
                    new_labels[
                        batch_idx,
                        pos
                        + num_image_tokens : pos
                        + num_image_tokens
                        + last_label_shape,
                    ] = label[pos + 1 :]

                # Next interval will be from this image token + 1
                prev = pos + 1

        return new_labels
