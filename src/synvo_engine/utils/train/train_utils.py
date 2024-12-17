from typing import Iterable, List, Union

import torch
from loguru import logger
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
                elif content["type"] == "audio_url":
                    new_message["content"].append(
                        {"type": "audio", "audio_url": content["audio_url"]["url"]}
                    )
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
    def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
        to_return = {
            k: t
            for k, t in named_params
            if any(key_match in k for key_match in keys_to_match)
        }
        to_return = {
            k: TrainUtilities.maybe_zero_3(v, ignore_status=True).cpu()
            for k, v in to_return.items()
        }
        return to_return

    @staticmethod
    def maybe_zero_3(param, ignore_status=False, name=None):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if not ignore_status:
                    logger.warning(
                        f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                    )
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
        return param
