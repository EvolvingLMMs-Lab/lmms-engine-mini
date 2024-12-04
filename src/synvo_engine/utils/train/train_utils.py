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
                    new_message["content"].append(content)
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
