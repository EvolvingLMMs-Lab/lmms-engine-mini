from ..utils.train import TrainUtilities
from .collator import GRPOCollator
from .vision_audio_dataset import VisionAudioSFTDataset

GRPO_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
MAX_AUDIO_LENGTH = 30


class GRPOPreferenceDataset(VisionAudioSFTDataset):
    def load_from_json(self, data, data_folder=None):
        images = []
        audios = []
        messages = data["messages"]
        new_messages = []
        solutions = []
        for message in messages:
            # Here is assuming the data is just Q and A with text output
            if message["role"] == "assistant":
                solutions.append(message["content"][0]["text"])
                continue
            new_content = []
            for idx, content in enumerate(message["content"]):
                if content["type"] == "image_url":
                    images.append(
                        self.load_image(
                            content["image_url"]["url"], data_folder=data_folder
                        )
                    )
                    new_content.append(content)
                elif content["type"] == "audio_url":
                    loaded_audios = self.load_audio(
                        content["audio_url"]["url"],
                        sr=self.processor.sampling_rate,
                        data_folder=data_folder,
                    )
                    audio_splits = []
                    # Split the loaded audio to 30s chunks and extend the messages content
                    for i in range(
                        0,
                        len(loaded_audios),
                        MAX_AUDIO_LENGTH * self.processor.sampling_rate,
                    ):
                        audio_splits.append(
                            loaded_audios[
                                i : i + MAX_AUDIO_LENGTH * self.processor.sampling_rate
                            ]
                        )
                    for _ in range(len(audio_splits)):
                        new_content.append(content)
                    audios.extend(audio_splits)
                else:
                    new_content.append(content)
            message["content"] = new_content
            new_messages.append(message)
        messages = new_messages

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if len(images) == 0:
            images = None
        if len(audios) == 0:
            audios = None
        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            audios=audios,
            sampling_rate=self.processor.sampling_rate,
            system_message=GRPO_SYSTEM_PROMPT,
            add_generation_prompt=True,
        )
        inputs["solution"] = solutions
        return inputs

    def get_collator(self):
        return GRPOCollator(self.processor)
