import unittest

from torch.utils.data import DataLoader

from synvo_engine.datasets import DatasetConfig, DatasetFactory
from synvo_engine.utils.train import TrainUtilities


class TestVisionDataset(unittest.TestCase):
    def test_sft_dataset(self):
        config = {
            "dataset_type": "vision",
            "dataset_format": "json",
            "dataset_path": "./examples/sample_json_data/synvo_engine.json",
            "processor_name": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            "chat_template": "qwen",
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        collator = dataset.get_collator()
        dataLoader = DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collator
        )
        for data in dataLoader:
            # TrainUtilities.sanity_check_labels(dataset.processor, data["input_ids"], data["labels"])
            print([f"{key}: {value.shape}" for key, value in data.items()])
            break

    def test_load_hf_dataset(self):
        config = {
            "dataset_type": "vision",
            "dataset_format": "hf_dataset",
            "dataset_path": "kcz358/LLaVA-NeXT-20k",
            "processor_name": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            "chat_template": "qwen",
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        collator = dataset.get_collator()
        dataLoader = DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collator
        )
        for data in dataLoader:
            # TrainUtilities.sanity_check_labels(dataset.processor, data["input_ids"], data["labels"])
            print([f"{key}: {value.shape}" for key, value in data.items()])
            break


if __name__ == "__main__":
    unittest.main()
