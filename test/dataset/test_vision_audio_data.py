import unittest

from torch.utils.data import DataLoader

from synvo_engine.datasets import DatasetConfig, DatasetFactory
from synvo_engine.utils.train import TrainUtilities


class TestVisionDataset(unittest.TestCase):
    def test_sft_dataset(self):
        config = {
            "dataset_type": "vision_audio",
            "dataset_format": "jsonl",
            "dataset_path": "./examples/sample_jsonl_data/voice_assis/voice_assis.jsonl",
            "processor_config": {
                "processor_name": "kcz358/kino-7b-init",
                "processor_modality": "vision_audio",
                "processor_type": "kino",
            },
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        collator = dataset.get_collator()
        dataLoader = DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collator
        )
        for data in dataLoader:
            # TrainUtilities.sanity_check_labels(dataset.processor.processor, data["input_ids"], data["labels"])
            print([f"{key}: {value.shape}" for key, value in data.items()])
            break


if __name__ == "__main__":
    unittest.main()
