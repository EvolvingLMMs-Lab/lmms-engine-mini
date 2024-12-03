import unittest

from torch.utils.data import DataLoader

from synvo_engine.datasets import DatasetConfig, DatasetFactory


class TestVisionDataset(unittest.TestCase):
    def test_sft_dataset(self):
        config = {
            "dataset_type": "vision",
            "dataset_format": "json",
            "dataset_path": "./examples/sample_json_data/synvo_engine.json",
            "processor_name": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        }

        dataset_config = DatasetConfig(**config)
        dataset = DatasetFactory.create_dataset(dataset_config)
        dataset.build()
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
        for data in dataLoader:
            assert data["image_sizes"][0, 0, 0] == 480
            break


if __name__ == "__main__":
    unittest.main()
