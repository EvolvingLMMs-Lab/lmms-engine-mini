# Data Preparation

You need to prepare a YAML file to specify the data path and data type. The YAML file should look like this:

```yaml
datasets:
  - json_path: train.json
    data_folder: .
    data_type: json
```

Below is an example script which may help you easier to prepare the YAML file. This script will download the dataset from the Hugging Face Hub and extract the dataset from the `tar.gz` file. You can modify the script to fit your needs.

```python
import tarfile
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

REPO_ID = "Evo-LMM/rlaif-v"
DATA_TYPE = "json"
DATA_PATH = "train.json"
OUTPUT_YAML = "data.yaml"

if __name__ == "__main__":
    data_path = snapshot_download(
        repo_id="Evo-LMM/rlaif-v",
        repo_type="dataset",
    )
    data_path = Path(data_path)
    for image_zip in data_path.glob("*.tar.gz"):
        with tarfile.open(image_zip, "r:gz") as tar:
            tar.extractall(path=data_path)
    data_dict = {
        "datasets": [
            {
                "json_path": str(data_path / DATA_PATH),
                "data_folder": str(data_path),
                "data_type": DATA_TYPE,
            }
        ]
    }
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(data_dict, f)
```
