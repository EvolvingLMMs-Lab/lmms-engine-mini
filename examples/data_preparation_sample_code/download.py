from huggingface_hub import snapshot_download
from pathlib import Path
import tarfile


if __name__ == "__main__":
    data_path = snapshot_download(repo_id="Evo-LMM/rlaif-v", repo_type="dataset", local_dir="/data/pufanyi/project/synvo_engine/playground/data/Evo-LMM/rlaif-v")
    data_path = Path(data_path)
    for image_zip in data_path.glob("*.tar.gz"):
        with tarfile.open(image_zip, "r:gz") as tar:
            tar.extractall(path=data_path)
