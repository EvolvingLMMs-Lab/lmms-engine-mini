import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="lmms-lab/LLaVA-NeXT-Data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_folder", type=str, default="./data")

    return parser.parse_args()


def convert_llava_to_openai(content, image_path: str):
    messages = []
    for item in content:
        if item["from"] == "human":
            content = []
            if "<image>" in item["value"]:
                content.append({"type": "image_url", "image_url": {"url": image_path}})
            content.append(
                {"type": "text", "text": item["value"].replace("<image>", "")}
            )
            messages.append({"role": "user", "content": content})
        elif item["from"] == "gpt":
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["value"]}],
                }
            )

    return messages


if __name__ == "__main__":
    args = parse_argument()
    dataset = load_dataset(args.dataset_path, split=args.split)
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "images"), exist_ok=True)
    json_dataset = []
    pbar = tqdm(total=len(dataset), desc="Saving...")
    for item in dataset:
        image_path = os.path.join(args.output_folder, "images", item["id"] + ".jpg")
        item["image"].convert("RGB").save(image_path)
        messages = convert_llava_to_openai(item["conversations"], image_path)
        json_dataset.append({"id": item["id"], "messages": messages})
        pbar.update(1)

    with open(os.path.join(args.output_folder, "synvo_engine.json"), "w") as f:
        json.dump(json_dataset, f, indent=4)
