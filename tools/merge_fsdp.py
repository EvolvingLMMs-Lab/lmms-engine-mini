import argparse
import importlib

import torch
import torch.distributed.checkpoint as dist_cp
from accelerate import init_empty_weights

from lmms_engine.models.factory import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge FSDP shards into a single checkpoint."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the FSDP shards to merge.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output file for the merged checkpoint.",
    )
    parser.add_argument("--model_name_or_class", type=str, default="")
    return parser.parse_args()


def main(args):
    model_path = args.model_name_or_class
    model_cls = ModelFactory.create_model(model_path)
    model = model_cls.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    state_dict = {"model": model.state_dict()}
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(args.input_dir),
        no_dist=True,
    )
    model.load_state_dict(state_dict["model"])
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
