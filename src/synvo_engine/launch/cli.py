import argparse
import json

from .controller import Controller


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to your launch config")
    return parser.parse_args()


def main():
    args = parse_argument()
    with open(args.config, "r") as f:
        config = json.load(f)
    controller = Controller(config)
    controller.run()


if __name__ == "__main__":
    main()
