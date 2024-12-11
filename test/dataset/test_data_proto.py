import json
import unittest

from pydantic import ValidationError

from synvo_engine.protocol import SFTChatData


class TestDataProto(unittest.TestCase):
    def test_json_data(self):
        with open("./examples/sample_json_data/synvo_engine.json", "r") as f:
            data = json.load(f)

        for da in data:
            try:
                checked = SFTChatData(**da)
            except ValidationError as e:
                print(e.errors())


if __name__ == "__main__":
    unittest.main()
