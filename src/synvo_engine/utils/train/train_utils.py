class TrainUtilities:
    @staticmethod
    def prepare_model():
        pass

    @staticmethod
    def convert_open_to_hf(messages):
        hf_messages = []
        for message in messages:
            new_message = {"role": message["role"], "content": []}
            for content in message["content"]:
                if content["type"] == "image_url":
                    new_message["content"].append({"type": "image"})
                else:
                    new_message["content"].append(content)
            hf_messages.append(new_message)

        return hf_messages
