from ..aero import AeroProcessor


class AeroOmniProcessor(AeroProcessor):
    valid_kwargs = [
        "chat_template",
        "audio_token",
        "audio_pad_token",
        "audio_special_token_prefix",
    ]

    def __init__(
        self,
        tokenizer=None,
        audio_processor=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        audio_pad_token="<|audio_pad|>",
        audio_special_token_prefix="<|audio_token_",
        **kwargs,
    ):
        self.audio_pad_token = audio_pad_token
        self.audio_special_token_prefix = audio_special_token_prefix
        super().__init__(
            tokenizer, audio_processor, chat_template, audio_token, **kwargs
        )
