from ..aero import AeroConfig


class AeroOmniConfig(AeroConfig):
    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_token_index=151648,
        projector_hidden_act="gelu",
        projector_type="mlp",
        tie_word_embeddings=False,
        code_book_size=4096,
        num_codebooks=7,
        **kwargs,
    ):
        self.code_book_size = code_book_size
        self.num_codebooks = num_codebooks
        super().__init__(
            text_config=text_config,
            audio_config=audio_config,
            audio_token_index=audio_token_index,
            projector_hidden_act=projector_hidden_act,
            projector_type=projector_type,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
