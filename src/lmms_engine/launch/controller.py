from .pipeline import Pipeline


class Controller:
    def __init__(self, config: dict) -> None:
        self.config = config

    def create_pipe(self):
        pipe = Pipeline(self.config)
        return pipe

    def run(self, **kwargs):
        pipe = self.create_pipe()
        pipe.run(**kwargs)
