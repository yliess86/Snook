class Trainer:
    def train(self) -> None:
        raise NotImplementedError("Training Step not Implemented yet!")

    def valid(self) -> None:
        raise NotImplementedError("Validation Step not Implemented yet!")

    def test(self) -> None:
        raise NotImplementedError("Testing Step not Implemented yet!")