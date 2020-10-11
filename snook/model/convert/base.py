class Convertor:
    def __call__(self, debug: bool = False) -> None:
        self.convert()
        self.compare()
        if debug:
            self.debug()

    def convert(self) -> None:
        self.vanilla()
        self.jit()
        self.onnx()

    def vanilla(self) -> None:
        raise NotImplementedError("Vanilla not implemented yet!")

    def jit(self) -> None:
        raise NotImplementedError("Jit not implemented yet!")
        
    def onnx(self, ops: int = 11) -> None:        
        raise NotImplementedError("Onnx not implemented yet!")
        
    def compare(self) -> None:
        raise NotImplementedError("Compare not implemented yet!")

    def debug(self) -> None:
        raise NotImplementedError("Debug not implemented yet!")