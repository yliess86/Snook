import numpy as np

from time import time
from tqdm import tqdm
from typing import Callable
from typing import List


class Benchmark:
    def __init__(self, name: str, outliers: float = 4.) -> None:
        self.name = name
        self.outliers = outliers
        self.samples: List[float] = []
        self.sealed = False

    @property
    def perf(self) -> float:
        samples = np.array(self.samples)
        data = np.abs(samples - np.median(samples))
        mdev = np.median(data)
        s = data / (mdev if mdev else 1)
        cleaned = samples[s < self.outliers]
        return np.mean(cleaned)

    @property
    def ms(self) -> float:
        return self.perf * 1000

    @property
    def fps(self) -> float:
        return 1 / self.perf

    def __call__(self, samples: int, fn: Callable) -> None:
        pbar = tqdm(range(samples), desc=f"[Snook][Benchmark]{self.name} Sampling")
        for _ in pbar:
            start = time(); fn(); end = time()
            self.append(end - start)
            pbar.set_postfix(ms=f"{self.ms:.2f}", fps=f"{self.fps:.2f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __enter__(self) -> "Benchmark":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.sealed = True

    def __repr__(self) -> str:
        return f"Benchmark(name={self.name}, samples={len(self)}, ms={self.ms:.2f}, fps={self.fps:.2f})"

    def reset(self) -> None:
        if not self.sealed:
            self.samples = []

    def append(self, dt: float) -> None:
        if not self.sealed:
            self.samples.append(dt)