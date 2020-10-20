import numpy as np

from PIL import Image, ImageFilter


class RandomGaussianBlur:
    def __init__(self, radius: int, p: float = 0.5) -> None:
        self.radius = radius
        self.p = p

    def __call__(self, img: Image) -> Image:
        if np.random.rand() < self.p:
            radius = np.random.rand() * self.radius
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img