import snook.model.network as network
import torch


class TestModelNetwork:
    def test_convbn2d(self) -> None:
        convbn2d = network.ConvBn2d(1, 1, kernel_size=1)
        out = convbn2d(torch.zeros((2, 1, 8, 8)))
        assert tuple(out.size()) == (2, 1, 8, 8)

    def test_inverted_residual(self) -> None:
        invres = network.InvertedResidual(1, 1)
        out = invres(torch.zeros((2, 1, 8, 8)))
        assert tuple(out.size()) == (2, 1, 8, 8)