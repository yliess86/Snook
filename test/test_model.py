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

    def test_encoder_decoder(self) -> None:
        enc = network.EncoderBlock(1, 1, t=1)
        dec = network.DecoderBlock(1, 1, t=1, scale=1)

        x = torch.zeros((2, 1, 8, 8))
        z, res = enc(x)
        y = dec(z, residual=res)

        assert tuple(y.size()) == (2, 1, 4, 4)

    def test_autoencoder(self) -> None:
        Layer = network.Layer
        autoenc = network.AutoEncoder(
            [Layer(4, 8, 1), Layer(8, 16, 6)], 3, 1
        )
        
        x = torch.zeros((2, 3, 256, 256))
        y = autoenc(x)

        assert tuple(y.size()) == (2, 1, 256, 256)