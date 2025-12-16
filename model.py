import torch.nn as nn

from quantisation import quantise


def encoder_block(loss_func, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        loss_func(),
    )

def decoder_block(loss_func, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        loss_func(),
    )
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            encoder_block(nn.LeakyReLU, 3, 128, 3, 2, 1),
            encoder_block(nn.LeakyReLU, 128, 128, 3, 2, 1),
            encoder_block(nn.Tanh, 128, 3, 3, 2, 1)
        )
        
        self.quantise = quantise
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.quantise(x, scale_factor=1)
        
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = nn.Sequential(
            decoder_block(nn.LeakyReLU, 3, 128, 3, 2, 1),
            decoder_block(nn.LeakyReLU, 128, 128, 3, 2, 1),
            decoder_block(nn.LeakyReLU, 128, 128, 3, 2, 1),
            nn.Conv2d(128, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.decoder(x)
        return x
        
class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
