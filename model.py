import torch.nn as nn

from quantisation import quantise


def encoder_block(loss_func, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        loss_func(),
    )

def decoder_block(loss_func, in_channels, out_channels, kernel_size, stride, padding, output_padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
        nn.BatchNorm2d(out_channels),
        loss_func(),
    )
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(  # 1x28x28
            encoder_block(nn.LeakyReLU, 1, 128, 3, 2, 1),  # 128x14x14
            encoder_block(nn.Tanh, 128, 8, 3, 2, 1)  # 8x7x7
        )
        
        self.quantise = quantise
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.quantise(x, scale_factor=1)
        
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = nn.Sequential(  # 8x7x7
            decoder_block(nn.LeakyReLU, 8, 128, 3, 2, 1, 1),  #128x14x14
            decoder_block(nn.LeakyReLU, 128, 128, 3, 2, 1, 1), #128x28x28
            nn.Conv2d(128, 1, 3, 1, 1),  #128x28x28
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
