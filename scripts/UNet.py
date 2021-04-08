import torch
import torch.nn as nn
from torch.nn import init

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self, encoder_channels):
        super(Encoder, self).__init__()
        self.encoder_channels = encoder_channels
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_channels)-1):
            self.encoder_blocks.append(Block(encoder_channels[i], encoder_channels[i+1]))
        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)
    def forward(self, x):
        filter_response = []
        for block in self.encoder_blocks:
            x = block(x)
            filter_response.append(x)
            x = self.pool(x)
        return filter_response

class Decoder(nn.Module):
    def __init__(self, decoder_channels):
        super(Decoder, self).__init__()
        self.decoder_channels= decoder_channels
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)-1):
            self.decoder_blocks.append(Block(decoder_channels[i], decoder_channels[i+1]))
            self.upconvs.append(nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], 2, 2))
        
    def forward(self, x, encoder_features):
        for i in range(len(self.decoder_channels)-1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.decoder_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, number_of_steps, number_of_class = 3):
        super(UNet, self).__init__()
        self.encoder_channels = []
        self.decoder_channels = [] 
        for i in range(number_of_steps+1):
            if i==0:
                self.encoder_channels.append(1)
            else:
                self.encoder_channels.append(64*(2**(i-1)))
        for i in range(number_of_steps-1, -1, -1):
            self.decoder_channels.append(64*(2**(i)))
        self.encoder = Encoder(self.encoder_channels)
        self.decoder = Decoder(self.decoder_channels)
        self.head = nn.Conv2d(64, number_of_class, 1)
        self.reset_params()
    
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
            
    def forward(self, x):
        filter_response = self.encoder(x)
        out = self.decoder(filter_response[::-1][0], filter_response[::-1][1:]) # [::-1] inverts the list!
        out = self.head(out)
        return out

def main():
    import numpy as np
    unet = UNet(number_of_steps=3, number_of_class=3)
    x = torch.randn(1, 1, 256, 256)
    print(unet(x).shape) # 1 3 256 256 ==> 3 channels = (f.g, b.g, boundary)

    model_parameters = filter(lambda p: p.requires_grad, unet.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable params = {}".format(params)) # 1861827 parameters for depth = 3 (i.e. number_of_steps = 3)
