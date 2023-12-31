import math
import functools
import torch

import torch.nn as nn

from generators.base_function import EncoderLayer, DecoderLayer, ToRGB

class Encoder(nn.Module):
    def __init__(
        self, 
        size, 
        input_dim, 
        channels, 
        num_labels=None, 
        match_kernels=None, 
        blur_kernel=[1, 3, 3, 1], 
        ):
        super().__init__()
        self.first = EncoderLayer(input_dim, channels[size], 1)
        self.convs = nn.ModuleList()
        # self.fcs = nn.ModuleList()

        log_size = int(math.log(size, 2))
        self.log_size = log_size

        in_channel = channels[size]
        for i in range(log_size-1, 3, -1):
            out_channel = channels[2 ** i]
            num_label = num_labels[2 ** i] if num_labels is not None else None
            match_kernel = match_kernels[2 ** i] if match_kernels is not None else None
            use_extraction = num_label and match_kernel
            conv = EncoderLayer(
                in_channel, 
                out_channel, 
                kernel_size=3, 
                downsample=True, 
                blur_kernel=blur_kernel,
                use_extraction=use_extraction,
                num_label=num_label,
                match_kernel=match_kernel
                )
            # fc_time = FC_time(out_channel)

            self.convs.append(conv)
            # self.fcs.append(fc_time)
            in_channel = out_channel

    def forward(self, input, recoder=None):
        out = self.first(input)
        for layer in self.convs:
            out = layer(out, recoder)
            # out = apply_conditions(out, fc_time(time_emb))
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        size,
        channels,
        num_labels,
        match_kernels,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()


        self.convs = nn.ModuleList()
        # input at resolution 16*16
        in_channel = channels[16]
        self.log_size = int(math.log(size, 2))
        
        for i in range(4, self.log_size + 1):
            out_channel = channels[2 ** i]
            num_label, match_kernel = num_labels[2 ** i], match_kernels[2 ** i]
            use_distribution = num_label and match_kernel
            upsample = (i != 4)
            
            base_layer = functools.partial(
                DecoderLayer,
                out_channel=out_channel,
                kernel_size=3, 
                blur_kernel=blur_kernel,
                use_distribution=use_distribution,
                num_label=num_label,
                match_kernel=match_kernel
                )

            up = nn.Module()   
            up.conv0 = base_layer(in_channel=in_channel, upsample=upsample)
            # up.fc_time0 = FC_time(out_channel)
            up.conv1 = base_layer(in_channel=out_channel, upsample=False)
            # up.fc_time1 = FC_time(out_channel)
            up.to_rgb = ToRGB(out_channel, upsample=upsample)
            self.convs.append(up)
            in_channel = out_channel
                
        self.num_labels, self.match_kernels = num_labels, match_kernels
    
    def forward(self, input, neural_textures, recoder):
        counter = 0
        out, skip = input, None
        for i, up in enumerate(self.convs):
            if self.num_labels[2**(i+4)] and self.match_kernels[2**(i+4)]:
                neural_texture_conv0 = neural_textures[counter]
                neural_texture_conv1 = neural_textures[counter+1]
                counter += 2
            else:
                neural_texture_conv0, neural_texture_conv1 = None, None
            out = up.conv0(out, neural_texture=neural_texture_conv0, recoder=recoder)
            # out = apply_conditions(out, up.fc_time0(time_emb))
            out = up.conv1(out, neural_texture=neural_texture_conv1, recoder=recoder)
            # out = apply_conditions(out, up.fc_time1(time_emb))
            skip = up.to_rgb(out, skip)
        image = skip
        return image


class FC_time(nn.Module) :
    def __init__(self, out_channel):
        super().__init__()
        self.fc_layer = nn.Sequential(nn.Linear(256, out_channel * 2),
                                 nn.SiLU(),
                                 nn.Linear(out_channel * 2, out_channel * 2), )
    def forward(self, time_emb):
        return self.fc_layer(time_emb)

def apply_conditions(h, emb=None):
    """
    apply conditions on the feature maps

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
    """

    while len(emb.shape) < len(h.shape):
        emb = emb[..., None]

    scale, shift = torch.chunk(emb, 2, dim=1)
    h = h * (1 + scale)
    h = h + shift

    return h