import collections
from torch import nn
import torch
from generators.base_module import Encoder, Decoder
import math

class Generator(nn.Module):
    def __init__(
        self,
        size,
        semantic_dim,
        channels,
        num_labels,
        match_kernels,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.size = size
        self.reference_encoder = Encoder(
            size, 3, channels, num_labels, match_kernels, blur_kernel
        )
            
        self.skeleton_encoder = Encoder(
            size, semantic_dim+3, channels,
            )

        self.target_image_renderer = Decoder(
            size, channels, num_labels, match_kernels, blur_kernel
        )

    def _cal_temp(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(self,
                ref_image,
                input_image, input_skeleton, timestep
                ):
        output_dict={}
        recoder = collections.defaultdict(list)
        b, c, h, w = ref_image.size()
        time_emb = timestep_embedding(timestep).to(ref_image.device).view(b, 1, h, w)

        input_ = torch.cat([input_image + time_emb, input_skeleton], 1)
        skeleton_feature = self.skeleton_encoder(input_)
        _ = self.reference_encoder(ref_image + time_emb, recoder = recoder)

        neural_textures = recoder["neural_textures"]
        output_dict['fake_image'] = self.target_image_renderer(
            skeleton_feature, neural_textures, recoder
            )
        output_dict['info'] = recoder
        return output_dict


def timestep_embedding(timesteps, dim=256*176, max_period=1000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding