import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchinfo import summary

#TODO: probably won't need this file in the official submission. can delete later.

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc



class Transformer2(nn.Module):
    """
        Bidirectional Transformer For Zoonosis Prediction.
    """
    def __init__(self, 
                 in_dim: int,# vocab size? very beginning of model
                 out_dim: int,        # vocab size? very end of model
                 num_out_tokens: int, # length of sequence? taregt dimension? #TODO: I don't think we need these... or at least only one is needed.
                 **kwargs #d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,dim_feedforward=2048, dropout=0.1,device=None,
                ):
        super().__init__()
        
        d_model = kwargs["d_model"]
        
        self.d_model = d_model
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_out_tokens = num_out_tokens
        
        self.in_proj = nn.Linear(in_dim, d_model)
        self.out_proj = nn.Linear(d_model, out_dim)
        
        self.pos_enc = PositionalEncoding1D(d_model)
        self.pos_proj = nn.Linear(d_model, d_model)

        # out_token = torch.randn(num_out_tokens, d_model)
        # out_token = nn.Parameter(
        #     out_token, requires_grad=True
        # )
        # self.register_parameter("out_token", out_token)
            
        self.transformer = nn.Transformer(**kwargs)
        self.config = {}
        self.config['vocab_size']= in_dim
        self.type_ = "t2"
        
        print('transformer', self)
        summary(self, (1, num_out_tokens), dtypes=[torch.long])
        
    def get_block_size(self):
        return self.num_out_tokens
    
    def forward(self, x: Tensor):
        # print('make prediction', x.shape)
        if x.dtype == torch.int64 or x.ndim == 2:
            x = torch.clip(x, max=self.in_dim - 1)
            x = F.one_hot(x, self.in_dim)
            x = x.float()#.type(self.out_token.dtype)
        x = self.in_proj(x)
        in_token = self.pos_enc(x) + x
        
        
        # out_token = self.out_token.repeat(
        #     len(in_token), 1, 1
        # )
        in_token = in_token.permute(1, 0, 2)
        # print(in_token.shape, out_token.shape)
        out = self.transformer.encoder(src=in_token)
        out = out.permute(1, 0, 2)


        out = self.out_proj(out)
        # print('logits',out.shape)
        return out, None