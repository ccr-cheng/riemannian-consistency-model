from training.networks_edm2 import *


class Block(torch.nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        emb_channels,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])

        self.x_linear_skip = MPConv(in_channels, out_channels, kernel=[]) if in_channels != out_channels else None
        
        self.x_linear_1 = MPConv(out_channels, out_channels, kernel=[])
        self.x_linear_2 = MPConv(out_channels, out_channels, kernel=[])
    
    def forward(self, x, emb):

        # x: [BS, 1, in_channels], emb: [BS, emb_channels]
        if self.x_linear_skip is not None:
            x = self.x_linear_skip(x)
        
        x = normalize(x, dim=-1)

        # Residual branch.
        y = self.x_linear_1(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(1).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.x_linear_2(y)

        # Connect the branches.
        x = mp_sum(x, y, t=0.3)

        return x


class MPModel(torch.nn.Module):
    def __init__(
        self, 
        in_channels=2,
        base_channels=128,
        x_channel_mult=[2, 4, 4, 2],
        emb_channel_mult=2,
        dropout=0.0,
    ):
        super().__init__()
        self.x_channel = [x * base_channels for x in x_channel_mult]
        self.emb_channel = emb_channel_mult * base_channels
        self.dropout = dropout

        self.emb_fourier = MPFourier(base_channels)
        self.emb_noise = MPConv(base_channels, self.emb_channel, kernel=[])

        self.enc = torch.nn.ModuleList()
        for i in range(len(self.x_channel)):
            if i == 0:
                mlp = MPConv(in_channels + 1, self.x_channel[0], kernel=[])
            else:
                mlp = Block(self.x_channel[i - 1], self.x_channel[i], self.emb_channel, dropout)
            self.enc.append(mlp)

        self.dec = torch.nn.ModuleList()
        self.dec.append(Block(self.x_channel[-1], self.x_channel[-1], self.emb_channel, dropout))
        for i in range(len(self.x_channel) - 1, 0, -1):
            mlp = Block(self.x_channel[i] * 2, self.x_channel[i - 1], self.emb_channel, dropout)
            self.dec.append(mlp)

        self.out_conv = MPConv(2 * self.x_channel[0], in_channels, kernel=[])
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        
    def forward(self, x, noise_labels):

         # x: [BS, 1, 2], emb: [BS,]
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        skips = []
        for i, block in enumerate(self.enc):
            if i == 0:
                x = block(x)
            else:
                x = block(x, emb)
            skips.append(x)
        
        # Decoder.
        for i, block in enumerate(self.dec):
            if i != 0:
                x = mp_cat(x, skips.pop(), dim=-1, t=0.5)
            x = block(x, emb)

        x = self.out_conv(mp_cat(x, skips.pop(), dim=-1, t=0.5), gain=self.out_gain)
        return x
    

@persistence.persistent_class
class FlowPrecond(torch.nn.Module):
    def __init__(self,
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        in_channels     = 2,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model = MPModel(in_channels, **model_kwargs)

    def forward(self, x, t):
        F_x = self.model(x, t.flatten())
        return F_x
