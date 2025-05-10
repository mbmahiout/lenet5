import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputRBFUnits(nn.Module):
    def __init__(self, in_dim, num_classes, ascii_prototypes=None):
        super().__init__()
        if ascii_prototypes is None:
            # initialize to +-1 (no custom bitmaps)
            protos = torch.randint(0, 2, (num_classes, in_dim), dtype=torch.float32) * 2 - 1
        else:
            protos = ascii_prototypes
        self.register_buffer('w', protos)  # fixed

    def forward(self, x):
        # x: (N, in_dim)
        # compute squared Euclidean distance to each w: (N, num_classes)
        diff = x.unsqueeze(1) - self.w.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=2)
        logits = -dist_sq
        return logits


class SparseConv2d(nn.Conv2d):
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int, 
            kernel_size: int, 
            connectivity_mask: torch.Tensor=None, 
            connectivity_is_random: bool=False
        ):
        super().__init__(in_ch, out_ch, kernel_size, bias=True)
        # connectivity_mask: shape (out_ch, in_ch) with 1s where connected, 0s otherwise
        if connectivity_mask is not None:
            mask = connectivity_mask
        elif connectivity_is_random:
            mask = (torch.rand(out_ch, in_ch) < 0.5).float()  # ~ half of the neurons are connected
        else:
            mask = torch.ones(out_ch, in_ch) # fully connected

        mask = mask.unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(
            x, 
            w, 
            bias=self.bias,
            stride=self.stride, 
            padding=self.padding
        )


class Subsampling(nn.Module):
    def __init__(self, num_maps):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_maps)) 
        self.bias = nn.Parameter(torch.zeros(num_maps)) 

    def forward(self, x):
        # NOTE:
        # unfold(2,2,2) is along H, blocks of height 2, non-overlapping
        # unfold(3, 2, 2) is along W, blocks of width 2, non-overlapping
        neighborhoods = x.unfold(2,2,2).unfold(3,2,2)  # (N, C=num_maps, H//2, W//2, 2, 2)
        neighborhood_sum = neighborhoods.sum(dim=[4,5]) # (N, C=num_maps, H//2, W//2)

        pre_act = neighborhood_sum * self.scale.view(1,-1,1,1) + self.bias.view(1,-1,1,1)
        return pre_act
    

class LeNet5(nn.Module):
    def __init__(
            self, 
            num_classes: int=10, 
            c3_connectivity: torch.Tensor=None, 
            c3_random: bool=False
        ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=6, # number of kernels
            kernel_size=5, 
        )
        self.subs2 = Subsampling(num_maps=6)

        self.conv3 = SparseConv2d(
            in_ch=6, 
            out_ch=16, 
            kernel_size=5,
            connectivity_mask=c3_connectivity # TODO
        )

        self.subs4 = Subsampling(num_maps=16)
        
        self.conv5 = nn.Conv2d(
            in_channels=16, 
            out_channels=120,
            kernel_size=5, 
        )
        self.fc6 = nn.Linear(
            in_features=120, # output dim of prev. layer
            out_features=84, # number of classes (symbols in alphabet)
        )

        self.rbf = OutputRBFUnits(
            in_dim=84, 
            num_classes=num_classes, 
            ascii_prototypes=None
        )

    def scaled_tanh(self, x):
        # The scaling params. A & S are chosen so that
        # f(1) ~ 1 & f(-1) ~ -1
        # See appendix A of LeCun et al. 1998
        A = 1.7159
        S = 2/3
        return A * torch.tanh(S*x)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.scaled_tanh(c1)

        s2 = self.subs2(c1)
        s2 = self.scaled_tanh(s2)

        c3 = self.conv3(s2)
        c3 = self.scaled_tanh(c3)

        s4 = self.subs4(c3)
        s4 = self.scaled_tanh(s4)

        c5 = self.conv5(s4)
        c5 = self.scaled_tanh(c5)
        c5 = c5.view(c5.size(0), -1) # (N,120)

        f6 = self.fc6(c5)
        f6 = self.scaled_tanh(f6)

        logits = self.rbf(f6)

        return logits