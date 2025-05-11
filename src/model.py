import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import glob

from src.fixed_params import get_c3_connectivity


class OutputRBFUnits(nn.Module):
    def __init__(self, in_dim, num_classes, ascii_prototypes=None):
        super().__init__()
        if ascii_prototypes is None:
            # initialize to +-1 (no custom bitmaps)
            protos = (
                torch.randint(0, 2, (num_classes, in_dim), dtype=torch.float32) * 2 - 1
            )
        else:
            protos = ascii_prototypes
        self.register_buffer("w", protos)  # fixed

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
        connectivity_mask: torch.Tensor,
    ):
        super().__init__(in_ch, out_ch, kernel_size, bias=True)
        # connectivity_mask: shape (out_ch, in_ch) with 1s where connected, 0s otherwise
        connectivity_mask = connectivity_mask.unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("connectivity_mask", connectivity_mask)

    def forward(self, x):
        w = self.weight * self.connectivity_mask
        return F.conv2d(x, w, bias=self.bias, stride=self.stride, padding=self.padding)


class Subsampling(nn.Module):
    def __init__(self, num_maps):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_maps))
        self.bias = nn.Parameter(torch.zeros(num_maps))

    def forward(self, x):
        # NOTE:
        # unfold(2,2,2) is along H, blocks of height 2, non-overlapping
        # unfold(3, 2, 2) is along W, blocks of width 2, non-overlapping
        neighborhoods = x.unfold(2, 2, 2).unfold(
            3, 2, 2
        )  # (N, C=num_maps, H//2, W//2, 2, 2)
        neighborhood_sum = neighborhoods.sum(dim=[4, 5])  # (N, C=num_maps, H//2, W//2)

        pre_act = neighborhood_sum * self.scale.view(1, -1, 1, 1) + self.bias.view(
            1, -1, 1, 1
        )
        return pre_act


class LeNet5(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        c3_connectivity: torch.Tensor = None,
        bitmap_dir: str = "data/bitmaps",
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,  # number of kernels
            kernel_size=5,
        )
        self.subs2 = Subsampling(num_maps=6)

        if c3_connectivity is None:
            c3_connectivity = get_c3_connectivity()
        self.conv3 = SparseConv2d(
            in_ch=6,
            out_ch=16,
            kernel_size=5,
            connectivity_mask=c3_connectivity,
        )

        self.subs4 = Subsampling(num_maps=16)

        self.conv5 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
        )
        self.fc6 = nn.Linear(
            in_features=120,  # output dim of prev. layer
            out_features=84,  # number of classes (symbols in alphabet)
        )

        self.rbf = OutputRBFUnits(
            in_dim=84, num_classes=num_classes, ascii_prototypes=None
        )
        if bitmap_dir:
            self.load_ascii_prototypes(bitmap_dir)

    def get_feature_embedding(self, x: torch.Tensor):
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
        c5 = c5.view(
            c5.size(0), -1
        )  # (N,120) # TODO: replace with c5 = torch.flatten(c5, 1)

        f6 = self.fc6(c5)
        f6 = self.scaled_tanh(f6)
        return f6

    def load_ascii_prototypes(self, bitmap_dir: str, device: torch.device = None):
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device

        proto_transform = transforms.Compose(
            [  # TODO:  move to utils or data_proc, and import from there
                transforms.Resize((32, 32), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        bitmap_files = sorted(glob.glob(f"{bitmap_dir}/char_*.png"))
        protos = []
        for fn in bitmap_files:
            img = Image.open(fn).convert("L")
            inp = proto_transform(img).unsqueeze(0)  # (1,1,32,32)
            with torch.no_grad():
                f6 = self.get_feature_embedding(inp)
            protos.append(f6.squeeze(0))
        proto_tensor = torch.stack(protos, dim=0)

        self.rbf.register_buffer("w", proto_tensor.to(device))

        if was_training:
            self.train()

    def scaled_tanh(self, x):
        # The scaling params. A & S are chosen so that
        # f(1) ~ 1 & f(-1) ~ -1
        # See appendix A of LeCun et al. 1998
        A = 1.7159
        S = 2 / 3
        return A * torch.tanh(S * x)

    def forward(self, x):
        f6 = self.get_feature_embedding(x)
        logits = self.rbf(f6)
        return logits
