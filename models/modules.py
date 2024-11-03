import torch
import torch.nn as nn
import numpy as np


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                assert 0 == 1  # error
        return tokens


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=2):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.SiLU(),
        )

    def forward(self, x):
        y = self.fc(x) + x
        return y


class CovLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer, self).__init__()
        self.fc_33 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_11 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_77 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_55 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_51 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding="same")
                for _ in range(k)
            ]
        )
        self.fc_15 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding="same")
                for _ in range(k)
            ]
        )

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                x = tokens[i][:, 1:, :]
                x = x.view(
                    x.shape[0],
                    int(np.sqrt(x.shape[1])),
                    int(np.sqrt(x.shape[1])),
                    x.shape[2],
                )
                # print(x.shape)
                x_temp = (
                    self.fc_11[i](x.permute(0, 3, 1, 2))
                    + self.fc_33[i](x.permute(0, 3, 1, 2))
                    + self.fc_55[i](x.permute(0, 3, 1, 2))
                    + self.fc_77[i](x.permute(0, 3, 1, 2))
                    + self.fc_15[i](x.permute(0, 3, 1, 2))
                    + self.fc_51[i](x.permute(0, 3, 1, 2))
                )
                tokens[i] = x_temp
                tokens[i] = (
                    tokens[i]
                    .permute(0, 2, 3, 1)
                    .view(tokens[i].shape[0], -1, tokens[i].shape[1])
                )
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](
                    tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous()
                )
        return tokens


if __name__ == "__main__":
    
    convLayer = CovLayer(1024, 768, 3)
    
    x = torch.ones(4, 4, 16*16 + 1, 1024)
    x = convLayer(x)
    pass