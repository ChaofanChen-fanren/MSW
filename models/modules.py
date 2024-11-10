import math
import torch
import torch.nn as nn
import numpy as np
from functools import partial
# from .deformable_attention.ops.modules import MSDeformAttn
from torch.nn import MultiheadAttention
import torch.nn.functional as F


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
        # 随机初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # 使用 Xavier 均匀分布初始化权重
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # 使用零初始化偏置

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


# Muti-layer visual feature fusion
class MLVFusion(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_levels=4,
                 n_heads=16,
                 n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropout=0.1, init_values=0.,
                 deformable_attention=True):
        super().__init__()
        self.select_layer = [_ for _ in range(n_levels)]
        self.query_layer = -1
        self.deformable_attention = deformable_attention

        if self.deformable_attention:
            self.cross_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        else:
            self.cross_attn = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.gamma1 = nn.Parameter(init_values * torch.ones(d_model), requires_grad=True)

        self.norm1 = norm_layer(d_model)
        if self.deformable_attention:
            self.self_attn = MSDeformAttn(d_model=d_model, n_levels=1, n_heads=n_heads, n_points=n_points)
        else:
            self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gamma2 = nn.Parameter(init_values * torch.ones(d_model), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            # if isinstance(m, MSDeformAttn):
            #     m._reset_parameters()
            if isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, srcs, masks=None, pos_embeds=None):
        # prepare input feat
        src_flatten = []
        spatial_shapes = []
        for lvl in self.select_layer:
            src = srcs[lvl]
            _, hw, _ = src.shape
            e = int(math.sqrt(hw))
            spatial_shape = (e, e)
            spatial_shapes.append(spatial_shape)
            src_flatten.append(src)
        feat = torch.cat(src_flatten, 1)

        if self.deformable_attention:
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

            # cross attn
            pos = None  # TODO
            query = srcs[self.query_layer]
            query = self.with_pos_embed(query, pos)  # bs, h*w, c
            query_e = int(math.sqrt(query.shape[1]))  # h == w

            reference_points = self.get_reference_points([(query_e, query_e)], device=query.device)
            attn = self.cross_attn(self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes,
                                   level_start_index, None)

            # self attn
            attn1 = self.norm1(attn)
            attn_pos = None  # TODO
            spatial_shapes_attn = torch.as_tensor([(query_e, query_e)], dtype=torch.long, device=attn1.device)
            level_start_index_attn = torch.cat(
                (spatial_shapes_attn.new_zeros((1,)), spatial_shapes_attn.prod(1).cumsum(0)[:-1]))
            reference_points_attn = self.get_reference_points(spatial_shapes_attn, device=attn1.device)
            attn2 = self.self_attn(self.with_pos_embed(attn1, attn_pos), reference_points_attn, attn1,
                                   spatial_shapes_attn,
                                   level_start_index_attn, None)
            attn = attn + self.gamma2 * attn2

            # Residual Connection
            tgt = query + self.gamma1 * attn
        else:
            # cross attn
            query = srcs[self.query_layer]
            pos = None  # TODO
            query = self.with_pos_embed(query, pos)  # bs, h*w, c
            attn, _ = self.cross_attn(self.query_norm(query), self.feat_norm(feat), self.feat_norm(feat))

            # self attn
            attn1 = self.norm1(attn)
            attn_pos = None  # TODO
            attn2, _ = self.self_attn(self.with_pos_embed(attn1, attn_pos), attn1, attn1)
            attn = attn + self.gamma2 * attn2

            # Residual Connection
            tgt = query + self.gamma1 * attn
        return tgt


class VisualPerceptionModule(nn.Module):
    def __init__(self,
                 dim: int,
                 n_alpha: int,
                 n_beta: int,
                 ):
        super().__init__()
        self.alpha_linear = nn.Linear(dim, n_alpha)
        self.beta_linear = nn.Linear(dim, n_beta)
        self.act = nn.GELU()

    def forward(self, srcs):
        x, y, z = srcs
        alpha = self.alpha_linear(x).reshape(x.shape[0], 1, -1)
        beta = self.beta_linear(x).reshape(x.shape[0], 1, -1)

        # y : lower_feature, z : higher_feature

        tgt = alpha * y + beta * z
        tgt = self.act(tgt) * tgt
        return tgt


# multi-scale Window Convolution
class MSW(nn.Module):
    def __init__(self,
                 dim: int = 768,
                 n_levels: int = 4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Multiscale Large Kernel Attention
        self.LKA1 = nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
        self.LKA7 = nn.Sequential(
            nn.Conv2d(chunk_dim, chunk_dim, 7, 1, 7 // 2, groups=chunk_dim),
            nn.Conv2d(chunk_dim, chunk_dim, 9, stride=1, padding=(9 // 2) * 4, groups=chunk_dim, dilation=4),
            nn.Conv2d(chunk_dim, chunk_dim, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(chunk_dim, chunk_dim, 5, 1, 5 // 2, groups=chunk_dim),
            nn.Conv2d(chunk_dim, chunk_dim, 7, stride=1, padding=(7 // 2) * 3, groups=chunk_dim, dilation=3),
            nn.Conv2d(chunk_dim, chunk_dim, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim),
            nn.Conv2d(chunk_dim, chunk_dim, 5, stride=1, padding=(5 // 2) * 2, groups=chunk_dim, dilation=2),
            nn.Conv2d(chunk_dim, chunk_dim, 1, 1, 0))

        self.LKA = [self.LKA1, self.LKA3, self.LKA5, self.LKA7]

        # Gate
        self.X1 = nn.Conv2d(chunk_dim, chunk_dim, 1, 1, 0, groups=chunk_dim)
        self.X3 = nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
        self.X5 = nn.Conv2d(chunk_dim, chunk_dim, 5, 1, 5 // 2, groups=chunk_dim)
        self.X7 = nn.Conv2d(chunk_dim, chunk_dim, 7, 1, 7 // 2, groups=chunk_dim)

        self.X = [self.X1, self.X3, self.X5, self.X7]

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()
        # self.proj_first = nn.Sequential(
        #     nn.Conv2d(n_feats, i_feats, 1, 1, 0))
        #
        # self.proj_last = nn.Sequential(
        #     nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        B, hw, dim = x.shape
        h = w = int(math.sqrt(hw))
        x = x.reshape(B, dim, h, w)

        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.LKA[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.LKA[i](xc[i])
            x_i = self.X[i](xc[i]) * s
            out.append(x_i)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x

        out = out.reshape(B, h * w, dim)
        return out


if __name__ == "__main__":
    # convLayer = CovLayer(1024, 768, 3)
    #
    # x = torch.ones(4, 4, 16*16 + 1, 1024)
    # x = convLayer(x)

    # block = MLVFusion(d_model=768, n_levels=3, deformable_attention=False)
    # x = [torch.ones(2, 256, 768) for _ in range(3)]
    # out_x = block(x)

    # block = VisualPerceptionModule(dim=768, n_alpha=768, n_beta=768)
    #
    # out_x = block([torch.ones(2, 768), torch.ones(2, 256, 768), torch.ones(2, 256, 768)])

    block = MSW(dim=768, n_levels=4)
    out_x = block(torch.ones(2, 256, 768))
    print(out_x.shape)
