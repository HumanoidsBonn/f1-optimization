"""RacelineNet: dilated TCN encoders for a history and a future window, conv + attention fusion, MLP head."""
import torch
import torch.nn as nn


class DilatedResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.res = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.res(x))


def _make_tcn(in_ch: int, out_ch: int, kernel_size: int, dilations) -> nn.Sequential:
    return nn.Sequential(*[
        DilatedResBlock(in_ch if i == 0 else out_ch, out_ch, kernel_size, d) for i, d in enumerate(dilations)
    ])


class RacelineNet(nn.Module):
    def __init__(
        self,
        f1_dim: int = 4,
        f2_dim: int = 3,
        window_size: int = 100,
        future_factor: int = 3,
        target_size: int = 10,
        conv_channels: int = 64,
        kernel_size: int = 5,
        mlp_hidden: int = 128,
        n_heads: int = 4,
        dilations=(1, 2, 4, 8),
    ):
        super().__init__()
        C, W, T = conv_channels, window_size, target_size
        self.target_size = T
        self.hist_conv = _make_tcn(f1_dim, C, kernel_size, dilations)
        self.fut_conv = _make_tcn(f2_dim, C, kernel_size, dilations)
        self.fut_align = nn.AdaptiveAvgPool1d(W)
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(2 * C, C, kernel_size=1),
            nn.BatchNorm1d(C),
            nn.ReLU(),
            nn.Conv1d(C, C, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(C),
            nn.ReLU(),
        )
        self.temporal_attn = nn.MultiheadAttention(embed_dim=C, num_heads=n_heads, batch_first=True)
        self.temporal_ln = nn.LayerNorm(C)
        self.seg_hist = nn.Parameter(torch.zeros(1, 1, C))
        self.seg_fut = nn.Parameter(torch.zeros(1, 1, C))
        self.mlp = nn.Sequential(
            nn.Linear(3 * C * W + f2_dim * T, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, T),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        h_enc = self.hist_conv(past.permute(0, 2, 1))
        f_enc = self.fut_align(self.fut_conv(future.permute(0, 2, 1)))

        x_conv = self.fuse_conv(torch.cat([h_enc, f_enc], dim=1))

        seq = torch.cat([h_enc.permute(0, 2, 1) + self.seg_hist,
                         f_enc.permute(0, 2, 1) + self.seg_fut], dim=1)
        attn_out, _ = self.temporal_attn(seq, seq, seq, need_weights=False)
        x_attn = self.temporal_ln(seq + attn_out).permute(0, 2, 1)

        target_geo = future[:, :self.target_size, :].flatten(1)
        return self.mlp(torch.cat([x_conv.flatten(1), x_attn.flatten(1), target_geo], dim=1))


def load_model(path: str, device: str = "cpu") -> RacelineNet:
    model = RacelineNet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.to(device).eval()
