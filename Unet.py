import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels, size, pos_embed=16):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels + pos_embed, 4, batch_first=True)
        self.pos = nn.Parameter(torch.randn(1, size * size, pos_embed))
        self.l1 = nn.Linear(channels + pos_embed, channels)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        x_ln = torch.cat([x_ln, self.pos.repeat(x.shape[0], 1, 1)], dim=-1)

        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = self.l1(attention_value)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, pool_kernal=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_kernal),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(size=size, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=1, embed_dim=272):
        super().__init__()

        # 64 * 800 * 800
        self.inc = DoubleConv(c_in, 32)

        # 128 * 200 * 200
        self.down1 = Down(32, 64, embed_dim, 4)

        # 256 * 50 * 50
        self.down2 = Down(64, 128, embed_dim, 4)
        # effective patch size 16 * 16
        self.sa1 = SelfAttention(128, 50)

        # 256 * 25 * 25
        self.down3 = Down(128, 256, embed_dim)
        # effective patch size 32 * 32
        self.sa2 = SelfAttention(256, 25)

        # 256 * 12 * 12
        self.down4 = Down(256, 256, embed_dim)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # 256 * 25 * 25
        self.up1 = Up(512, 128, (25, 25), embed_dim)
        self.sa4 = SelfAttention(128, 25)

        # 128*50*50
        self.up2 = Up(256, 64, (50, 50), embed_dim)
        self.sa5 = SelfAttention(64, 50)

        # 64 * 200 * 200
        self.up3 = Up(128, 32, (200, 200), embed_dim)

        # 32 * 800 * 800
        self.up4 = Up(64, 64, (800, 800), embed_dim)

        # 1 * 800 * 800
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x, t):

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa3(x5)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x4, t)
        x = self.sa4(x)
        x = self.up2(x, x3, t)
        x = self.sa5(x)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        output = self.outc(x)
        output = torch.sigmoid(output).squeeze(1)
        return output


if __name__ == "__main__":
    # net = UNet(device="cpu")
    net = UNet(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
