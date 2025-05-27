import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs, targets, num_boxes=None, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if num_boxes is None:
        num_boxes = targets.shape[0]

    return loss.mean(1).sum() / num_boxes


class SelfAttention(nn.Module):
    def __init__(self, channels, size, pos_embed=16):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels + pos_embed, 4, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, size * size, pos_embed))
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
        attention_value = attention_value  # + x
        attention_value = self.ff_self(attention_value)  # + attention_value
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
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return F.gelu(self.double_conv(x))


class UNet2(nn.Module):
    def __init__(self, c_in=3, c_out=1, embed_dim=272, sigmoid=True):
        super().__init__()
        # self.ini = DoubleConv()

        self.c_in = c_in

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

        # 128 * 50 * 50
        self.up2 = Up(256, 64, (50, 50), embed_dim)
        self.sa5 = SelfAttention(64, 50)

        # 64 * 200 * 200
        self.up3 = Up(128, 32, (200, 200), embed_dim)

        # 32 * 800 * 800
        self.up4 = Up(64, 64, (800, 800), embed_dim)

        # 1 * 800 * 800
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.sigmoid = sigmoid

    def forward(self, x, t):

        x = x.view(-1, self.c_in, 800, 800)

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
        output = self.outc(x).squeeze(1)
        if self.sigmoid:
            output = torch.sigmoid(output)
        return output


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, pool_kernal=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_kernal),
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(emb_dim, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=None):
        super().__init__()

        self.up = nn.Upsample(size=size, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            # DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels),
        )
        if emb_dim is not None:
            self.emb_layer = nn.Sequential(
                nn.Linear(emb_dim, out_channels),
                nn.SiLU(),
            )

    def forward(self, x, skip_x, t=None):
        x = self.up(x)
        # print("up", x.shape, skip_x.shape)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        if t is None:
            return x

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


if __name__ == "__main__":
    # net = UNet(device="cpu")
    net = UNet(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)


class UNet6(nn.Module):
    def __init__(self, c_in=3, c_out=1, embed_dim=272, sigmoid=True):
        super().__init__()
        # self.ini = DoubleConv()

        # 16 * 800 * 800
        self.inc = DoubleConv(c_in, 16)

        # 16 * 400 * 400
        self.down1 = Down(16, 32, embed_dim)

        # 64 * 200 * 200
        self.down2 = Down(32, 64, embed_dim)

        # 128 * 100 * 100
        self.down3 = Down(64, 128, embed_dim)

        # 256 * 50 * 50
        self.down4 = Down(128, 256, embed_dim)

        # 512 * 25 * 25
        self.down5 = Down(256, 512, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 625, 512))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="gelu",
            ),
            num_layers=8,
        )
        # effective patch size 16 * 16
        # self.sa1 = SelfAttention(128, 50)
        # self.sa2 = SelfAttention(128, 50)
        # self.sa3 = SelfAttention(128, 50)
        # self.sa4 = SelfAttention(128, 50)
        # 256 * 25 * 25
        # self.down5 = Down(128, 256, embed_dim)
        # # effective patch size 32 * 32
        # self.sa2 = SelfAttention(256, 25)

        # # 256 * 12 * 12
        # self.down6 = Down(256, 256, embed_dim)
        # self.sa3 = SelfAttention(256, 12)

        # self.bot1 = DoubleConv(256, 512)
        # self.bot2 = DoubleConv(512, 512)
        # self.bot3 = DoubleConv(512, 256)

        # # 256 * 25 * 25
        # self.up1 = Up(512, 128, (25, 25), embed_dim)
        # self.sa4 = SelfAttention(128, 25)

        # # 128 * 50 * 50
        self.up2 = Up(512 + 256, 256, (50, 50), embed_dim)
        # self.sa5 = SelfAttention(128, 50)

        # 128 * 100 * 100
        self.up3 = Up(256 + 128, 64, (100, 100), embed_dim)

        # 64 * 200 * 200
        self.up4 = Up(128, 32, (200, 200), embed_dim)

        # 32 * 400 * 400
        self.up5 = Up(64, 16, (400, 400), embed_dim)

        # 32 * 800 * 800
        self.up6 = Up(32, 32, (800, 800), embed_dim)

        # 1 * 800 * 800
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        self.sigmoid = sigmoid

    def forward(self, x, t):
        # x = x.view(-1, 1, 800, 800)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        x6 = self.down5(x5, t)
        # x5 = self.sa1(x5)
        # print(x5.shape)
        # x5 = self.sa2(x5)
        # x5 = self.sa3(x5)
        # x5 = self.sa4(x5)
        # print(x5.shape)
        # x6 = x5
        x6 = x6.view(-1, 512, 625).transpose(1, 2)
        x6 += self.pos_embed
        # x5 = x5.transpose()
        x = self.transformer(x6)
        x = x.transpose(1, 2).view(-1, 512, 25, 25)
        # x6 = self.down5(x5, t)
        # x6 = self.sa2(x6)
        # x7 = self.down6(x6, t)
        # x7 = self.sa3(x7)

        # x7 = self.bot1(x7)
        # x7 = self.bot2(x7)
        # x7 = self.bot3(x7)

        # x = self.up1(x7, x6, t)
        # x = self.sa4(x)
        # print(x.shape, x5.shape)
        x = self.up2(x, x5, t)
        # print(x.shape)
        # x = self.sa5(x5)
        # print(x.shape)
        # print(x.shape, x4.shape)
        x = self.up3(x, x4, t)
        x = self.up4(x, x3, t)
        x = self.up5(x, x2, t)
        x = self.up6(x, x1, t)
        output = self.outc(x).squeeze(1)
        if self.sigmoid:
            output = torch.sigmoid(output)
        return output


class Easynet(nn.Module):
    def __init__(self, c_in=1):
        super().__init__()
        # self.inc = DoubleConv(c_in, 16)
        # self.dec = DoubleConv(16, 1)
        # self.convs = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 1, 3, padding=1),
        # )
        self.convs = nn.Sequential(
            DoubleConv(1, 16),
            DoubleConv(16, 16),
            DoubleConv(16, 16),
            DoubleConv(16, 16),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        # self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # self.conv2 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x, t):
        x = x.view(-1, 1, 800, 800)
        x = self.convs(x)
        # x = self.conv1(x)
        # x = x.relu()
        # x = self.conv2(x)
        x = x.squeeze(1)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, outdim=16):
        super(CNNEncoder, self).__init__()

        # Convolutional layers to reduce spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 800x800 -> 800x800
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # 800x800 -> 800x800
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 800x800 -> 400x400

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # 400x400 -> 400x400
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 400x400 -> 200x200

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # 200x200 -> 200x200
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200x200 -> 100x100

        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # 100x100 -> 100x100
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100x100 -> 50x50

        # Fully connected layers to reduce to 16 dimensions
        self.fc1 = nn.Linear(16 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, outdim)

    def forward(self, x):
        # x = x.squeeze()
        # x = x.unsqueeze(1)
        x = x.view(-1, 1, 800, 800)

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = F.relu(self.conv4(x))
        x = self.pool3(x)

        x = F.relu(self.conv5(x))
        x = self.pool4(x)

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output 16-dimensional vector

        return x


class tryUnet(L.LightningModule):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.unet = UNet6(embed_dim=embed_dim)
        self.encode = CNNEncoder(embed_dim)

    def forward(self, x, mask):
        t = self.encode(mask)
        res = self.unet(mask, t)
        return res

    def training_step(self, batch):
        pixel_values, _, mask = batch
        pixel_values = pixel_values.to(self.device).float()
        mask = mask.to(self.device).float()
        t = self.forward(pixel_values, mask)
        loss = sigmoid_focal_loss(t, mask, alpha=0.8)
        self.log("training loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch):
        pixel_values, _, mask = batch
        pixel_values = pixel_values.to(self.device).float()
        mask = mask.to(self.device).float()
        t = self.forward(pixel_values, mask)
        loss = sigmoid_focal_loss(t, mask, alpha=0.8)
        self.log("validate loss", loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optim


def sigmoid_focal_loss(
    inputs, targets, num_boxes=None, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if num_boxes is None:
        num_boxes = targets.shape[0]

    return loss.mean(1).sum() / num_boxes


class ConvAdd(nn.Module):
    def __init__(self, c_in, c_out, embed_dim):
        super().__init__()
        self.l = nn.Linear(embed_dim, c_out)
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x, t):
        B, C, H, W = x.shape
        x = self.conv(x)
        t = self.l(t)
        # print(t.shape, x.shape)
        t = t[:, :, None, None].repeat(1, 1, H, W)
        # print(t.shape)

        return F.relu(x + t)


class ViTSegmentation(nn.Module):
    def __init__(
        self,
        c_in=3,
        img_size=800,
        patch_size=32,
        last_layer=64,
        input_embed=256,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
    ):
        """
        Vision Transformer for Image Segmentation.

        Args:
            img_size (int): Input image size (assumes square images).
            patch_size (int): Size of each patch.
            num_classes (int): Number of output segmentation classes.
            embed_dim (int): Dimension of the embedding space.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
        """
        super(ViTSegmentation, self).__init__()

        self.c_in = c_in

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Linear projection of flattened patches

        self.conv1 = ConvAdd(c_in, 8, input_embed)
        self.conv2 = ConvAdd(8, 16, input_embed)
        self.conv3 = ConvAdd(16, last_layer // 2, input_embed)

        # self.inc = nn.Sequential(
        #     nn.Conv2d(c_in, 8, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, last_layer // 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        self.patch_embed = nn.Conv2d(
            last_layer // 2, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.embed = nn.Sequential(nn.Linear(input_embed, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
            ),
            num_layers=num_layers,
        )

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, last_layer // 2, kernel_size=1),
        )

        # Upsampling layer to resize the output to the original image size
        self.upsample = nn.Upsample(
            scale_factor=patch_size, mode="bilinear", align_corners=False
        )

        self.output = nn.Conv2d(last_layer, 1, 3, padding=1)

        # Initialize parameters

    #     self._init_weights()

    # def _init_weights(self):
    #     nn.init.trunc_normal_(self.pos_embed, std=0.02)
    #     # nn.init.trunc_normal_(self.cls_token, std=0.02)
    #     for module in self.segmentation_head:
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.kaiming_normal_(
    #                 module.weight, mode="fan_out", nonlinearity="relu"
    #             )

    def forward(self, x, t):
        """
        Forward pass of the Vision Transformer for segmentation.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Segmentation mask of shape (B, num_classes, H, W).
        """
        x = x.view(-1, self.c_in, 800, 800)
        # x0 = self.inc(x)
        x = self.conv1(x, t)
        x = self.conv2(x, t)
        x0 = self.conv3(x, t)
        B, C, H, W = x.shape
        assert (
            H == self.img_size and W == self.img_size
        ), "Input image size must match model's img_size."

        # Step 1: Patch embedding
        x = self.patch_embed(x0)  # Shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)

        # Step 2: Add class token and positional embedding
        t = self.embed(t).unsqueeze(1)
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: (B, 1, embed_dim)
        x = torch.cat((t, x), dim=1)  # Shape: (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed  # Add positional embedding

        # Step 3: Transformer encoder
        x = self.transformer(x)  # Shape: (B, num_patches + 1, embed_dim)

        # Step 4: Remove class token and reshape to feature map
        x = x[:, 1:, :]  # Remove class token, Shape: (B, num_patches, embed_dim)
        x = x.transpose(1, 2).reshape(
            B,
            self.embed_dim,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
        )

        # Step 5: Segmentation head
        x = self.segmentation_head(
            x
        )  # Shape: (B, num_classes, H/patch_size, W/patch_size)

        # Step 6: Upsample to original image size
        x = self.upsample(x)  # Shape: (B, num_classes, H, W)
        x = torch.cat([x, x0], dim=1)
        x = self.output(x)
        x = x.squeeze(1)
        return x
