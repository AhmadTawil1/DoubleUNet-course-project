import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class Conv2D(nn.Module):
    """
    Custom 2D Convolution block with Batch Normalization and ReLU activation
    Args:
        in_c: Number of input channels
        out_c: Number of output channels
        kernel_size: Size of the convolutional kernel (default: 3)
        padding: Padding size (default: 1)
        dilation: Dilation rate (default: 1)
        bias: Whether to use bias in convolution (default: False)
        act: Whether to apply ReLU activation (default: True)
    """
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        # Define the convolution block with BatchNorm
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class squeeze_excitation_block(nn.Module):
    """
    Squeeze and Excitation block for channel-wise attention
    Args:
        in_channels: Number of input channels
        ratio: Reduction ratio for the bottleneck (default: 8)
    """
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        # Global average pooling to get channel-wise statistics
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Two fully connected layers with ReLU and Sigmoid activation
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        # Squeeze operation
        y = self.avgpool(x).view(batch_size, channel_size)
        # Excitation operation
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        # Scale the input features
        return x*y.expand_as(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module
    Args:
        in_c: Number of input channels
        out_c: Number of output channels
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        # Global average pooling branch
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )

        # Multiple parallel atrous convolutions with different dilation rates
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)

        # Final 1x1 convolution to combine all branches
        self.c5 = Conv2D(out_c*5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        # Process through each branch
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        # Concatenate all branches
        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)

        return y

class conv_block(nn.Module):
    """
    Basic convolution block with two Conv2D layers and SE attention
    Args:
        in_c: Number of input channels
        out_c: Number of output channels
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a1(x)
        return x

class encoder1(nn.Module):
    """
    First encoder using VGG19 backbone
    Extracts features at different scales using pre-trained VGG19 layers
    """
    def __init__(self):
        super().__init__()

        network = vgg19(pretrained=True)

        # Split VGG19 into different feature extraction stages
        self.x1 = network.features[:4]    # First stage
        self.x2 = network.features[4:9]   # Second stage
        self.x3 = network.features[9:18]  # Third stage
        self.x4 = network.features[18:27] # Fourth stage
        self.x5 = network.features[27:36] # Fifth stage

    def forward(self, x):
        x0 = x
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1]

class decoder1(nn.Module):
    """
    First decoder for the first U-Net
    Upsamples features and combines with skip connections
    """
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Define convolution blocks for each upsampling stage
        self.c1 = conv_block(64+512, 256)
        self.c2 = conv_block(512, 128)
        self.c3 = conv_block(256, 64)
        self.c4 = conv_block(128, 32)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        # Upsampling and skip connection concatenation
        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], axis=1)
        x = self.c4(x)

        return x

class encoder2(nn.Module):
    """
    Second encoder for the second U-Net
    Uses custom convolution blocks instead of VGG19
    """
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        # Define convolution blocks for each stage
        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        # Feature extraction with max pooling
        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]

class decoder2(nn.Module):
    """
    Second decoder for the second U-Net
    Combines features from both encoders
    """
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Define convolution blocks for each upsampling stage
        self.c1 = conv_block(832, 256)
        self.c2 = conv_block(640, 128)
        self.c3 = conv_block(320, 64)
        self.c4 = conv_block(160, 32)

    def forward(self, x, skip1, skip2):
        # Upsampling and concatenation with skip connections from both encoders
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x

class build_doubleunet(nn.Module):
    """
    Complete Double U-Net architecture
    Combines two U-Nets in series with attention mechanism
    """
    def __init__(self):
        super().__init__()

        # First U-Net components
        self.e1 = encoder1()
        self.a1 = ASPP(512, 64)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Second U-Net components
        self.e2 = encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        # First U-Net forward pass
        x0 = x
        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        # Apply attention from first U-Net
        input_x = x0 * self.sigmoid(y1)
        
        # Second U-Net forward pass
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        return y1, y2

# Test code to verify model architecture
if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = build_doubleunet()
    y1, y2 = model(x)
    print(y1.shape, y2.shape)
