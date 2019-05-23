import torch
import torch.nn as nn

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)

        x = self.doubleConv(x)
        return x

class doubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(doubleConv, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        x = self.main(input)
        return x

class Unet6(nn.Module):
    def __init__(self, nc, nf):
        super(Unet6, self).__init__()

        self.initConv = doubleConv(nc, nf)
        self.doubleConv = doubleConv(nf, nf)

        self.maxPool = nn.MaxPool2d(kernel_size=2)

        self.upConv = up(nf, nf)

        self.outConv = nn.Conv2d(nf, 1, kernel_size=5, padding=2)
        self.tanh = nn.Tanh()


    def forward(self, input):
        conv1 = self.initConv(input)
        conv2 = self.doubleConv(self.maxPool(conv1))
        conv3 = self.doubleConv(self.maxPool(conv2))
        conv4 = self.doubleConv(self.maxPool(conv3))
        conv5 = self.doubleConv(self.maxPool(conv4))

        bottleneck = self.doubleConv(self.maxPool(conv5))

        upconv1 = self.upConv(bottleneck, conv5)
        upconv2 = self.upConv(upconv1, conv4)
        upconv3 = self.upConv(upconv2, conv3)
        upconv4 = self.upConv(upconv3, conv2)
        upconv5 = self.upConv(upconv4, conv1)

        out = self.outConv(upconv5)

        return out

class discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(discriminator, self).__init__()
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 256 x 1600
            nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nf) x 128 x 800
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nf) x 64 x 400
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nf) x 32 x 200
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nf) x 16 x 100
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nf) x 8 x 50
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is (nf) x 4 x 25
            nn.Conv2d(ndf, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
            # state size is (nf) x 2 x 13
        )
        self.fc1 = nn.Linear(ndf * 2 * 2, 256)
        self.LRelu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(256, 1)
        self.out = nn.Sigmoid()

    def forward(self, input):
        out = self.main(input)
        out = self.fc1(out.view(-1, self.ndf * 2 * 2))
        out = self.LRelu(out)
        out = self.fc2(out)
        out = self.out(out)
        return out

class Generator(nn.Module):
    def __init__(self, scale_factor):


        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(scale_factor)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return block8

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x



