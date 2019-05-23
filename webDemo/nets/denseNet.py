import torch
import torch.nn as nn
from .sub_module import DenseBlock


class denseNet(nn.Module):
    """
    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'kernel_c':1,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28
    }
    """

    def __init__(self, params):
        super(denseNet, self).__init__()
        upscale_factor = params['upscale_factor']

        self.dense1 = DenseBlock(params)
        params['num_channels'] = 65
        self.dense2 = DenseBlock(params)
        params['num_channels'] = 128
        self.dense3 = DenseBlock(params)
        # params['num_channels'] = 128
        # self.dense4 = sm.DenseBlock(params)
        params['num_channels'] = 64
        self.upconv = nn.Conv2d(in_channels=64, out_channels=(64 * upscale_factor * upscale_factor), kernel_size=3,
                                padding=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.convOut = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, input):
        d1 = self.dense1(input)

        d2 = self.dense2(torch.cat((input, d1), dim=1))

        d3 = self.dense3(torch.cat((d1, d2), dim=1))

        #d4 = self.dense3(torch.cat((d2, d3), dim=1))

        out = self.upconv(d3)
        out = self.pixelshuffle(out)
        out = self.convOut(out)

        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class denseNetBW(nn.Module):
    """
    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'kernel_c':1,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28
    }
    """

    def __init__(self, params):
        super(denseNetBW, self).__init__()

        self.dense1 = DenseBlock(params)
        params['num_channels'] = 65
        self.dense2 = DenseBlock(params)
        params['num_channels'] = 128
        self.dense3 = DenseBlock(params)
        # params['num_channels'] = 128
        # self.dense4 = sm.DenseBlock(params)
        params['num_channels'] = 64

        self.convOut = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def forward(self, input):
        d1 = self.dense1(input)

        d2 = self.dense2(torch.cat((input, d1), dim=1))

        d3 = self.dense3(torch.cat((d1, d2), dim=1))

        #d4 = self.dense3(torch.cat((d2, d3), dim=1))

        out = self.convOut(d3)

        return out
