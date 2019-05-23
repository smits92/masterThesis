import torch
import torch.nn as nn
import nets.sub_module as sm


class TinyQuickNAT(nn.Module):
    """
    A PyTorch implementation of QuickNAT that is tiny
    Coded by Magda Paschali
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

    def __init__(self, params, convolutional_downsampling, convolutional_upsampling):
        super(TinyQuickNAT, self).__init__()
        self.convolutional_downsampling = convolutional_downsampling
        self.convolutional_upsampling = convolutional_upsampling

        if self.convolutional_downsampling:
            self.encode1 = sm.EncoderBlockConv(params)
        else:
            self.encode1 = sm.EncoderBlock(params)

        params['num_channels'] = 64
        self.bottleneck = sm.DenseBlock(params)

        params['num_channels'] = 128
        if self.convolutional_upsampling:
            self.decode1 = sm.DecoderBlockConv(params)
        else:
            self.decode1 = sm.DecoderBlock(params)

        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input):
        if self.convolutional_downsampling:
            e1, out1 = self.encode1.forward(input)
        else:
            e1, out1, ind1 = self.encode1.forward(input)

        bn = self.bottleneck.forward(e1)

        if self.convolutional_upsampling:
            d1 = self.decode1.forward(bn, out1)
        else:
            d1 = self.decode1.forward(bn, out1, ind1)

        prob = self.classifier.forward(d1)

        if self.training:
            return prob
        else:
            return torch.clamp(prob, min=0, max=1)

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
