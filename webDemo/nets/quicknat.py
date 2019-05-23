# MIT License
#
# Copyright (c) 2018 Abhijit Guha Roy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import deepformer.nets.sub_module as sm


class quickNAT(nn.Module):
    """
    A PyTorch implementation of QuickNAT
    Coded by Abhijit
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
        super(quickNAT, self).__init__()

        self.convolutional_downsampling = convolutional_downsampling
        self.convolutional_upsampling = convolutional_upsampling

        if self.convolutional_downsampling:
            self.encode1 = sm.EncoderBlockConv(params)
            params['num_channels'] = 64
            self.encode2 = sm.EncoderBlockConv(params)
            self.encode3 = sm.EncoderBlockConv(params)
            self.encode4 = sm.EncoderBlockConv(params)
        else:
            self.encode1 = sm.EncoderBlock(params)
            params['num_channels'] = 64
            self.encode2 = sm.EncoderBlock(params)
            self.encode3 = sm.EncoderBlock(params)
            self.encode4 = sm.EncoderBlock(params)

        self.bottleneck = sm.DenseBlock(params)
        params['num_channels'] = 128
        if self.convolutional_upsampling:
            self.decode1 = sm.DecoderBlockConv(params)
            self.decode2 = sm.DecoderBlockConv(params)
            self.decode3 = sm.DecoderBlockConv(params)
            self.decode4 = sm.DecoderBlockConv(params)
        else:
            self.decode1 = sm.DecoderBlock(params)
            self.decode2 = sm.DecoderBlock(params)
            self.decode3 = sm.DecoderBlock(params)
            self.decode4 = sm.DecoderBlock(params)

        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input):
        if self.convolutional_downsampling:
            e1, out1 = self.encode1.forward(input)
            e2, out2 = self.encode2.forward(e1)
            e3, out3 = self.encode3.forward(e2)
            e4, out4 = self.encode4.forward(e3)
        else:
            e1, out1, ind1 = self.encode1.forward(input)
            e2, out2, ind2 = self.encode2.forward(e1)
            e3, out3, ind3 = self.encode3.forward(e2)
            e4, out4, ind4 = self.encode4.forward(e3)

        bn = self.bottleneck.forward(e4)

        if self.convolutional_upsampling:
            d4 = self.decode4.forward(bn, out4)
            d3 = self.decode1.forward(d4, out3)
            d2 = self.decode2.forward(d3, out2)
            d1 = self.decode3.forward(d2, out1)
        else:
            d4 = self.decode4.forward(bn, out4, ind4)
            d3 = self.decode1.forward(d4, out3, ind3)
            d2 = self.decode2.forward(d3, out2, ind2)
            d1 = self.decode3.forward(d2, out1, ind1)
        prob = self.classifier.forward(d1)

        return prob

    def features(self, input):
        if self.convolutional_downsampling:
            e1, out1 = self.encode1.forward(input)
            e2, out2 = self.encode2.forward(e1)
            e3, out3 = self.encode3.forward(e2)
            e4, out4 = self.encode4.forward(e3)
        else:
            e1, out1, ind1 = self.encode1.forward(input)
            e2, out2, ind2 = self.encode2.forward(e1)
            e3, out3, ind3 = self.encode3.forward(e2)
            e4, out4, ind4 = self.encode4.forward(e3)

        bn = self.bottleneck.forward(e4)

        return bn

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
