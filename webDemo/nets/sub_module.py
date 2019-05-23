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

# List of APIs
import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    '''
    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28
    }

    '''

    def __init__(self, params):
        super(DenseBlock, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(params['num_channels'] + params['num_filters'] +
                             params['num_filters'])

        self.conv1 = ConvBlock(params, params['num_channels'], padding_h, padding_w, params['kernel_h'], params['kernel_w'])

        self.conv2 = ConvBlock(params, conv1_out_size, padding_h, padding_w, params['kernel_h'], params['kernel_w'])

        self.conv3 = ConvBlock(params, conv2_out_size, 0, 0, 1, 1)

        self.batchnorm1 = nn.InstanceNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.InstanceNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.InstanceNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()

    def forward(self, input):
        o1 = self.batchnorm1(input)
        o2 = self.prelu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.prelu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input, o3, o7), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.prelu(o9)
        out = self.conv3(o10)
        return out


class ConvBlock(nn.Module):
    def __init__(self, params, channels, padding_h, padding_w, kernel_h, kernel_w):
        super(ConvBlock, self).__init__()
        self.params = params
        self.channels = channels
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        if self.params['padding'] == 'zero':
            self.conv_block = nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.params['num_filters'],
                kernel_size=(self.kernel_h, self.kernel_w),
                padding=(self.padding_h, self.padding_w),
                stride=self.params['stride_conv'])
        elif self.params['padding'] == 'reflection':
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d((self.padding_h, self.padding_w, self.padding_h, self.padding_w)),
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.params['num_filters'],
                    kernel_size=(self.kernel_h, self.kernel_w),
                    stride=self.params['stride_conv'])
            )
        else:
            raise Exception('Invalid Padding Argument!')

    def forward(self, input):
        return self.conv_block(input)


class EncoderBlock(DenseBlock):
    def __init__(self, params):
        super(EncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'],
            stride=params['stride_pool'],
            return_indices=True)

    def forward(self, input):
        out_block = super(EncoderBlock, self).forward(input)
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class EncoderBlockConv(DenseBlock):
    def __init__(self, params):
        super(EncoderBlockConv, self).__init__(params)
        self.params = params

        self.channels = params['num_filters']
        self.kernel_h = params['kernel_h']
        self.kernel_w = params['kernel_w']
        self.padding_h = int((self.kernel_h - 1) / 2)
        self.padding_w = int((self.kernel_w - 1) / 2)
        if self.params['padding'] == 'zero':
            self.conv = nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=(self.kernel_h, self.kernel_w),
                padding=(self.padding_h, self.padding_w),
                stride=self.params['stride_pool'])
        elif self.params['padding'] == 'reflection':
            self.conv = nn.Sequential(
                nn.ReflectionPad2d((self.padding_h, self.padding_w, self.padding_h, self.padding_w)),
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=(self.kernel_h, self.kernel_w),
                    stride=self.params['stride_pool'])
            )
        else:
            raise Exception('Invalid Padding Argument!')

    def forward(self, input):
        out_block = super(EncoderBlockConv, self).forward(input)
        out_encoder = self.conv(out_block)
        return out_encoder, out_block


class DecoderBlock(DenseBlock):
    def __init__(self, params):
        super(DecoderBlock, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block, indices):
        unpool = self.unpool(input, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)

        return out_block


class DecoderBlockConv(DenseBlock):
    def __init__(self, params):
        super(DecoderBlockConv, self).__init__(params)
        self.params = params

        self.channels = params['num_filters']
        self.kernel_h = params['kernel_h']
        self.kernel_w = params['kernel_w']
        self.padding_h = int((self.kernel_h - 1) / 2)
        self.padding_w = int((self.kernel_w - 1) / 2)

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(self.kernel_h, self.kernel_w),
            padding=(self.padding_h, self.padding_w),
            output_padding=1,
            stride=self.params['stride_pool'])

    def forward(self, input, out_block):
        unpool = self.conv_transpose(input)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlockConv, self).forward(concat)

        return out_block


class ClassifierBlock(nn.Module):
    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_classes'],
                              params['kernel_c'], params['stride_conv'])
        # self.softmax = nn.Softmax2d()

    def forward(self, input):
        out_conv = self.conv(input)
        # out_logit = self.softmax(out_conv)
        return out_conv

