from __future__ import print_function

import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os

import numpy as np

def superResolve(imgPath, modelPath, resize=None):
    img = Image.open(imgPath).convert('YCbCr')

    basewidth = 300
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    y, cb, cr = img.split()

    model = torch.load(modelPath, map_location='cpu')

    img_to_tensor = ToTensor()

    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    # out_img_y -= np.min(out_img_y)
    # out_img_y /= np.max(out_img_y)
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    bic_img = img.resize(out_img_y.size, Image.BICUBIC).convert('RGB')

    savePath = 'Statistics/static/Statistics'
    out_img.save(os.path.join(savePath, 'tmpSR.png'))
    bic_img.save(os.path.join(savePath, 'tmpLR.png'))

