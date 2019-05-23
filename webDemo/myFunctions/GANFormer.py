import torch
import numpy as np
from nets.tiny_quicknat import TinyQuickNAT
import time
import torch.jit as jt


class GANFormer(object):
    def __init__(self, param, device):
        self.localDevice = torch.device(device)
        self.model = TinyQuickNAT(param, convolutional_downsampling=False, convolutional_upsampling=False).to(self.localDevice)

    def loadWeights(self, path):  
        if 1:
            checkpoint = torch.load(path, map_location='cpu')
        else:
            checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['gen_state_dict'])
        
    def run(self, input):
        print('Beamforming with GANFormer...')
        start = time.time()
        input = input.transpose((2, 0, 1))
        input = np.expand_dims(input, 0)
        mean = np.mean(input)
        stddev = np.std(input)
        input = (input - mean) / stddev
        input = (input - np.min(input)) / (np.max(input) - np.min(input))
        inputTensor = torch.from_numpy(input).float().to(self.localDevice)

        self.model.eval()

        with torch.no_grad():
            output = self.model(inputTensor)
            output = np.squeeze(output.data.cpu().numpy())

        output *= 255

        print('Time per frame: {:.2f}'.format(time.time()-start))
        return np.clip(output, a_min=0, a_max=255)

    def loadJit(self, path):
        self.model = jt.load(path, map_location='cpu')

    def runJit(self, input):
        print('Beamforming with GANFormer...')
        start = time.time()
        input = input.transpose((2, 0, 1))
        input = np.expand_dims(input, 0)
        inputTensor = torch.from_numpy(input).float().to(self.localDevice)

        output = self.model.forward(inputTensor)
        output = np.squeeze(output.data.cpu().numpy())

        output *= 255

        print('Time per frame: {:.2f}'.format(time.time()-start))
        return np.clip(output, a_min=0, a_max=255)
  

    