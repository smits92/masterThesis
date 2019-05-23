import numpy as np
import time
import os
from myFunctions.mhd_reader import MHD_reader
import imageio
from scipy.misc import imresize, imsave

path = '/Users/smits/Dropbox/Master/Tesis/Elemental/logPics/evalData/RD_Data_TF/Left/Logitudinal/Bicep/rawData_1_beamformed.mhd'
MVpath = '/Users/smits/Dropbox/Master/Tesis/Elemental/logPics/evalData/BF_MV_Data_TF/Left/Logitudinal/Bicep/rawData_1_beamformed.mhd'
# path = '/Volumes/clusterstorage/polyaxon/data1/DeepFormerData/MICCAI2019/valid_1/RD_Data_TF/Walter_Simson_26_M/Left/Transverse/Carotid/rawData_1_beamformed.mhd'
# MVpath = '/Volumes/clusterstorage/polyaxon/data1/DeepFormerData/MICCAI2019/valid_1/BF_MV_Data_TF/Walter_Simson_26_M/Left/Transverse/Carotid/rawData_1_beamformed.mhd'
modelPath = '/Users/smits/Dropbox/Master/Tesis/Elemental/Django/data/models/GANFormer.pth'

aspect = 3.8 / 3.2

def npFromMHD(path, frame, raw=True):
    print('...Loading MDH file')
    start = time.time()
    reader = MHD_reader(path)
    shape = reader.get_data_shape()
    if raw:
        imgs = np.zeros((shape[2], 256, 2000, 64))
        #for i in range(reader.number_frames):
        imgs[frame, :,:,:] = reader._read_frame(frame)
        imgs = np.array(imgs)[:,:,100:1600, :]
    else:
        imgs = np.zeros((shape[1],shape[0], shape[2]))
        #for i in range(2):#(reader.number_frames):
        imgs[:, :, frame] = reader._read_frame(frame)[:, :, 0]
        imgs = np.array(imgs)[:, 100:1600, :]

    print('...Loaded after {:.02f} seconds'.format(time.time() - start))

    return imgs


def createGIF(imgs, name, duration=0.5, aspect=1):
    with imageio.get_writer(name + '.gif', duration=duration, mode='I') as writer:

        for i in range(imgs.shape[2]):
            img = imgs[:,:,i]
            img = np.squeeze(np.array(img, dtype='uint8')).T

            img = imresize(img, (int(aspect * 256), 256))
            writer.append_data(img)


def DASFormer(raw):
    das = np.sum(raw, axis=2)
    das = np.log(abs(das)+1)
    das = np.clip(das, a_min=0, a_max=150)

    return das


def getMHDfilename(path='data/raw'):
    files = os.listdir(path)
    mhdFile = None

    for file in files:
        if '.mhd' in file:
            mhdFile = file

    return mhdFile



if __name__ == "__main__":
    volume = npFromMHD(path)


    # from myFunctions.GANFormer import GANFormer
    #
    # param = {
    #     'num_channels': 64,
    #     'num_filters': 64,
    #     'kernel_h': 3,
    #     'kernel_w': 3,
    #     'kernel_c': 1,
    #     'stride_conv': 1,
    #     'pool': 2,
    #     'stride_pool': 2,
    #     'num_classes': 1,
    #     'padding': 'reflection'
    # }
    # model = GANFormer(param, 'cpu')
    #
    # model.loadWeights(modelPath)
    #
    # GANformed = np.zeros((256, 1500, 10))
    #
    # for i in range(2):
    #     GANformed[:,:,i] = model.run(volume[i])
    #
    #
    # MV = npFromMHD(MVpath, raw=False)
    # MV = np.squeeze(MV)[:,:,:10]
    # imsave("Label_long.png", imresize(MV[:,:,1].T, (int(aspect * 256), 256)))
    # imsave("Recon_long.png", imresize(GANformed[:,:,1].T, (int(aspect * 256), 256)))

    # das = DASFormer(volume[0])
    #
    # imsave("DAS.png", imresize(das[:, :].T, (int(aspect * 256), 256)))

    # createGIF(GANformed, 'Rec', duration=0.3, aspect=aspect)
    # createGIF(MV, 'MinVar', duration=0.3, aspect=aspect)




