from django.shortcuts import render, redirect
import numpy as np
import functions as fun
import time
from myFunctions.GANFormer import GANFormer
from .forms import DocumentForm, Filter, ColumnNames
from django.conf import settings
from scipy.misc import imresize, imsave
import os


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            initial_path = obj.document.path
            obj.save()
            obj.document.name = '/tmp/tmp.jpg'
            new_path = settings.MEDIA_ROOT + obj.document.name
            os.rename(initial_path, new_path)
            obj.save()
            return redirect("/imshow")
    else:
        form = DocumentForm()
    return render(request, 'Statistics/upload.html', {'form': form})

def run_model_raw(path,
                  modelPath='data/models/GANFormer.pth',
                  frame=0):
    volume = fun.npFromMHD(path, frame, raw=True)

    param = {
        'num_channels': 64,
        'num_filters': 64,
        'kernel_h': 3,
        'kernel_w': 3,
        'kernel_c': 1,
        'stride_conv': 1,
        'pool': 2,
        'stride_pool': 2,
        'num_classes': 1,
        'padding': 'reflection'
    }
    model = GANFormer(param, 'cpu')

    model.loadWeights(modelPath)

    GANformed = model.run(volume[frame])

    DASformed = fun.DASFormer(volume[frame])

    imsave("Statistics/static/Statistics/tmp/ganformed.png",
           imresize(GANformed[:,:].T, (int(3.8 / 3.2 * 256), 256)))
    imsave("Statistics/static/Statistics/tmp/dasformed.png",
           imresize(DASformed[:, :].T, (int(3.8 / 3.2 * 256), 256)))


def super_resolution(request, nr=0):
    raw_path = 'data/raw'
    mhd_file = fun.getMHDfilename(raw_path)

    if request.method == 'POST':
        filter_form = Filter(request.POST)

        run_model_raw(os.path.join(raw_path, mhd_file),frame=int(filter_form['frame'].value()))


        return redirect("/imshow/9")


    else:
        filter_form = Filter()




        context = {
        'file_form': filter_form,
        'img_nr' : nr
        }



        return render(request, 'Statistics/plot.html', context)

