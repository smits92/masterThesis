from django.shortcuts import render, redirect
import numpy as np
import functions as fun
from myFunctions import mySRGAN as SR
from .forms import DocumentForm, Filter, ColumnNames
from django.conf import settings
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


def super_resolution(request, nr=0):
    model_path = 'data/models/SRGAN4.pth'
    img_path = 'data/tmp/tmp.jpg'

    if request.method == 'POST':
        file_form = DocumentForm(request.POST, request.FILES)
        if file_form.is_valid():
            obj = file_form.save(commit=False)
            initial_path = obj.document.path
            obj.save()
            obj.document.name = '/tmp/tmp.jpg'
            new_path = settings.MEDIA_ROOT + obj.document.name
            os.rename(initial_path, new_path)
            obj.save()

            #SR.superResolve(img_path, model_path)


            return redirect("/imshow")
    #
    # else:
    #     file_form = DocumentForm()
    #
    # if request.method == 'POST' and not file_form.is_valid():
    #     file_form = DocumentForm()
    #
    #     main_df = fun.open_file(csv_path)
    #     df = fun.new_table(main_df, cols, cols2fix)
    #
    #     filter_form = Filter(request.POST)
    #
    #     df = fun.filter_room(df, filter_form['room_filter'].value())
    #     rooms = df.Room.unique()
    #
    #     df = fun.filter_date(df, filter_form['start_date'].value(), filter_form['end_date'].value())
    #
    #     title, xaxis, yaxis = fun.myPlot(df, plot_nr, filter_form['outliers'].value(), per_room=filter_form['per_room'].value())
    #     start_date = df.Date[0]
    #     end_date = df.Date[df.Date.shape[0] - 2]
    #     context = {'plot_nr': plot_nr,
    #                'plot_path': fun.get_plotpath(plot_path, plot_nr),
    #                'file_form': file_form,
    #                'filter': filter_form,
    #                'rooms': rooms,
    #                'start_date': start_date,
    #                'end_date': end_date,
    #                'title' : title,
    #                'xaxis' : xaxis,
    #                'yaxis' : yaxis}
    #     return render(request, 'Statistics/plot.html',context)


    else:
        file_form = DocumentForm()

        context = {
        'file_form': file_form,
        'img_nr' : nr
        }



        return render(request, 'Statistics/plot.html', context)

