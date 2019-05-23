from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _
import datetime #for checking renewal date range.
from .models import Document
import datetime


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('document',)


class Filter(forms.Form):

    frame = forms.IntegerField(initial=0)




class ColumnNames(forms.Form): # test not working
    date_Col = forms.CharField(initial='Datum', required=True, widget=forms.TextInput())
    roomName_Col = forms.CharField(initial='Zi', required=True, widget=forms.TextInput())
    callType_Col = forms.CharField(initial='Rufart', required=True, widget=forms.TextInput())
    callTime_Col = forms.CharField(initial='Zeit', required=True, widget=forms.TextInput())
    arrTime_Col = forms.CharField(initial='Quitt', required=True, widget=forms.TextInput())
    leaveTime_Col = forms.CharField(initial='Erl', required=False, widget=forms.TextInput())