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
    start_date = forms.DateField(widget=forms.widgets.SelectDateWidget(years=[y for y in range(2017, 2020)]))
    end_date = forms.DateField(widget=forms.widgets.SelectDateWidget(years=[y for y in range(2017, 2020)]))
    outliers = forms.IntegerField(initial=300)
    room_filter = forms.CharField(initial='', required=False, widget=forms.TextInput(),help_text="Room name starts with:")
    per_room = forms.BooleanField(initial=True, widget=forms.widgets.CheckboxInput(), required=False, help_text="Check to divide by # of Rooms")



    def __init__(self, *args, init_start_date=datetime.date(2017, 1, 5), init_end_date=datetime.date(2017, 1, 5), **kwargs):
        super(Filter, self).__init__(*args, **kwargs)
        self.fields['start_date'] = forms.DateField(widget=forms.widgets.SelectDateWidget(years=[y for y in range(init_start_date.year,init_end_date.year+1)]),
                                                    initial=init_start_date)
        self.fields['end_date'] = forms.DateField(widget=forms.widgets.SelectDateWidget(years=[y for y in range(init_start_date.year,init_end_date.year+1)]),
                                                  initial=init_end_date)


class ColumnNames(forms.Form): # test not working
    date_Col = forms.CharField(initial='Datum', required=True, widget=forms.TextInput())
    roomName_Col = forms.CharField(initial='Zi', required=True, widget=forms.TextInput())
    callType_Col = forms.CharField(initial='Rufart', required=True, widget=forms.TextInput())
    callTime_Col = forms.CharField(initial='Zeit', required=True, widget=forms.TextInput())
    arrTime_Col = forms.CharField(initial='Quitt', required=True, widget=forms.TextInput())
    leaveTime_Col = forms.CharField(initial='Erl', required=False, widget=forms.TextInput())