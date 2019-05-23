from django.db import models


class Document(models.Model):
    document = models.FileField(upload_to='')
    uploaded_at = models.DateTimeField(auto_now_add=True)

