from django.db import models


# Create your models here.
class Users(models.Model):
    username = models.CharField(max_length=6, primary_key=True)
    password = models.IntegerField(null=False)
    mail = models.EmailField(max_length=30, null=False)

    def __str__(self):
        return self.username
