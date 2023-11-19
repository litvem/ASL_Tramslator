from django.db import models
import datetime

# User model
class Admin(models.Model):
    username = models.CharField(max_length=30)
    password = models.CharField(max_length=30)

    def __str__(self):
        return f'{self.username} {self.password}'