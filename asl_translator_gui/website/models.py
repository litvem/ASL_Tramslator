from django.db import models
from django.conf import settings
import datetime
import os
from .validations import *

# User translation input
class Translation_input(models.Model):
    input_id = models.AutoField(primary_key=True)
    input_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True)
    input_date = models.DateField(default=datetime.datetime.today)
    input_file = models.FileField(upload_to="input/", null=True, validators=[validate_mp4, validate_file_size])

    def __str__(self):
        return f'Translation_input - ID: {self.input_id}, User: {self.input_user.username}'
    
    def file_name(self):
        return os.path.basename(self.input_file.name)
    
    
# User translation output
class Translation_output(models.Model):
    output_id = models.AutoField(primary_key=True)
    output_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True)
    output_source = models.ForeignKey(Translation_input, on_delete=models.CASCADE)
    output_file = models.FileField(upload_to="output/", null=True)

    def __str__(self):
        return f'Translation_input - ID: {self.output_id}, User: {self.output_user.username}'
    
    def file_name(self):
        return os.path.basename(self.output_file.name)
    

# Training input
class Training_input(models.Model):
    tr_input_id = models.AutoField(primary_key=True)
    tr_input_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    tr_input_file = models.FileField(upload_to="input/", null=True, default={}, validators=[validate_json])

    def __str__(self):
        return f'Training_input - ID: {self.tr_input_id}, User: {self.tr_input_user}'


# Training
class Training(models.Model):
    model_id = models.AutoField(primary_key=True)
    tr_input_file = models.ForeignKey(Training_input, on_delete=models.CASCADE, blank=True, null=True)
    training_date = models.DateField(default=datetime.datetime.today)
    training_accuracy = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    testing_accuracy = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    model_weights = models.FileField(upload_to="models/", blank=True, null=True)
    is_deployed = models.BooleanField(default=False, blank=True, null=True)
  
    class Meta:
        ordering = ('-training_accuracy', '-testing_accuracy',)

    def __str__(self):
        return str(self.model_id)