from django.db import models
from django.conf import settings
import datetime
import os
from .validations import *

# User translation input
class Translation_input(models.Model):
    input_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    input_date = models.DateField(default=datetime.datetime.today)
    input_file = models.FileField(upload_to="input/", null=True, validators=[validate_file_format, validate_file_size])

    def __str__(self):
        return f'Translation_input - ID: {self.id}, User: {self.input_id.username}'
    
    def file_name(self):
        return os.path.basename(self.input_file.name)
    
    
# User translation output
class Translation_output(models.Model):
    output_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    output_sourse = models.ForeignKey(Translation_input, on_delete=models.CASCADE)
    output_file = models.FileField(upload_to="output/", null=True)

    def __str__(self):
        return f'Translation_input - ID: {self.id}, User: {self.output_id.username}'
    

# Training
class Training(models.Model):
    modelid = models.AutoField(primary_key=True)
    trainingdate = models.DateField(default=datetime.datetime.today)
    training_accuracy = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    testing_accuracy = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    model_weights = models.FileField(upload_to="models/", null=True)
    is_deployed = models.BooleanField(default=False)

    class Meta:
        ordering = ('-training_accuracy', '-testing_accuracy',)

    def __str__(self):
        return str(self.modelid)
