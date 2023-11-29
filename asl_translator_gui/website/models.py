from django.db import models
from django.conf import settings
import datetime

# User translation history
class Translation(models.Model):
    translation_id = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    translation_name = models.CharField(max_length=100)
    translation_date = models.DateField(default=datetime.datetime.today)
    translation_file = models.FileField(upload_to="output/", null=True)

    def __str__(self):
        return self.translation_name

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
