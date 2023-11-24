from django.db import models
import datetime
    
# Model training
class Training(models.Model):
    modelid = models.AutoField(primary_key=True)
    trainingdate = models.DateField(default=datetime.datetime.today)
    accuracy = models.DecimalField(max_digits=5, decimal_places=2)
    modelweights = models.FileField(upload_to="media/models/", null=True)
    isdeployed = models.BooleanField(default=False)

    class Meta:
        ordering = ('-accuracy',)

    def __str__(self):
        return str(self.modelid)
