from django.db import models
import datetime
    
# Model training
class Training(models.Model):
    model_id = models.AutoField(primary_key=True)
    training_date = models.DateField(default=datetime.datetime.today)
    accuracy = models.DecimalField(default=0, decimal_places=2, max_digits=5)

    def __str__(self):
        return self.model_id