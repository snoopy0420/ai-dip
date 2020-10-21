from django.db import models


class Post(models.Model):
    number = models.IntegerField(
        verbose_name='お仕事No.'
        )

    value = models.FloatField(
        verbose_name="応募数　合計"
    )

    def __str__(self):
        return str(self.number)