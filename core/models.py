from django.db import models
import uuid

def unique_mri_upload(instance, filename):
    ext = filename.split('.')[-1]
    return f'mri/{uuid.uuid4().hex}.{ext}'

class Patient(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other'),
    ]

    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    mri_image = models.ImageField(upload_to=unique_mri_upload)
    mri_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)

    detected = models.CharField(max_length=20, blank=True, null=True, default=None)
    classified = models.CharField(max_length=50, blank=True, null=True, default=None)
    segmented = models.ImageField(upload_to='segmented/', null=True, blank=True)
    tumor_area = models.IntegerField(null=True, blank=True)
    tumor_center_x = models.IntegerField(null=True, blank=True)
    tumor_center_y = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} (#{self.pk})"
