from django.contrib import admin
from .models import Patient

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('name', 'age', 'gender', 'detected', 'classified', 'created_at')
    search_fields = ('name', 'classified', 'detected')
    list_filter = ('gender', 'detected', 'classified')



