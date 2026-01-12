from django.urls import path
from . import views

urlpatterns = [
    path('',views.home_view, name='home'),
    path('register/',views.register_patient, name='register'),
    path('detect/<int:pk>/',views.detection_view, name='detect'),
    path('segment/<int:pk>/', views.segment_view, name='segment'),
    path('research/',views.research_dashboard,name='research'),
    path('batch-upload/', views.batch_upload_view, name='batch_upload'),
    path('batch-dashboard/', views.batch_dashboard_view, name='batch_dashboard'),



]
