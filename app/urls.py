from django.urls import path
from .views import UploadView
from .views import move
from .views import export

#app_name = 'app'

urlpatterns = [
    path('', UploadView.as_view(), name='index'),
    path('form_valid', UploadView.form_valid, name ='form_valid'),
    path('/move', move, name='move'),
    path('export', export, name ='export'),
    ]
