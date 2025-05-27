from django.urls import path
from . import views

urlpatterns = [
    path('', views.prediccion_vih, name='home'),  # Esta línea es nueva
    path('prediccion/', views.prediccion_vih, name='prediccion_vih'),
    path('resultado/<int:persona_id>/', views.resultado_vih, name='resultado_vih'),
]