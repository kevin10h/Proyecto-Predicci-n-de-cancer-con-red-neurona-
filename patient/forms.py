from django import forms
from .models import Persona

class PersonaForm(forms.ModelForm):
    class Meta:
        model = Persona
        exclude = ['resultado_prediccion', 'porcentaje_confianza']