import os
import joblib
import pandas as pd
import tensorflow as tf
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import PersonaForm
from .models import Persona

# Carga de modelo y scaler
modelo = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'media', 'modelo_vih_optimizado.h5'))
scaler = joblib.load(os.path.join(settings.BASE_DIR, 'media', 'escalador_vih_optimizado.pkl'))

# Columnas de entrenamiento que usó el modelo
columnas_entrenamiento = [
    'edad', 'sexo_F', 'sexo_M', 'estado_civil_Casado', 'estado_civil_Soltero', 'estado_civil_Divorciado', 'estado_civil_Viudo', 
    'orientacion_sexual_Heterosexual', 'orientacion_sexual_Homosexual', 'orientacion_sexual_Bisexual', 
    'uso_proteccion', 'consumo_drogas', 'n_parejas_sexuales', 'historial_its', 
    'nivel_socioeconomico_Alto', 'nivel_socioeconomico_Bajo', 'nivel_socioeconomico_Medio', 
    'cantidad_tatuajes', 'cantidad_donaciones_dadas', 'cantidad_donaciones_recibidas', 
    
]

# Obtener pesos y bias de la primera neurona
pesos = modelo.layers[0].get_weights()[0]  # shape (n_inputs, n_neuronas)
bias = modelo.layers[0].get_weights()[1]   # shape (n_neuronas,)

# Construir ecuación de la primera neurona
ecuacion = []
for i, col in enumerate(columnas_entrenamiento):
    coef = round(float(pesos[i][0]), 3)  # redondeo exacto
    signo = "+" if coef >= 0 else ""
    ecuacion.append(f"{signo}{coef} × {col}")

bias_0 = round(float(bias[0]), 3)
signo_bias = "+" if bias_0 >= 0 else ""
ecuacion_str = "<br>".join(ecuacion) + f"<br>{signo_bias}{bias_0}"

def prediccion_vih(request):
    if request.method == 'POST':
        form = PersonaForm(request.POST)
        if form.is_valid():
            persona = form.save(commit=False)

            # Transformar los datos a DataFrame
            data = pd.DataFrame([form.cleaned_data])

            # Convertir booleanos
            for col in data.columns:
                if data[col].dtype == bool:
                    data[col] = data[col].astype(int)

            # Obtener variables dummy
            data = pd.get_dummies(data)
            print(scaler.feature_names_in_)

            # Asegurar todas las columnas necesarias
            for col in scaler.feature_names_in_:
                if col not in data.columns:
                    data[col] = 0

            # Ordenar columnas como el entrenamiento
            data = data[scaler.feature_names_in_]

            # Escalar
            print(data.columns)
            input_data = scaler.transform(data)

            # Predicción
            pred = modelo.predict(input_data)[0][0]
            persona.porcentaje_confianza = float(pred)
            persona.resultado_prediccion = pred > 0.5
            persona.save()

            return redirect('resultado_vih', persona_id=persona.id)
    else:
        form = PersonaForm()

    return render(request, 'formulario_vih.html', {'form': form, 'ecuacion': ecuacion_str})


def resultado_vih(request, persona_id):
    persona = Persona.objects.get(id=persona_id)
    return render(request, 'resultado_vih.html', {
        'resultado': 'POSITIVO' if persona.resultado_prediccion else 'NEGATIVO',
        'confianza': round(persona.porcentaje_confianza * 100, 2)
    })
