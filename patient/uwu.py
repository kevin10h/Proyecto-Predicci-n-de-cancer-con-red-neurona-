import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import random

# 🔄 Limpiar sesiones anteriores
tf.keras.backend.clear_session()

# 🎯 Fijar semillas para reproducibilidad (opcional pero útil)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Cargar datos
df = pd.read_excel("C:\\Users\\kevin\\OneDrive - UNIVERSIDAD NACIONAL DE INGENIERIA\\Desktop\\prediction\\media\\pacientes_vih_reales.xlsx")

# Preparar X e y
y = df["resultado_prediccion"]
X = df.drop(columns=["resultado_prediccion", "porcentaje_confianza"], errors='ignore')
X = pd.get_dummies(X)

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Class weights (más peso para positivos)
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: pesos[0], 1: pesos[1] * 1.5}  # Aumentamos penalización de FN

# Modelo optimizado
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

# EarlyStopping por sensibilidad (recall)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_recall', mode='max',
                                               patience=10, restore_best_weights=True)

# Entrenamiento
modelo.fit(X_train, y_train,
           epochs=150,
           batch_size=64,
           validation_split=0.2,
           class_weight=class_weights,
           callbacks=[early_stop],
           verbose=1)

# Predicción con umbral ajustado (más sensible a positivos)
umbral = 0.49
y_pred_probs = modelo.predict(X_test)
y_pred = (y_pred_probs > umbral).astype(int).reshape(-1)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {round(accuracy * 100, 2)}%")
print("\n📊 Reporte completo:")
print(classification_report(y_test, y_pred, digits=3))

# Matriz de Confusión
matriz = confusion_matrix(y_test, y_pred)
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Real")
plt.title("🔍 Matriz de Confusión (umbral=0.49)")
plt.show()

# Guardado
modelo.save("../media/modelo_vih_optimo_FN.h5")
joblib.dump(scaler, "../media/escalador_vih_optimo_FN.pkl")
with open("../media/columnas_entrenamiento_optimo_FN.txt", "w") as f:
    f.write("\n".join(X.columns))
