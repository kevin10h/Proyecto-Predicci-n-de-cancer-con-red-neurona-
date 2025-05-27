import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from tensorflow.keras.metrics import Recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import random

# 🔄 Limpiar sesiones anteriores
tf.keras.backend.clear_session()

# 🎯 Fijar semillas para reproducibilidad (opcional pero útil)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


# Cargar datos reales
df = pd.read_excel("C:\\Users\\kevin\\OneDrive - UNIVERSIDAD NACIONAL DE INGENIERIA\\Desktop\\prediction\\media\\pacientes_vih_reales.xlsx")


# Preparar X e y
y = df["resultado_prediccion"]
X = df.drop(columns=["resultado_prediccion", "porcentaje_confianza"], errors='ignore')
X = pd.get_dummies(X)

# Calcular la matriz de correlación
correlation_matrix = X.corr()


# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar train y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Calcular pesos de clase para balanceo
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# Ajuste más agresivo de los pesos (por ejemplo, multiplicando manualmente)
class_weights = {0: pesos[0], 1: pesos[1] * 1.5}


# Red neuronal mejorada
#recomendable usar al inico lstm, luego GRU y al final LSTM
# Se recomienda usar LSTM para datos secuenciales, pero aquí se usa una red densa por simplicidad
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),  # más regularización
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy'])

# EarlyStopping para evitar sobreajuste
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True) 

# Entrenamiento
modelo.fit(X_train, y_train,
           epochs=100,
           batch_size=32,
           validation_split=0.2,
           class_weight=class_weights,
           callbacks=[early_stop],
           verbose=1)

# Evaluación
y_pred_probs = modelo.predict(X_test)
# Prueba con un umbral más bajo
threshold = 0.45
y_pred = (y_pred_probs > threshold).astype(int).reshape(-1)


# Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Porcentaje de efectividad (accuracy): {round(accuracy * 100, 2)}%")
print("\n📊 Reporte completo:")
print(classification_report(y_test, y_pred, digits=3))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
# Umbral que maximiza sensibilidad (tpr)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"🔧 Umbral óptimo sugerido: {optimal_threshold:.2f}")

recall = recall_score(y_test, y_pred)
print(f"📌 Recall (sensibilidad): {recall:.3f}")


modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy', Recall(name='recall')])

# Calcular matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.title('🔍 Matriz de Confusión')
plt.tight_layout()
plt.show()



# Guardado
modelo.save("../media/modelo_vih_optimizado.h5")
joblib.dump(scaler, "../media/escalador_vih_optimizado.pkl")
with open("../media/columnas_entrenamiento_optimizado.txt", "w") as f:
    f.write("\n".join(X.columns))
