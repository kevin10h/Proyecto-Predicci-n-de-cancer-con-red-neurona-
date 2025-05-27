import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import random

# 1. Reproducibilidad
tf.keras.backend.clear_session()
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 2. Cargar datos
df = pd.read_excel("C:\\Users\\kevin\\OneDrive - UNIVERSIDAD NACIONAL DE INGENIERIA\\Desktop\\prediction\\media\\pacientes_vih_reales.xlsx")

# 3. Preparar X e y
y = df["resultado_prediccion"]
X = df.drop(columns=["resultado_prediccion", "porcentaje_confianza"], errors='ignore')
X = pd.get_dummies(X)

# 4. Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Convertir a formato secuencial (simulado con dimensión temporal=1)
X_scaled_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 6. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_seq, y, test_size=0.2, random_state=seed)

# 7. Balanceo de clases
from sklearn.utils.class_weight import compute_class_weight
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: pesos[0], 1: pesos[1]}

# 8. Modelo híbrido LSTM-GRU
modelo = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1, X_scaled.shape[1])),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 9. EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# 10. Entrenamiento
modelo.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
           class_weight=class_weights, callbacks=[early_stop], verbose=1)

# 11. Evaluación
y_pred_probs = modelo.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# 12. Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred, digits=3))

# 13. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('🔍 Matriz de Confusión')
plt.tight_layout()
plt.show()

# 14. Guardado
modelo.save("../media/modelo_vih_lstm_gru.h5")
joblib.dump(scaler, "../media/escalador_vih_lstm_gru.pkl")
with open("../media/columnas_entrenamiento_lstm_gru.txt", "w") as f:
    f.write("\n".join(X.columns))
