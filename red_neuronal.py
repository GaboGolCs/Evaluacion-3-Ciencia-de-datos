import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# =====================================
# Cargar datos
# =====================================
df = pd.read_csv("nuevoCSV.csv", sep=";")

# Variable objetivo
y = df["TIPO_CONSULTA"]

# Variables predictoras
X = df[["SUCURSAL", "INSTITUCION", "MATERIA", "SUBMATERIA"]]

# Codificar Y
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# OneHot para variables categóricas
columnas = ["SUCURSAL", "INSTITUCION", "MATERIA", "SUBMATERIA"]

preprocesador = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), columnas)
], remainder="passthrough")

X_encoded = preprocesador.fit_transform(X)

# Escalado
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_encoded)

# =====================================
# División de datos
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

# =====================================
# Crear modelo de red neuronal
# =====================================

num_features = X_train.shape[1]
num_classes = len(encoder_y.classes_)

# No usamos Sigmoide porque:
# - Se satura fácilmente.
# - Es más lenta en entrenar.
# - Está pensada para salidas binarias, no multiclase.
model = Sequential([
    Dense(64, activation="relu", input_shape=(num_features,)),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax") # multiclase y probabilistico
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 40

historial = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_split=0.2
)

# =====================================
# Evaluación
# =====================================
loss, accuracy = model.evaluate(X_test, y_test)
print("\n =========================")
print(f"  Exactitud del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(" =========================\n")

# =====================================
# Gráficos del entrenamiento
# =====================================

plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(historial.history["accuracy"])
plt.plot(historial.history["val_accuracy"])
plt.title("Precisión durante el entrenamiento")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

# Loss
plt.subplot(1, 2, 2)
plt.plot(historial.history["loss"])
plt.plot(historial.history["val_loss"])
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()

# =====================================
# Predicción con un dato nuevo
# =====================================
ejemplo = pd.DataFrame({
    "SUCURSAL": ["SANTIAGO"],
    "INSTITUCION": ["REGISTRO CIVIL"],
    "MATERIA": ["DOCUMENTOS"],
    "SUBMATERIA": ["CEDULA"]
})

ejemplo_encoded = preprocesador.transform(ejemplo)
ejemplo_scaled = scaler.transform(ejemplo_encoded)
prediccion = model.predict(ejemplo_scaled)

indice = prediccion.argmax()
clase_predicha = encoder_y.inverse_transform([indice])[0]

print(f"Predicción final del modelo: {clase_predicha}")
