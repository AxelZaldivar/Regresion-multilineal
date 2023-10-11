import numpy as np
import matplotlib.pyplot as plt

# Leer datos desde el archivo CSV
with open('Student_Performance.csv', 'r') as file:
    data = [line.strip().split(',') for line in file.readlines()]

# Extraer las cabeceras y los datos
headers = data[0]
data = data[1:]

# Convertir 'Yes' y 'No' a valores numéricos en la columna 'Extracurricular Activities'
activity_mapping = {'Yes': 1, 'No': 0}
activity_col_idx = headers.index('Extracurricular Activities')
for row in data:
    row[activity_col_idx] = activity_mapping[row[activity_col_idx]]

# Convertir los datos a matrices
data = np.array(data, dtype=float)

# Separar características (X) y variable objetivo (y)
X = data[:, [0, 1, activity_col_idx, 3, 4]]
y = data[:, -1]

# Normalizar los datos
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
num_samples = X.shape[0]
num_train = int(0.8 * num_samples)
X_train, X_test = X[:num_train], X[num_train:]
y_train, y_test = y[:num_train], y[num_train:]

# Añadir una columna de unos a la matriz X para representar el término de sesgo
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Inicialización
coefficients = np.zeros(X_train.shape[1])
learning_rate = 0.01
num_iterations = 1000

# Descenso del Gradiente
for _ in range(num_iterations):
    error = X_train @ coefficients - y_train
    gradient = X_train.T @ error
    coefficients -= learning_rate * gradient / num_samples

# Realizar predicciones en el conjunto de prueba
y_pred = X_test @ coefficients

# Evaluar el rendimiento del modelo
error_cuadratico_medio = np.mean((y_test - y_pred) ** 2)

# Graficar las predicciones vs. los valores reales con la línea de regresión
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Horas de estudio')
plt.ylabel('Puntuación proyectada')
plt.title('Predicciones vs. Valores reales con Regresión Multilineal\nError cuadrático medio: ' + str(error_cuadratico_medio))
plt.show()