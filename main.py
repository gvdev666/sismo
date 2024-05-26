import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import re

# Función para convertir coordenadas
def convert_coordinates(coord):
    match = re.match(r"(\d+\.?\d*)°?\s*([NSEW])", coord)
    if not match:
        raise ValueError(f"Invalid coordinate format: {coord}")
    value, direction = match.groups()
    value = float(value)
    if direction in ('S', 'W'):
        value = -value
    return value

# Función para convertir comas a puntos en valores numéricos
def convert_comma_to_dot(value):
    return float(value.replace(',', '.'))

# Cargar los datos
try:
    df = pd.read_excel('Datos_sismicos.xlsx')
except FileNotFoundError:
    raise FileNotFoundError("El archivo 'Datos_sismicos.xlsx' no se encontró.")
except ImportError:
    raise ImportError("Necesitas instalar la librería 'openpyxl'. Usa el comando 'pip install openpyxl'.")

# Eliminar filas con valores nulos en las columnas de coordenadas y características
df.dropna(subset=['Epicenter Latitude', 'Epicenter Longitude', 'Sample BHE', 'Sample BHN', 'Sample BHZ'], inplace=True)

# Convertir las coordenadas de los epicentros
df['Epicenter Latitude'] = df['Epicenter Latitude'].astype(str).apply(convert_coordinates)
df['Epicenter Longitude'] = df['Epicenter Longitude'].astype(str).apply(convert_coordinates)

# Convertir comas a puntos en las muestras
df['Sample BHE'] = df['Sample BHE'].astype(str).apply(convert_comma_to_dot)
df['Sample BHN'] = df['Sample BHN'].astype(str).apply(convert_comma_to_dot)
df['Sample BHZ'] = df['Sample BHZ'].astype(str).apply(convert_comma_to_dot)

# Características y objetivo
X = df[['Sample BHE', 'Sample BHN', 'Sample BHZ']]
y_latitude = df['Epicenter Latitude']
y_longitude = df['Epicenter Longitude']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train_lat, y_test_lat, y_train_lon, y_test_lon = train_test_split(X, y_latitude, y_longitude, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelos de regresión para la latitud
svr_lat = SVR()
svr_lat.fit(X_train_scaled, y_train_lat)
rf_lat = RandomForestRegressor()
rf_lat.fit(X_train_scaled, y_train_lat)

# Entrenar modelos de regresión para la longitud
svr_lon = SVR()
svr_lon.fit(X_train_scaled, y_train_lon)
rf_lon = RandomForestRegressor()
rf_lon.fit(X_train_scaled, y_train_lon)

# Evaluar los modelos para la latitud
y_pred_lat_svr = svr_lat.predict(X_test_scaled)
y_pred_lat_rf = rf_lat.predict(X_test_scaled)
mse_lat_svr = mean_squared_error(y_test_lat, y_pred_lat_svr)
mse_lat_rf = mean_squared_error(y_test_lat, y_pred_lat_rf)

# Evaluar los modelos para la longitud
y_pred_lon_svr = svr_lon.predict(X_test_scaled)
y_pred_lon_rf = rf_lon.predict(X_test_scaled)
mse_lon_svr = mean_squared_error(y_test_lon, y_pred_lon_svr)
mse_lon_rf = mean_squared_error(y_test_lon, y_pred_lon_rf)

print(f'MSE Latitud SVR: {mse_lat_svr}')
print(f'MSE Latitud Random Forest: {mse_lat_rf}')
print(f'MSE Longitud SVR: {mse_lon_svr}')
print(f'MSE Longitud Random Forest: {mse_lon_rf}')

# Predicción de nuevos datos (Ejemplo)
new_data = pd.DataFrame({'Sample BHE': [29], 'Sample BHN': [1616], 'Sample BHZ': [-1353]})
new_data_scaled = scaler.transform(new_data)
lat_pred = svr_lat.predict(new_data_scaled)
lon_pred = svr_lon.predict(new_data_scaled)
print(f'Predicción de la latitud: {lat_pred[0]}')
print(f'Predicción de la longitud: {lon_pred[0]}')

# Graficar los resultados
plt.figure(figsize=(12, 6))

# Latitud
plt.subplot(1, 2, 1)
plt.scatter(y_test_lat, y_pred_lat_svr, label='SVR', alpha=0.5)
plt.scatter(y_test_lat, y_pred_lat_rf, label='Random Forest', alpha=0.5)
plt.plot([y_test_lat.min(), y_test_lat.max()], [y_test_lat.min(), y_test_lat.max()], 'k--', lw=2)
plt.xlabel('Latitud Real')
plt.ylabel('Latitud Predicha')
plt.title('Predicción de la Latitud del Epicentro')
plt.legend()

# Longitud
plt.subplot(1, 2, 2)
plt.scatter(y_test_lon, y_pred_lon_svr, label='SVR', alpha=0.5)
plt.scatter(y_test_lon, y_pred_lon_rf, label='Random Forest', alpha=0.5)
plt.plot([y_test_lon.min(), y_test_lon.max()], [y_test_lon.min(), y_test_lon.max()], 'k--', lw=2)
plt.xlabel('Longitud Real')
plt.ylabel('Longitud Predicha')
plt.title('Predicción de la Longitud del Epicentro')
plt.legend()

plt.tight_layout()
plt.show()
