"""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
automobile_df = pd.read_csv('automobile.csv')
reg1_df = pd.read_csv('Reg1.csv')

# 2. Inspeccionar la estructura de los datos
print("Estructura del DataFrame Automobile:\n", automobile_df.info())
print("\nEstructura del DataFrame Reg1:\n", reg1_df.info()) 

# 3. Análisis descriptivo básico 
print("\nEstadísticas descriptivas Automobile:\n", automobile_df.describe(include='all')) 
print("\nEstadísticas descriptivas Reg1:\n", reg1_df.describe()) 

# 4. Visualización inicial (opcional, pero recomendable)
# Ejemplo para Automobile: histograma del precio
sns.histplot(automobile_df['price'])
plt.title('Distribución del Precio de los Automóviles')
plt.show()

# Ejemplo para Reg1: gráfico de dispersión
sns.scatterplot(x='x', y='y', data=reg1_df)
plt.title('Relación entre x e y en Reg1')
plt.show()"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# --- Fase 1: Preparación y Exploración Inicial ---
# 1. Cargar los datos
automobile_df = pd.read_csv('automobile.csv')
reg1_df = pd.read_csv('Reg1.csv')

# 2. Reemplazar "?" por NaN en "automobile_df"
automobile_df.replace('?', pd.NA, inplace=True)

# 3. Eliminar filas con valores faltantes en 'price' (variable objetivo)
automobile_df.dropna(subset=['price'], inplace=True)

# 4. Convertir 'price' a numérico 
automobile_df['price'] = pd.to_numeric(automobile_df['price'], errors='coerce')

# 5. Explorar y seleccionar la variable predictora para 'Automobile'
# Usaremos 'engine-size' como ejemplo por su alta correlación con 'price'
numeric_cols = ['engine-size', 'horsepower', 'city-mpg', 'highway-mpg', 'curb-weight', 'width', 'length', 'wheel-base']  

# Convertir las columnas seleccionadas a numéricas
for col in numeric_cols:
    automobile_df[col] = pd.to_numeric(automobile_df[col], errors='coerce')

# Calcular la correlación con 'price'
correlations = automobile_df[numeric_cols + ['price']].corr()['price'].sort_values(ascending=False)
print("\nCorrelaciones con 'price':\n", correlations)

# 6. Imputar valores faltantes en 'engine-size' con la media
automobile_df['engine-size'] = automobile_df['engine-size'].fillna(automobile_df['engine-size'].mean())

# 7. Eliminar columnas innecesarias para simplificar 
cols_to_drop = ['normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                'drive-wheels', 'engine-location', 'height', 'curb-weight', 
                'engine-type', 'num-of-cylinders', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 
                'peak-rpm'] 
automobile_df.drop(columns=cols_to_drop, inplace=True)

# 8. Manejar valores faltantes en "Reg1" (si los hay)
reg1_df.dropna(inplace=True) 

# --- Análisis descriptivo básico ---
print("\nEstructura del DataFrame Automobile:\n", automobile_df.info())
print("\nEstadísticas descriptivas Automobile:\n", automobile_df.describe()) 
print("\nEstructura del DataFrame Reg1:\n", reg1_df.info()) 
print("\nEstadísticas descriptivas Reg1:\n", reg1_df.describe()) 

# --- Visualizaciones iniciales ---
plt.figure(figsize=(8, 6)) 
sns.histplot(automobile_df['price'])
plt.title('Distribución del Precio de los Automóviles')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', data=reg1_df)
plt.title('Relación entre x e y en Reg1')
plt.xlabel('Variable x')
plt.ylabel('Variable y')
plt.show()

# --- Fase 2: Preprocesamiento de Datos ---
# Dividir los datos en conjuntos de entrenamiento y prueba
X_auto = automobile_df[['engine-size']]
y_auto = automobile_df['price']
X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_split(X_auto, y_auto, test_size=0.2, random_state=42)

X_reg1 = reg1_df[['x']]
y_reg1 = reg1_df['y']
X_train_reg1, X_test_reg1, y_train_reg1, y_test_reg1 = train_test_split(X_reg1, y_reg1, test_size=0.2, random_state=42)

# --- Fase 3: Construcción y Evaluación del Modelo ---
# 1. Regresión Lineal
model_lr_auto = LinearRegression()
model_lr_auto.fit(X_train_auto, y_train_auto)

model_lr_reg1 = LinearRegression()
model_lr_reg1.fit(X_train_reg1, y_train_reg1)

# 2. k-NN (usando k=5 como ejemplo)
model_knn_auto = KNeighborsRegressor(n_neighbors=5)
model_knn_auto.fit(X_train_auto, y_train_auto)

model_knn_reg1 = KNeighborsRegressor(n_neighbors=5)
model_knn_reg1.fit(X_train_reg1, y_train_reg1)

# 3. Hacer predicciones
y_pred_lr_auto = model_lr_auto.predict(X_test_auto)
y_pred_knn_auto = model_knn_auto.predict(X_test_auto)

y_pred_lr_reg1 = model_lr_reg1.predict(X_test_reg1)
y_pred_knn_reg1 = model_knn_reg1.predict(X_test_reg1)

# 4. Evaluar el rendimiento
results = {
    'Dataset': ['Automobile', 'Automobile', 'Reg1', 'Reg1'],
    'Modelo': ['Regresión Lineal', 'k-NN', 'Regresión Lineal', 'k-NN'],
    'R²': [r2_score(y_test_auto, y_pred_lr_auto), 
           r2_score(y_test_auto, y_pred_knn_auto),
           r2_score(y_test_reg1, y_pred_lr_reg1),
           r2_score(y_test_reg1, y_pred_knn_reg1)],
    'RMSE': [mean_squared_error(y_test_auto, y_pred_lr_auto, squared=False),
             mean_squared_error(y_test_auto, y_pred_knn_auto, squared=False),
             mean_squared_error(y_test_reg1, y_pred_lr_reg1, squared=False),
             mean_squared_error(y_test_reg1, y_pred_knn_reg1, squared=False)]
}

results_df = pd.DataFrame(results)
print("\nResultados:\n", results_df)

# 5. Guardar los resultados en un CSV
results_df.to_csv('resultados_modelos.csv', index=False)

# --- Fase 4: Interpretación y Visualización ---
print("\n----- Regresión Lineal: Interpretación de Coeficientes -----")
print("\nAutomobile:")
print(f"Intersección (b): {model_lr_auto.intercept_}")
print(f"Pendiente (m): {model_lr_auto.coef_[0]}")

print("\nReg1:")
print(f"Intersección (b): {model_lr_reg1.intercept_}")
print(f"Pendiente (m): {model_lr_reg1.coef_[0]}")

# --- Visualizar los resultados ---
# --- Automobile ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='engine-size', y='price', data=automobile_df)
plt.plot(X_test_auto, y_pred_lr_auto, color='red', label='Regresión Lineal')
plt.plot(X_test_auto, y_pred_knn_auto, color='green', label='k-NN')
plt.title('Automobile: Comparación de Modelos')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

# --- Reg1 ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', data=reg1_df)
plt.plot(X_test_reg1, y_pred_lr_reg1, color='red', label='Regresión Lineal')
plt.plot(X_test_reg1, y_pred_knn_reg1, color='green', label='k-NN')
plt.title('Reg1: Comparación de Modelos')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# --- Fase 5: Predicción e Intervalos de Confianza ---

# --- Automobile ---
nuevo_engine_size = 150 
precio_predicho = model_lr_auto.predict([[nuevo_engine_size]])[0]
n_auto = len(X_train_auto)  
se_auto = mean_squared_error(y_train_auto, model_lr_auto.predict(X_train_auto), squared=False) 
t_value_auto = stats.t.ppf(0.975, n_auto - 2)  
intervalo_auto = t_value_auto * se_auto * (1/n_auto + (nuevo_engine_size - X_train_auto['engine-size'].mean())**2 / ((X_train_auto['engine-size'] - X_train_auto['engine-size'].mean())**2).sum())**(1/2)
valor_real_auto = automobile_df[automobile_df['engine-size'] == nuevo_engine_size]['price'].values[0] if nuevo_engine_size in automobile_df['engine-size'].values else None
esta_dentro_auto = (precio_predicho - intervalo_auto <= valor_real_auto <= precio_predicho + intervalo_auto) if valor_real_auto else None

# --- Reg1 ---
nuevo_x = 0.5  
y_predicho = model_lr_reg1.predict([[nuevo_x]])[0]
n_reg1 = len(X_train_reg1)
se_reg1 = mean_squared_error(y_train_reg1, model_lr_reg1.predict(X_train_reg1), squared=False)
t_value_reg1 = stats.t.ppf(0.975, n_reg1 - 2)  
intervalo_reg1 = t_value_reg1 * se_reg1 * (1/n_reg1 + (nuevo_x - X_train_reg1['x'].mean())**2 / ((X_train_reg1['x'] - X_train_reg1['x'].mean())**2).sum())**(1/2)
valor_real_reg1 = reg1_df[reg1_df['x'] == nuevo_x]['y'].values[0] if nuevo_x in reg1_df['x'].values else None
esta_dentro_reg1 = (y_predicho - intervalo_reg1 <= valor_real_reg1 <= y_predicho + intervalo_reg1) if valor_real_reg1 else None

# --- Crear DataFrame con datos generados ---
datos_generados = pd.DataFrame({
    'Dataset': ['Automobile', 'Reg1'],
    'Variable Predictora': ['engine-size', 'x'],
    'Valor Predictor': [nuevo_engine_size, nuevo_x],
    'Ecuación del Modelo': [
        "price = {:.2f} + {:.2f} * engine-size".format(model_lr_auto.intercept_, model_lr_auto.coef_[0]),
        "y = {:.2f} + {:.2f} * x".format(model_lr_reg1.intercept_, model_lr_reg1.coef_[0])
    ],
    'Valor Predicho': [precio_predicho, y_predicho],
    'Intervalo de Confianza': [
        f"[{precio_predicho - intervalo_auto:.2f}, {precio_predicho + intervalo_auto:.2f}]",
        f"[{y_predicho - intervalo_reg1:.2f}, {y_predicho + intervalo_reg1:.2f}]"
    ],
    'Valor Real': [valor_real_auto, valor_real_reg1],
    '¿Dentro del Intervalo?': [esta_dentro_auto, esta_dentro_reg1],
    'Explicación Coloquial': [
        "Para un auto con motor de {} unidades, el modelo predice un precio de {:.2f}, con un margen de error entre {:.2f} y {:.2f}. El precio real {} dentro de este rango.".format(
            nuevo_engine_size, precio_predicho, precio_predicho - intervalo_auto, precio_predicho + intervalo_auto, "está" if esta_dentro_auto else "no está"
        ),
        "Para un valor de x de {}, el modelo predice un valor de y de {:.2f}, con un margen de error entre {:.2f} y {:.2f}. El valor real {} dentro de este rango.".format(
            nuevo_x, y_predicho, y_predicho - intervalo_reg1, y_predicho + intervalo_reg1, "está" if esta_dentro_reg1 else "no está"
        )
    ]
})

# Guardar los datos generados en un CSV
datos_generados.to_csv('datos_generados.csv', index=False)

print("\n----- Datos Generados -----\n", datos_generados)