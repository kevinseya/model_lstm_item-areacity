# ========================================================
# 📥 LIBRERÍAS
# ========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# ========================================================
# 📂 CONFIGURACIÓN DE DIRECTORIOS Y LOGS
# ========================================================
save_dir = r'D:\model_por_item'
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, 'logs_entrenamiento_352.txt')

def log_message(message):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)

# Crear carpetas para almacenar modelos y gráficas
modelos_dir = os.path.join(save_dir, "modelos")
graficas_dir = os.path.join(save_dir, "graficas")
os.makedirs(modelos_dir, exist_ok=True)
os.makedirs(graficas_dir, exist_ok=True)

# ========================================================
# 📊 (1) CARGA Y PREPROCESAMIENTO DE DATOS
# ========================================================
log_message("\n📥 Cargando datos...")

df = pd.read_csv('./data/data_ventas.csv')

if df.empty:
    log_message("❌ ERROR: El archivo CSV está vacío.")
    exit()

df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['year_week'] = df['date'].dt.strftime('%Y-%U')

# 🔹 Cargar combinaciones con más de 1000 registros
data_counts = pd.read_csv('item_ciudad_viables_para_lstm_copy.csv')
productos_ciudades = set(zip(data_counts['itemdescr'], data_counts['areacity']))

# 🔹 Filtrar solo combinaciones con suficientes datos
df = df[df.apply(lambda row: (row['itemdescr'], row['areacity']) in productos_ciudades, axis=1)]

if df.empty:
    log_message("❌ ERROR: No hay datos después del filtrado.")
    exit()

df['is_big_city'] = df['areacity'].apply(lambda x: 1 if x in ['QUITO', 'GUAYAQUIL', 'CUENCA'] else 0)
df['season'] = df['date'].dt.month.apply(lambda x: 1 if x in [1, 2, 3, 4, 5, 10, 11, 12] else 0)

df_weekly = df.groupby(['year_week', 'areacity', 'itemdescr'], as_index=False).agg({
    'qty': 'sum',
    'is_big_city': 'max',
    'season': 'max'
})

# Aplicar winsorize solo si es necesario
if df_weekly['qty'].skew() > 1:  # Solo si hay outliers significativos
    df_weekly['qty'] = winsorize(df_weekly['qty'], limits=[0.01, 0.05])

df_weekly['qty_log'] = np.log1p(df_weekly['qty'])

# Calcular tendencia
df_weekly['qty_trend'] = df_weekly.groupby(['areacity', 'itemdescr'])['qty'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

features = ['qty_trend', 'is_big_city', 'season']
scaler_feats = MinMaxScaler()
df_weekly[features] = scaler_feats.fit_transform(df_weekly[features])
joblib.dump(scaler_feats, os.path.join(save_dir, 'scaler_ciudad_item_feats_352.pkl'))

scaler_target = MinMaxScaler()
df_weekly[['qty_log']] = scaler_target.fit_transform(df_weekly[['qty_log']])
joblib.dump(scaler_target, os.path.join(save_dir, 'scaler_ciudad_item_target_352.pkl'))

df_weekly.drop(columns=['year_week'], inplace=True)

# ========================================================
# 📈 (2) CREACIÓN DE SECUENCIAS PARA LSTM
# ========================================================
ventana = 16  # Aumentar el tamaño de la ventana temporal

def crear_secuencias(df):
    X_feats, y_all = [], []
    
    for _, subset in df.groupby(['areacity', 'itemdescr']):
        if len(subset) < ventana + 1:  # Asegurar que hay suficientes datos para crear secuencias
            continue

        target_vals = subset['qty_log'].values
        feats = subset[features].values  

        for i in range(ventana, len(subset)):
            X_feats.append(feats[i-ventana:i])
            y_all.append(target_vals[i])

    return np.array(X_feats), np.array(y_all)

# ========================================================
# ⚙️ (3) MODELO LSTM MEJORADO
# ========================================================
def construir_modelo_mejorado():
    feats_input = Input(shape=(ventana, len(features)))

    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(feats_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = LSTM(128, return_sequences=True, dropout=0.3, kernel_regularizer=l2(0.02))(x)
    x = LSTM(64, return_sequences=True, dropout=0.3, kernel_regularizer=l2(0.02))(x)
    x = LSTM(32, return_sequences=False, dropout=0.3, kernel_regularizer=l2(0.02))(x)

    fusion = Dense(64, activation='relu', kernel_regularizer=l2(0.02))(x)
    fusion = BatchNormalization()(fusion)
    fusion = Dropout(0.3)(fusion)

    output = Dense(1, activation='relu')(fusion)

    model = Model(inputs=[feats_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae')

    return model

# ========================================================
# 🚀 (4) ENTRENAMIENTO POR ITEM Y CIUDAD
# ========================================================
resultados_metricas = []

def guardar_grafica_loss(history, item, city):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Entrenamiento', color='blue')
    plt.plot(history.history['val_loss'], label='Validación', color='red')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MAE)')
    plt.title(f'Pérdida - {item} en {city}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graficas_dir, f'loss_{item.replace(" ", "_")}_{city.replace(" ", "_")}.png'))
    plt.close()

# Filtrar el DataFrame para incluir solo los ítems y ciudades seleccionados
df_weekly = df_weekly[df_weekly.apply(lambda row: (row['itemdescr'], row['areacity']) in productos_ciudades, axis=1)]

for (item, city), df_item in df_weekly.groupby(['itemdescr', 'areacity']):
    X_item, y_item = crear_secuencias(df_item)

    if len(X_item) == 0:
        log_message(f"❌ No hay suficientes datos para {item} en {city}. Saltando...")
        continue

    model = construir_modelo_mejorado()
    train_size = int(len(X_item) * 0.85)
    X_train, X_val = X_item[:train_size], X_item[train_size:]
    y_train, y_val = y_item[:train_size], y_item[train_size:]

    # Callback para guardar el mejor modelo
    checkpoint = ModelCheckpoint(
        os.path.join(modelos_dir, f'modelo_{item.replace(" ", "_")}_{city.replace(" ", "_")}.keras'),
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    history = model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            checkpoint
        ]
    )

    guardar_grafica_loss(history, item, city)

    y_pred = model.predict(X_val).flatten()
    y_real = np.expm1(y_val)

    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
    wmape = np.sum(np.abs(y_real - y_pred)) / np.sum(y_real) * 100
    precision = 100 - wmape

    resultados_metricas.append([item, city, round(rmse,3), round(mae,3), round(wmape,3), round(precision,3)])

df_metricas = pd.DataFrame(resultados_metricas, columns=["Item", "Ciudad", "RMSE", "MAE", "wMAPE", "Precisión"])
df_metricas.to_csv(os.path.join(save_dir, "metricas_modelos_352.csv"), index=False)

log_message("\n✅ Evaluación completada y métricas guardadas.")