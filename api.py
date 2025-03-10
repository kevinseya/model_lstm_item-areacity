import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta

# ========================================================
# 📂 CONFIGURACIÓN DE RUTAS
# ========================================================
MODEL_DIR = r"D:\model_por_item\modelos\modelos validos"
SCALER_FEATS_PATH = r"D:\model_por_item\scaler_ciudad_item_feats_352.pkl"
SCALER_TARGET_PATH = r"D:\model_por_item\scaler_ciudad_item_target_352.pkl"

# ========================================================
# 📥 CARGAR SCALERS
# ========================================================
if os.path.exists(SCALER_FEATS_PATH) and os.path.exists(SCALER_TARGET_PATH):
    scaler_feats = joblib.load(SCALER_FEATS_PATH)
    scaler_target = joblib.load(SCALER_TARGET_PATH)
    print("✅ Scalers cargados correctamente.")
else:
    raise FileNotFoundError("❌ No se encontraron los scalers. Verifica las rutas.")

# ========================================================
# 🚀 CONFIGURACIÓN DE FASTAPI
# ========================================================
app = FastAPI()

# Habilitar CORS para acceso externo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================
# 📤 MODELO DE ENTRADA PARA JSON
# ========================================================
class PredictionRequest(BaseModel):
    item: str
    ciudad: str
    fecha_inicio: str
    fecha_fin: str
    ultimas_semanas: list

# ========================================================
# 📅 FUNCIONES AUXILIARES
# ========================================================
def get_model_path(item, ciudad):
    """Obtiene la ruta del modelo correspondiente al item y la ciudad."""
    model_filename = f"modelo_{item.replace(' ', '_')}_{ciudad.replace(' ', '_')}.keras"
    model_path = os.path.join(MODEL_DIR, model_filename)
    return model_path if os.path.exists(model_path) else None

def get_week_start_dates(fecha_inicio, fecha_fin):
    """Genera las fechas de inicio de semana dentro del rango solicitado."""
    start_date = pd.to_datetime(fecha_inicio)
    end_date = pd.to_datetime(fecha_fin)
    current_date = start_date - timedelta(days=start_date.weekday())  # Obtener lunes anterior
    week_dates = []
    
    while current_date <= end_date:
        week_dates.append(current_date.strftime('%Y-%U'))
        current_date += timedelta(days=7)
    
    return week_dates

# ========================================================
# 🔮 FUNCIÓN DE PREDICCIÓN
# ========================================================
@app.post("/predict")
def predict_sales(request: PredictionRequest):
    try:
        # Validar existencia del modelo para item y ciudad
        model_path = get_model_path(request.item, request.ciudad)
        if not model_path:
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado para {request.item} en {request.ciudad}.")

        # Cargar modelo
        model = load_model(model_path)

        # Validar que haya exactamente 16 semanas de datos
        if len(request.ultimas_semanas) != 16:
            raise HTTPException(status_code=400, detail="Se requieren exactamente 16 semanas de datos previos.")

        # Convertir datos de entrada a DataFrame
        df = pd.DataFrame(request.ultimas_semanas)

        # Validar que las columnas necesarias estén presentes
        required_columns = {"qty_trend", "season", "is_big_city", "sales"}
        if not required_columns.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"Faltan columnas en los datos de entrada: {required_columns - set(df.columns)}")

        # Escalar características de entrada
        input_features = df[["qty_trend", "season", "is_big_city"]].values
        input_features_scaled = scaler_feats.transform(input_features)
        
        # Agregar dimensión para que sea un input válido para LSTM
        input_sequence = np.expand_dims(input_features_scaled, axis=0)

        # Obtener las fechas de predicción
        semanas_prediccion = get_week_start_dates(request.fecha_inicio, request.fecha_fin)
        predicciones = []

        for semana in semanas_prediccion:
            # Hacer predicción
            prediction_scaled = model.predict(input_sequence)[0][0]

            # Invertir transformación del scaler de salida
            predicted_qty = np.expm1(scaler_target.inverse_transform([[prediction_scaled]])[0][0])  # np.expm1() revierte log1p()

            # Guardar predicción
            predicciones.append({
                "semana": semana,
                "ventas_predichas": int(predicted_qty)
            })

            # Preparar nueva fila para la siguiente predicción
            nueva_fila = np.array([[predicted_qty, int(semana.split('-')[1]) % 2, 1]])  # season basado en el mes
            nueva_fila_scaled = scaler_feats.transform(nueva_fila)

            # Agregar nueva fila al input_sequence eliminando la más antigua
            input_sequence = np.append(input_sequence[:, 1:, :], np.expand_dims(nueva_fila_scaled, axis=0), axis=1)

        return {
            "item": request.item,
            "ciudad": request.ciudad,
            "fecha_inicio": request.fecha_inicio,
            "fecha_fin": request.fecha_fin,
            "predicciones_semanales": predicciones
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========================================================
# ▶️ EJECUCIÓN LOCAL
# ========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
