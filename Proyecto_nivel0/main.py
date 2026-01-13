from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import io
from PIL import Image, ImageOps, ImageEnhance # Añadimos ImageEnhance

app = FastAPI()
modelo = joblib.load('mnist_model.pkl')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 1. Abrir y convertir a escala de grises
    image = Image.open(io.BytesIO(contents)).convert('L')
    
    # 2. Aumentar el contraste (ayuda a separar el trazo del fondo)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # 3. Invertir colores (Fondo negro, número blanco)
    image = ImageOps.invert(image)
    
    # 4. LIMPIEZA: Aplicar un umbral (Threshold)
    # Píxeles menores a 128 se vuelven 0 (negro), mayores se vuelven 255 (blanco)
    image = image.point(lambda x: 0 if x < 128 else 255)
    
    # 5. Redimensionar a 28x28
    image = image.resize((28, 28))
    
    # 6. Preparar datos para el modelo
    img_array = np.array(image).reshape(1, 784)
    
    # 7. Predicción y Probabilidades
    prediction = modelo.predict(img_array)[0]
    probabilidades = modelo.predict_proba(img_array)
    confianza = np.max(probabilidades) * 100
    
    return {
        "prediccion": int(prediction),
        "confianza": f"{confianza:.2f}%"
    }