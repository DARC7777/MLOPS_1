FROM python:3.11-slim
WORKDIR /app

# Instalamos dependencias de sistema necesarias para algunas librerías de ML
#RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 3. Instalamos las librerías necesarias
RUN pip install fastapi uvicorn scikit-learn joblib numpy python-multipart Pillow gradio

# Copiamos TODOS tus archivos (incluyendo el .pkl)
COPY . .

# Exponemos el puerto y lanzamos la app
CMD ["python", "app_gradio.py"]