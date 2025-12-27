# 1. Usamos una imagen de Python oficial y ligera
FROM python:3.11-slim

# 2. Creamos una carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalamos las librerías necesarias directamente
RUN pip install fastapi uvicorn

# 4. Copiamos tu archivo al contenedor
#COPY main.py .
COPY . .

# 5. Comando para arrancar la API
# Usamos 0.0.0.0 para que sea visible desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]