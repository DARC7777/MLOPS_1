import gradio as gr
import joblib
import numpy as np
from PIL import Image, ImageOps

# 1. Carga del modelo
modelo = joblib.load('mnist_model.pkl')

def predecir_numero(data):
    # Verificamos que existan capas en el dibujo
    if data is None or 'layers' not in data or not data['layers']:
        return None, "Error: No se detectó dibujo."

    # 2. PROCESAMIENTO ULTRA-ROBUSTO
    # Extraemos la primera capa (donde se dibuja por defecto)
    # Convertimos a escala de grises para ver el trazo
    raw_layer = data['layers'][0]
    
    # El trazo puede estar en el canal alpha (3) o en los colores (0,1,2)
    # Sumamos los canales para detectar cualquier píxel que no sea negro
    image_array = np.max(raw_layer[:, :, :4], axis=2) 
    
    image = Image.fromarray(image_array.astype(np.uint8)).convert('L')

    # 3. VERIFICACIÓN: ¿Hay contenido?
    if np.max(np.array(image)) == 0:
        return None, "Lienzo vacío o borrado."

    # 4. CENTRADO Y ESCALADO (Vital para RandomForest)
    # Recortamos al borde del número
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        # Añadimos un margen de 20px (padding)
        image = ImageOps.expand(image, border=20, fill=0)

    # Redimensionamos a la cuadrícula 28x28
    image_28x28 = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # BINARIZACIÓN: Forzamos blanco puro sobre negro puro
    # Esto hace que la imagen sea perfectamente visible en la vista previa
    image_final = image_28x28.point(lambda x: 0 if x < 50 else 255)

    # 5. PREDICCIÓN
    img_array = np.array(image_final).reshape(1, 784)
    pred = modelo.predict(img_array)[0]
    probs = modelo.predict_proba(img_array)
    confianza = np.max(probs) * 100

    return image_final, f"Número: {int(pred)} (Confianza: {confianza:.2f}%)"

# 6. INTERFAZ CON REINICIO TOTAL DE ESTADO
with gr.Blocks(title="MNIST Final Robust") as demo:
    gr.Markdown("# MNIST: Reconocimiento Robusto")
    
    with gr.Row():
        with gr.Column():
            # 'layers=True' es necesario para versiones nuevas de Gradio
            dibujo = gr.Sketchpad(label="Dibuja aquí", type="numpy", layers=True)
            with gr.Row():
                btn_clear = gr.Button("Eliminar Pizarra", variant="stop")
                btn_run = gr.Button("Predecir", variant="primary")
        
        with gr.Column():
            vista_previa = gr.Image(label="Lo que la IA ve (28x28)", width=200, height=200)
            resultado = gr.Label(label="Resultado")

    # Acciones
    btn_run.click(fn=predecir_numero, inputs=dibujo, outputs=[vista_previa, resultado])
    
    # Reinicio completo: se borran los 3 componentes
    btn_clear.click(
        fn=lambda: (None, None, None), 
        inputs=None, 
        outputs=[dibujo, vista_previa, resultado]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)