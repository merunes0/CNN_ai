import gradio as gr
from main import IMG_SIZE
from main import tf
from main import model


def predict_image(img):
    # Предобработка изображения
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, axis=0)
    img = img / 255.0

    # Предсказание
    pred = model.predict(img)[0][0]
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    diagnosis = 'Pneumonia' if pred > 0.5 else 'Normal'

    return f"{diagnosis} (confidence: {confidence:.2%})"


iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(shape=IMG_SIZE),
    outputs="text",
    title="Pneumonia Detection from Chest X-Ray",
    description="Upload a chest X-ray image to classify as Normal or Pneumonia"
)

iface.launch()