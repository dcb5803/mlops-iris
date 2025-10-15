import gradio as gr
from app.predict import predict

def classify(sepal_length, sepal_width, petal_length, petal_width):
    sample = [sepal_length, sepal_width, petal_length, petal_width]
    label = predict(sample)
    return f"Predicted class: {label}"

demo = gr.Interface(
    fn=classify,
    inputs=["number", "number", "number", "number"],
    outputs="text",
    title="Iris Classifier"
)

demo.launch()
