import gradio as gr

def hello():
    return "RAG app running!"

gr.Interface(fn=hello, inputs=None, outputs="text").launch()