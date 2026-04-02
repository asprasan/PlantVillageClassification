from fastapi import FastAPI
import gradio as gr

from gradio_ui import gradio_interface

app=FastAPI()

app = gr.mount_gradio_app(app, gradio_interface, path="/")
