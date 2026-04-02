from pathlib import Path
import io
import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

model = None
transform = None
model_path = Path('weights')
ort_session = onnxruntime.InferenceSession(
            model_path / "checkpoint.onnx",
            providers=["CPUExecutionProvider"]
)


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum())

def transform_image(image):
    transform = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]
                                                         )
                                    ])
    tensor = transform(image).unsqueeze(0)
    onnx_input = [tensor.numpy(force=True)]
    return onnx_input

def get_prediction(onnxruntime_input):
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
    probs = softmax(onnxruntime_outputs).tolist()[0]  # e.g., [0.7, 0.3] for 2 classes
    
    # Dummy class names
    class_names = ["Diseased", "Healthy"]
    predictions = [
            {"label": class_names[i], "confidence": f"{float(prob):0.4f}"}
            for i, prob in enumerate(probs)
        ]
        
    # Sort by confidence descending
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return predictions

def predict(img:Image.Image)->str:
        try:
            onnx_input = transform_image(img)
            onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_input)}
            predictions = get_prediction(onnxruntime_input)
            result_str = ""
            for prediction in predictions:
                    result_str += f"Image predicted as {prediction['label']} with confidence {prediction['confidence']}\n"
            return result_str

        except Exception as e:
            return e

gradio_interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
         type="pil",
         label="Input image"
         ),
    outputs=gr.Textbox(
         label="Prediction"
    ),
    examples="examples/"
    )

if __name__ == "__main__":
    gradio_interface.launch(
                            server_port=7860)
