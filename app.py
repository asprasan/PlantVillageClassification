from pathlib import Path
from flask import Flask, request, render_template
import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__, static_url_path="/static", static_folder="static")
model = None
transform = None

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'JPG', 'JPEG'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/jpg'}  # Add more if needed, e.g., 'image/png'
model_path = Path('weights')
ort_session = onnxruntime.InferenceSession(
            model_path / "checkpoint.onnx",
            providers=["CPUExecutionProvider"]
)
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_mime_type(mimetype):
    return mimetype in ALLOWED_MIME_TYPES

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum())

def transform_image(image_bytes):
    transform = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]
                                                         )
                                    ])

    image = Image.open(io.BytesIO(image_bytes))
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

@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template("upload_error.html", error='No file selected')
        
        if not allowed_file(file.filename):
            return render_template("upload_error.html", error='Invalid file extension. Only JPG/JPEG allowed.')
        
        if not allowed_mime_type(file.mimetype):
            return render_template("upload_error.html", error='Invalid file type. Only JPEG images allowed.')

        try:
            img = file.read()
            onnx_input = transform_image(img)
            onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_input)}
            prediction = get_prediction(onnxruntime_input)
            return render_template("index.html", result=prediction)
            # return jsonify(prediction)
        except Exception as e:
            return render_template("index.html", result=[{'error': e}])
    else:
        return render_template("index.html", result=[])

if __name__ == "__main__":
    # Flask default port is 5000, but Heroku dynamically assigns a port.
    app.run(debug=True)
