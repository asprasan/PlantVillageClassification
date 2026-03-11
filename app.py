from pathlib import Path
import yaml
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

from models import MODEL_REGISTRY, efficientnet

app = Flask(__name__, static_url_path="/static", static_folder="static")
model = None
transform = None

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'JPG', 'JPEG'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_image(image_bytes):
    transform = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]
                                                         )
                                    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        # max returns (value ,index)
    softmax = nn.Softmax(dim=1)
    probs = softmax(outputs).tolist()[0]  # e.g., [0.7, 0.3] for 2 classes
    
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
        try:
            img = request.files['image'].read()
            tensor = transform_image(img)
            prediction = get_prediction(tensor)
            return render_template("index.html", result=prediction)
            # return jsonify(prediction)
        except Exception as e:
            return render_template("index.html", result={'error': e})
    else:
        return render_template("index.html", result={})

if __name__ == "__main__":
    model_path = Path('weights') / "config.yml"
    with open(model_path, 'r') as f:
        config = yaml.safe_load(f)
    config['batch_size'] = 1
    model_class = MODEL_REGISTRY[config['model']]
    model = model_class(config['num_classes'])
    # load model
    checkpoint = model_path.parent / "checkpoint.pth"
    model_state = torch.load(checkpoint, map_location='cpu')['model']
    model.load_state_dict(model_state)
    model.eval()
    # Flask default port is 5000, but Heroku dynamically assigns a port.
    app.run(debug=True)
