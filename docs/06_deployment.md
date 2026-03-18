# Model deployment with Flask on Render

In this section, we will deploy our trained model using Flask on Render. Render is a cloud platform that allows you to easily deploy web applications and APIs.

## Step 1: Create a Flask application

First, we need to create a Flask application that will serve our model. Create a new file called `app.py` and add the following code:

```python
# app.py
from flask import Flask, request, render_template

app = Flask(__name__, static_url_path="/static", static_folder="static")

@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template("upload_error.html", error='No file selected')

        if not allowed_mime_type(file.mimetype):
            return render_template("upload_error.html", error='Invalid file type. Only JPEG images allowed.')
```

What we have done here is set up a basic Flask application with a route that accepts POST requests. The route checks if an image file is uploaded and if it is of the correct MIME type (JPEG). If the file is valid, we can proceed to process it with our model.

## Step 2: Load the model

Next, we need to load our trained model. We will use the `onnxruntime` library to load the ONNX model. We choose ONNX over PyTorch as ONNX provides better performance and compatibility across different platforms. Add the following code to `app.py`:

```python

def predict():
    # ... (previous code)

    try:
        img = file.read()
        onnx_input = transform_image(img)
        onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_input)}
        prediction = get_prediction(onnxruntime_input)
        return render_template("index.html", result=prediction)
        # return jsonify(prediction)
    except Exception as e:
        return render_template("index.html", result=[{'error': e}])
```

In this code, we read the uploaded image, transform it into the format required by our model, and then get the prediction. The prediction is then rendered on the `index.html` template.

## Step 3: Create HTML templates

We need to create HTML templates for our Flask application. Create a folder called `templates` and add 3 files: `base.html`, `index.html`, and `upload_error.html`.

- `base.html` will contain the basic structure of our HTML pages
- `index.html` will display the prediction results
- `upload_error.html` will display error messages when the file upload fails

The contents of the html files can be found on the [repository](https://github.com/asprasan/PlantVillageClassification).

## Deploying on Render

To deploy our Flask application on Render, follow these steps:

1. Create a new account on Render and log in.
2. Click on "New" and select "Web Service".
3. Connect your GitHub repository that contains the Flask application.
4. Choose the branch you want to deploy and click "Next".
5. In the "Environment" section, select "Python 3" and specify the build and start commands:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
6. Click "Create Web Service" and wait for the deployment to complete.

Once the deployment is successful, you will receive a URL where your Flask application is hosted. You can access this URL to upload images and see the predictions from your model.

The deployed application can be found at [Plant Village Classification](https://plant-village-kxps.onrender.com/).
