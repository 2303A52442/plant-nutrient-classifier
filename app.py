from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('plant_nutrient_classifier.h5')
class_names = ['Healthy', 'Nutrient']

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = float(prediction[0][predicted_label])
    return class_names[predicted_label], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction, confidence = predict_image(file_path)
            return render_template('index.html', prediction=prediction, confidence=confidence, image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
