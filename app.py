"""
Plant Nutrient Classifier — Flask App
Steps 3-6: Prediction logic, threshold tuning, severity, recommendations.
"""
from flask import Flask, render_template, request
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Load Model ===
model = tf.keras.models.load_model('plant_nutrient_classifier.h5')

# Load class names from training mapping when available
RAW_CLASS_NAMES = ['Healthy', 'Nutrient']
if os.path.exists('class_indices.json'):
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        RAW_CLASS_NAMES = [name for name, _idx in sorted(class_indices.items(), key=lambda item: item[1])]
    except Exception:
        RAW_CLASS_NAMES = ['Healthy', 'Nutrient']

CANONICAL_LABELS = {
    'healthy': 'Healthy',
    'nutrient': 'Nutrient Deficiency',
    'nutrient deficiency': 'Nutrient Deficiency'
}

def _canonical_label(name):
    key = str(name).strip().lower()
    return CANONICAL_LABELS.get(key, name)

HEALTHY_INDEX = 0
NUTRIENT_INDEX = 1
for idx, name in enumerate(RAW_CLASS_NAMES):
    if 'nutrient' in str(name).lower():
        NUTRIENT_INDEX = idx
    if 'healthy' in str(name).lower():
        HEALTHY_INDEX = idx

HEALTHY_LABEL = _canonical_label(RAW_CLASS_NAMES[HEALTHY_INDEX])
NUTRIENT_LABEL = _canonical_label(RAW_CLASS_NAMES[NUTRIENT_INDEX])

# === Step 4: Prediction Threshold ===
# Higher threshold = fewer false positives (more precise)
# Lower threshold = fewer false negatives (more sensitive)
# 0.65 is the best balance found in threshold sweeping for this model
THRESHOLD = float(os.getenv('NUTRIENT_THRESHOLD', '0.65'))


# === Step 3: Correct Prediction Function ===
def predict_image(image_path):
    """
    Predict whether a plant is Healthy or has Nutrient Deficiency.

    Returns: (label, confidence, probabilities)
    - label: 'Healthy' or 'Nutrient Deficiency'
    - confidence: float 0-100 (percentage)
    - probabilities: dict with both class probabilities
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    # Handle both softmax (2 outputs) and sigmoid (1 output)
    if prediction.shape[-1] == 2:
        # Softmax output: [prob_healthy, prob_nutrient]
        healthy_prob = float(prediction[0][HEALTHY_INDEX])
        nutrient_prob = float(prediction[0][NUTRIENT_INDEX])
    else:
        # Sigmoid output: single probability for nutrient class
        nutrient_prob = float(prediction[0][0])
        healthy_prob = 1.0 - nutrient_prob

    # Apply threshold on nutrient probability
    if nutrient_prob >= THRESHOLD:
        label = NUTRIENT_LABEL
        confidence = round(nutrient_prob * 100, 1)
    else:
        label = HEALTHY_LABEL
        confidence = round(healthy_prob * 100, 1)

    probabilities = {
        HEALTHY_LABEL: round(healthy_prob * 100, 1),
        NUTRIENT_LABEL: round(nutrient_prob * 100, 1)
    }

    return label, confidence, probabilities


# === Step 6: Severity + Recommendations ===
def get_severity(label, confidence):
    """Determine severity based on prediction and confidence."""
    if label == HEALTHY_LABEL:
        return None

    # For Nutrient Deficiency
    if confidence >= 90:
        return 'Severe'
    elif confidence >= 75:
        return 'Moderate'
    else:
        return 'Mild'


def get_recommendation(label, severity):
    """Agriculture-specific recommendations based on prediction and severity."""
    if label == HEALTHY_LABEL:
        return {
            'title': 'Plant is Healthy',
            'message': 'Your plant appears healthy. Continue your current care routine.',
            'actions': [
                'Maintain regular watering schedule',
                'Continue current fertilizer regimen',
                'Monitor for early signs of stress',
            ]
        }

    recommendations = {
        'Mild': {
            'title': 'Minor Nutrient Deficiency Detected',
            'message': 'Early signs of nutrient deficiency. Easy to correct with minor adjustments.',
            'actions': [
                'Apply balanced NPK fertilizer (10-10-10)',
                'Check soil pH — aim for 6.0 to 7.0',
                'Ensure adequate watering without waterlogging',
                'Re-examine in 1–2 weeks',
            ]
        },
        'Moderate': {
            'title': 'Moderate Nutrient Deficiency',
            'message': 'Noticeable nutrient deficiency. Intervention needed to prevent further damage.',
            'actions': [
                'Apply targeted fertilizer based on symptoms',
                'Test soil for specific nutrient levels (N, P, K)',
                'Improve soil drainage if needed',
                'Consider foliar spray for quick absorption',
                'Re-examine in 1 week',
            ]
        },
        'Severe': {
            'title': 'Severe Nutrient Deficiency',
            'message': 'Significant nutrient deficiency detected. Immediate action required.',
            'actions': [
                'Apply emergency foliar nutrient spray',
                'Conduct comprehensive soil test immediately',
                'Amend soil with compost or organic matter',
                'Consider micronutrient supplements (Fe, Mn, Zn)',
                'Consult agricultural extension officer',
                'Monitor daily and re-examine in 3–5 days',
            ]
        }
    }
    return recommendations.get(severity, recommendations['Mild'])


# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            label, confidence, probabilities = predict_image(file_path)
            severity = get_severity(label, confidence)
            recommendation = get_recommendation(label, severity)

            return render_template('result.html',
                                   prediction=label,
                                   confidence=confidence,
                                   probabilities=probabilities,
                                   severity=severity,
                                   recommendation=recommendation,
                                   image_path=file_path)

    return render_template('index.html')


@app.route('/metrics')
def metrics():
    """Step 7: Metrics dashboard."""
    metrics_data = None
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics_data = json.load(f)
    return render_template('metrics.html', metrics=metrics_data)


if __name__ == '__main__':
    app.run(debug=True)
