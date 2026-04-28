# 🌿 Plant Nutrient Classifier

AI-powered binary image classification system that detects **Nutrient Deficiency** in plant leaves using deep learning.

## What's New

- Improved model performance with a tuned classification threshold.
- Cleaner Flask UI with prediction, severity, and recommendation output.
- Metrics dashboard for accuracy, confusion matrix, and per-class evaluation.
- More focused project structure for easier maintenance and deployment.

---

## 🎯 Project Overview

| Feature | Detail |
|---|---|
| **Task** | Binary Classification (Healthy vs Nutrient Deficiency) |
| **Model** | MobileNetV2 with Transfer Learning |
| **Input** | Leaf images (128×128 RGB) |
| **Output** | Prediction + Confidence + Severity + Recommendations |
| **Accuracy** | 90.6% |
| **Threshold** | 0.65 |
| **Framework** | TensorFlow/Keras + Flask |

---

## 🏗️ Project Structure

```
plant-nutrient-classifier/
├── app.py
├── plant_nutrient_classifier.h5
├── train_model.py               # Optional training script
├── evaluate_model.py            # Optional metrics generator
├── metrics.json                 # Optional metrics data for dashboard
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   ├── result.html
│   └── metrics.html
└── static/
	├── css/
	└── uploads/                 # Ignored
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain the Model
```bash
python prepare_dataset.py   # Organize dataset
python train_model.py        # Train MobileNetV2
```

### 3. Generate Metrics
```bash
python evaluate_model.py
```

This creates or updates `metrics.json`, which powers the `/metrics` page in the Flask app.

### 4. Run the App
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🧠 Key Technical Decisions

### Transfer Learning with MobileNetV2
- Pre-trained on ImageNet (1.4M images, 1000 classes)
- Base layers frozen — only the custom head is trained
- Significantly faster training and better accuracy with small datasets

### Threshold-Based Prediction (0.65)
- Default argmax can cause unstable predictions near 50%
- Threshold of 0.65 reduces false positives (healthy plants flagged as deficient)
- Trade-off: slightly higher false negatives, but safer for agriculture use

### Metrics Dashboard
- The `/metrics` page reads from `metrics.json` when available.
- It is meant for quick inspection of accuracy, confusion matrix, and error analysis.

### Class Weight Balancing
- Automatically compensates for unequal class sizes
- Prevents the model from defaulting to the majority class

---

## 📊 Features

| Feature | Description |
|---|---|
| **Image Upload** | Drag-and-drop or click to upload leaf images |
| **Prediction** | Healthy or Nutrient Deficiency classification |
| **Confidence** | Percentage confidence with visual progress bar |
| **Severity** | Mild / Moderate / Severe based on confidence level |
| **Recommendations** | Agriculture-specific actions based on severity |
| **Metrics Dashboard** | Accuracy, confusion matrix, per-class metrics, error analysis |

---

## 🔍 Error Analysis & Limitations

### Known Limitations
1. **Small dataset** — Model may not generalize well to all plant species
2. **Binary only** — Cannot identify specific nutrient deficiencies (N, P, K, Fe, etc.)
3. **Image quality dependency** — Poor lighting or blurry images reduce accuracy
4. **Single leaf focus** — Cannot analyze whole-plant health

### Common Error Patterns
- **False positives**: Healthy leaves with natural discoloration flagged as deficient
- **False negatives**: Early-stage deficiency with subtle symptoms missed
- **Low confidence zone**: Predictions between 50–65% are unreliable

---

## 🔮 Future Improvements

1. **Multi-class classification** — Identify specific deficiencies (Nitrogen, Phosphorus, Potassium, Iron)
2. **Larger dataset** — Collect more diverse leaf images across species
3. **Fine-tuning** — Unfreeze top MobileNetV2 layers for better feature extraction
4. **Grad-CAM visualization** — Show which leaf regions influence the prediction
5. **Mobile deployment** — Convert to TFLite for on-device inference
6. **History tracking** — Store past predictions for trend analysis

---

## ✅ Release Notes

- Final model accuracy: **90.6%**
- Threshold tuned to **0.65** for more stable predictions
- Added metrics page support through `metrics.json`
- Cleaned project structure and removed temporary scripts from the final repo

---

## 👤 Author

Myaka Akhila
B.Tech CSE | AI/ML Enthusiast

---
