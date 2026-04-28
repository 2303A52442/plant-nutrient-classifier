"""
Step 7: Model Evaluation & Metrics Generation
Generates metrics.json for the /metrics Flask page.
Run after training to generate evaluation data.
"""
import os
import numpy as np
import json
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

IMG_SIZE = 128
MODEL_PATH = 'plant_nutrient_classifier.h5'


def load_test_data():
    """Load test images from dataset/test/ or fallback to Healthy/Nutrient folders."""
    images, labels, filenames = [], [], []

    if os.path.exists('dataset/test'):
        base = 'dataset/test'
        class_dirs = sorted(os.listdir(base))
    else:
        base = '.'
        class_dirs = ['Healthy', 'Nutrient']

    for idx, cls in enumerate(class_dirs):
        cls_dir = os.path.join(base, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(os.path.join(cls_dir, fname)).convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img) / 255.0)
                    labels.append(idx)
                    filenames.append(fname)
                except Exception as e:
                    print(f"Error: {e}")

    return np.array(images), np.array(labels), filenames, class_dirs


def generate_metrics():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading test data...")
    X_test, y_true, filenames, class_names = load_test_data()
    print(f"Loaded {len(X_test)} test images")

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_conf = np.max(predictions, axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    correct_samples, incorrect_samples = [], []
    for i in range(len(y_true)):
        sample = {
            'filename': filenames[i],
            'true_label': class_names[y_true[i]],
            'predicted_label': class_names[y_pred[i]],
            'confidence': round(float(y_conf[i]) * 100, 1)
        }
        if y_true[i] == y_pred[i] and len(correct_samples) < 10:
            correct_samples.append(sample)
        elif y_true[i] != y_pred[i] and len(incorrect_samples) < 10:
            incorrect_samples.append(sample)

    total_errors = int(np.sum(y_true != y_pred))
    correct_mask = y_true == y_pred
    avg_conf_correct = round(float(np.mean(y_conf[correct_mask])) * 100, 1) if np.any(correct_mask) else 0
    avg_conf_incorrect = round(float(np.mean(y_conf[~correct_mask])) * 100, 1) if np.any(~correct_mask) else 0

    insights = []
    if accuracy >= 0.9:
        insights.append(f"Strong performance with {accuracy*100:.1f}% accuracy on test data.")
    elif accuracy >= 0.75:
        insights.append(f"Good performance at {accuracy*100:.1f}% accuracy. Room for improvement.")
    else:
        insights.append(f"Accuracy is {accuracy*100:.1f}%. Consider more data or fine-tuning.")

    if avg_conf_correct > avg_conf_incorrect:
        insights.append(f"Well-calibrated: correct predictions avg {avg_conf_correct}% confidence vs {avg_conf_incorrect}% for incorrect.")
    else:
        insights.append("Warning: confidence is poorly calibrated — incorrect predictions have high confidence.")

    insights.append(f"Prediction distribution: {class_names[0]}={int(np.sum(y_pred==0))}, {class_names[1]}={int(np.sum(y_pred==1))} out of {len(y_pred)} samples.")

    metrics = {
        'accuracy': round(accuracy * 100, 1),
        'precision': round(precision * 100, 1),
        'recall': round(recall * 100, 1),
        'f1_score': round(f1 * 100, 1),
        'confusion_matrix': cm,
        'class_names': class_names,
        'per_class': {
            cls: {
                'precision': round(report[cls]['precision'] * 100, 1),
                'recall': round(report[cls]['recall'] * 100, 1),
                'f1': round(report[cls]['f1-score'] * 100, 1),
                'support': int(report[cls]['support'])
            } for cls in class_names
        },
        'total_test_samples': len(y_true),
        'total_errors': total_errors,
        'error_rate': round(total_errors / len(y_true) * 100, 1),
        'avg_confidence_correct': avg_conf_correct,
        'avg_confidence_incorrect': avg_conf_incorrect,
        'correct_samples': correct_samples,
        'incorrect_samples': incorrect_samples,
        'insights': insights,
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAccuracy:  {metrics['accuracy']}%")
    print(f"Precision: {metrics['precision']}%")
    print(f"Recall:    {metrics['recall']}%")
    print(f"F1 Score:  {metrics['f1_score']}%")
    print(f"Errors:    {total_errors}/{len(y_true)}")
    print(f"\n✓ Metrics saved to 'metrics.json'")


if __name__ == '__main__':
    generate_metrics()
