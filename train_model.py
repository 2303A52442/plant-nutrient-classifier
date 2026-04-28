"""
Step 2: Model Training with MobileNetV2
Transfer learning for binary classification: Healthy vs Nutrient Deficiency.

Requirements:
- Run prepare_dataset.py first to create dataset/ folder
- Input size: 128x128, transfer learning with frozen base
- Dropout, class weights, early stopping, 25 epochs
"""
import numpy as np
import json
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

# === Configuration ===
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / 'plant_nutrient_classifier.h5'
TRAIN_DIR = BASE_DIR / 'dataset' / 'train'
TEST_DIR = BASE_DIR / 'dataset' / 'test'
CLASS_INDICES_PATH = BASE_DIR / 'class_indices.json'
TRAIN_HISTORY_PATH = BASE_DIR / 'training_history.json'

# === Data Generators with Augmentation ===
# Training: augmentation helps generalize against lighting, background, and mild blur/pose changes
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=(0.7, 1.3),
    fill_mode='nearest',
    validation_split=0.15  # 15% for validation
)

# Test data: normalize only, no augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class mapping (e.g., {'Healthy': 0, 'Nutrient': 1})
class_indices = train_generator.class_indices
print(f"Class mapping: {class_indices}")
with open(CLASS_INDICES_PATH, 'w') as f:
    json.dump(class_indices, f)

# === Build MobileNetV2 Model ===
# Pre-trained on ImageNet, remove top layer
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base - don't retrain ImageNet features

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # 50% dropout to prevent overfitting
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Class Weights (handle imbalance) ===
cw = class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_generator.classes), y=train_generator.classes
)
class_weights_dict = dict(enumerate(cw))
print(f"Class weights: {class_weights_dict}")

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

# === Train ===
print("\n" + "=" * 50)
print("Starting Training...")
print("=" * 50)
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint]
)

# === Fine-tune top MobileNetV2 layers ===
# Keep the architecture the same, but let the top feature layers adapt to leaf-specific patterns.
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    epochs=max(10, EPOCHS // 2),
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint]
)

# === Evaluate ===
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save training history for metrics page
history_data = {
    'accuracy': [float(v) for v in history.history['accuracy']],
    'val_accuracy': [float(v) for v in history.history['val_accuracy']],
    'loss': [float(v) for v in history.history['loss']],
    'val_loss': [float(v) for v in history.history['val_loss']],
    'fine_tune_accuracy': [float(v) for v in fine_tune_history.history['accuracy']],
    'fine_tune_val_accuracy': [float(v) for v in fine_tune_history.history['val_accuracy']],
    'fine_tune_loss': [float(v) for v in fine_tune_history.history['loss']],
    'fine_tune_val_loss': [float(v) for v in fine_tune_history.history['val_loss']],
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
}
with open(TRAIN_HISTORY_PATH, 'w') as f:
    json.dump(history_data, f)

print(f"\n✓ Model saved to '{MODEL_PATH}'")
print(f"✓ Training history saved to 'training_history.json'")
print(f"✓ Class indices saved to 'class_indices.json'")
