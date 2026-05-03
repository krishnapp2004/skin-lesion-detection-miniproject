"""
ADVANCED Skin Lesion Detection with Invalid Image Detection
Target: 95%+ Accuracy (10 classes including 'Invalid')
Windows-Compatible Version
Python 3.10.6 Compatible
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                     BatchNormalization, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                       ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import json
from datetime import datetime
import math

# ==================== WINDOWS FIX ====================
# Disable problematic multiprocessing on Windows
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

print("=" * 90)
print("🚀 SKIN LESION DETECTION WITH INVALID IMAGE REJECTION")
print("🎯 TARGET: 95%+ ACCURACY (10 CLASSES)")
print("🪟 Windows-Compatible Version")
print("=" * 90)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Platform: Windows (multiprocessing disabled for compatibility)")
print("=" * 90)

# ==================== CONFIGURATION ====================

IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 60

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "models/melanoma_10class_invalid_detection.keras"

os.makedirs("models", exist_ok=True)

print("\n📊 Configuration:")
print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Max Epochs: {EPOCHS}")
print(f"   Classes: 9 lesion types + 1 invalid = 10 TOTAL")

# ==================== DATA AUGMENTATION ====================

print("\n📂 Loading Training Data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.2,
    fill_mode='reflect',
    validation_split=0.16
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Test data
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes

print(f"\n✅ Data Loaded Successfully!")
print(f"   Training Samples: {train_generator.samples}")
print(f"   Validation Samples: {val_generator.samples}")
print(f"   Test Samples: {test_generator.samples}")
print(f"   Number of Classes: {NUM_CLASSES}")

class_names = list(train_generator.class_indices.keys())
print(f"\n   Classes: {', '.join(class_names)}")

if NUM_CLASSES != 10:
    print(f"\n⚠️  WARNING: Expected 10 classes, found {NUM_CLASSES}")
    print("   Make sure you have the 'invalid' folder with non-medical images!")

# Save class indices
with open('models/class_indices_10class.json', 'w') as f:
    json.dump(train_generator.class_indices, f, indent=2)

# ==================== BALANCED CLASS WEIGHTS ====================

print("\n⚖️  Calculating Class Weights...")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weight_dict = dict(enumerate(class_weights))

print("\n   Original Class Weights:")
for class_name, class_idx in sorted(train_generator.class_indices.items(), 
                                     key=lambda x: x[1]):
    weight = class_weight_dict[class_idx]
    marker = " ⚠️ (TOO HIGH!)" if weight > 3.0 else ""
    print(f"      {class_idx}. {class_name}: {weight:.3f}{marker}")

# FIX: Cap the invalid class weight to prevent extreme imbalance
# Find invalid class (case-insensitive search)
invalid_idx = None
invalid_class_name = None

for class_name, idx in train_generator.class_indices.items():
    if 'invalid' in class_name.lower():
        invalid_idx = idx
        invalid_class_name = class_name
        break

if invalid_idx is not None:
    original_weight = class_weight_dict[invalid_idx]
    print(f"\n   🔍 Found invalid class: '{invalid_class_name}' (index {invalid_idx})")
    
    # Cap at 3.0 instead of letting it be 13.382
    if original_weight > 3.0:
        print(f"   🔧 Capping weight: {original_weight:.3f} → 3.0")
        print("      (Prevents model from rejecting everything as invalid)")
        class_weight_dict[invalid_idx] = 3.0
    else:
        print(f"   ✅ Weight is already reasonable: {original_weight:.3f}")
else:
    print(f"\n   ⚠️  WARNING: No 'invalid' class found!")
    print(f"      Make sure you have the 'invalid' folder in your dataset.")

print("\n   ✅ Final Adjusted Class Weights:")
for class_name, class_idx in sorted(train_generator.class_indices.items(), 
                                     key=lambda x: x[1]):
    weight = class_weight_dict[class_idx]
    marker = " ⭐" if class_name == 'invalid' else ""
    print(f"      {class_idx}. {class_name}: {weight:.3f}{marker}")

# ==================== MODEL ARCHITECTURE ====================

print("\n🏗️  Building Model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=1.0
)

base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D(name='global_avg_pool')(x)

# Deep classification head
x = Dense(512, activation='relu', name='fc1')(x)
x = BatchNormalization(name='bn1')(x)
x = Dropout(0.5, name='dropout1')(x)

x = Dense(256, activation='relu', name='fc2')(x)
x = BatchNormalization(name='bn2')(x)
x = Dropout(0.4, name='dropout2')(x)

x = Dense(128, activation='relu', name='fc3')(x)
x = BatchNormalization(name='bn3')(x)
x = Dropout(0.3, name='dropout3')(x)

outputs = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = Model(inputs, outputs, name='SkinAI_10Class')

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', 
             keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print("\n✅ Model Built Successfully!")
print(f"   Total Parameters: {model.count_params():,}")
trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"   Trainable Parameters: {trainable:,}")

# ==================== CALLBACKS ====================

def cosine_decay(epoch, lr):
    """Cosine decay learning rate"""
    if epoch < 3:
        return lr * (epoch + 1) / 3
    progress = (epoch - 3) / (EPOCHS - 3)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))

callbacks_phase1 = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

# ==================== PHASE 1: TRANSFER LEARNING ====================

print("\n" + "=" * 90)
print("🚀 PHASE 1: TRANSFER LEARNING (15 Epochs)")
print("=" * 90)
print("Training classifier head...\n")

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1,
    verbose=1,
    workers=1,                    # WINDOWS FIX: Single worker
    use_multiprocessing=False     # WINDOWS FIX: Disabled
)

# ==================== PHASE 2: PARTIAL FINE-TUNING ====================

print("\n" + "=" * 90)
print("🔥 PHASE 2: PARTIAL FINE-TUNING (Top 50 Layers)")
print("=" * 90)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', 
             keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"   Trainable Parameters: {trainable:,}\n")

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    initial_epoch=15,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1,
    verbose=1,
    workers=1,
    use_multiprocessing=False
)

# ==================== PHASE 3: FULL FINE-TUNING ====================

print("\n" + "=" * 90)
print("🚀 PHASE 3: FULL FINE-TUNING (All Layers)")
print("=" * 90)

for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', 
             keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"   Trainable Parameters: {trainable:,}\n")

callbacks_phase3 = [
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', mode='max', 
                   save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=12, 
                 restore_best_weights=True, verbose=1),
    LearningRateScheduler(cosine_decay, verbose=0)
]

history_phase3 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=40,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase3,
    verbose=1,
    workers=1,
    use_multiprocessing=False
)

# ==================== EVALUATION ====================

print("\n" + "=" * 90)
print("📊 FINAL EVALUATION")
print("=" * 90)

best_model = keras.models.load_model(MODEL_PATH)

test_results = best_model.evaluate(test_generator, verbose=0)
test_loss = test_results[0]
test_accuracy = test_results[1]
test_auc = test_results[2]
test_precision = test_results[3]
test_recall = test_results[4]

print(f"\n✅ Test Results:")
print(f"   Accuracy:  {test_accuracy * 100:.2f}%")
print(f"   AUC:       {test_auc:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   Loss:      {test_loss:.4f}")

# Classification report
print("\n📋 Detailed Classification Report:")
y_pred = best_model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

class_names_list = [k for k, v in sorted(train_generator.class_indices.items(), 
                                          key=lambda x: x[1])]

print("\n" + classification_report(y_true, y_pred_classes, 
                                    target_names=class_names_list, 
                                    digits=4))

# Invalid class performance
if invalid_idx is not None:
    invalid_mask = y_true == invalid_idx
    invalid_correct = np.sum((y_pred_classes == invalid_idx) & invalid_mask)
    invalid_total = np.sum(invalid_mask)
    if invalid_total > 0:
        invalid_accuracy = invalid_correct / invalid_total
        print(f"\n🎯 Invalid Detection Performance ('{invalid_class_name}' class):")
        print(f"   Correct rejections: {invalid_correct}/{invalid_total}")
        print(f"   Invalid accuracy: {invalid_accuracy * 100:.2f}%")
    else:
        print(f"\n⚠️  No invalid images found in test set!")
else:
    print(f"\n⚠️  No invalid class detected in model.")

# ==================== SUMMARY ====================

print("\n" + "=" * 90)
print("📈 TRAINING SUMMARY")
print("=" * 90)

all_accuracy = (history_phase1.history['accuracy'] + 
                history_phase2.history['accuracy'] + 
                history_phase3.history['accuracy'])
all_val_accuracy = (history_phase1.history['val_accuracy'] + 
                    history_phase2.history['val_accuracy'] + 
                    history_phase3.history['val_accuracy'])

best_epoch = np.argmax(all_val_accuracy) + 1
best_val_acc = max(all_val_accuracy)

print(f"\n   Best Validation Accuracy: {best_val_acc * 100:.2f}% (Epoch {best_epoch})")
print(f"   Final Training Accuracy: {all_accuracy[-1] * 100:.2f}%")
print(f"   Final Validation Accuracy: {all_val_accuracy[-1] * 100:.2f}%")
print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")

if test_accuracy >= 0.95:
    print(f"\n   🎉🎉🎉 EXCEPTIONAL! {test_accuracy * 100:.2f}% (≥95%)")
elif test_accuracy >= 0.90:
    print(f"\n   🎉🎉 EXCELLENT! {test_accuracy * 100:.2f}% (≥90%)")
elif test_accuracy >= 0.85:
    print(f"\n   🎉 VERY GOOD! {test_accuracy * 100:.2f}% (≥85%)")

# Save history
history_dict = {
    'model_name': 'SkinAI_10Class_InvalidDetection',
    'num_classes': NUM_CLASSES,
    'class_names': class_names_list,
    'train_accuracy': [float(x) for x in all_accuracy],
    'val_accuracy': [float(x) for x in all_val_accuracy],
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'auc': float(test_auc),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'loss': float(test_loss)
    },
    'best_epoch': int(best_epoch),
    'best_val_accuracy': float(best_val_acc)
}

with open('models/training_history_10class.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

print("\n" + "=" * 90)
print("✅ TRAINING COMPLETED!")
print("=" * 90)
print(f"📦 Model: {MODEL_PATH}")
print(f"📊 Class indices: models/class_indices_10class.json")
print(f"📈 History: models/training_history_10class.json")
print(f"\n⚡ Ready for deployment with invalid image rejection!")
print("=" * 90 + "\n")