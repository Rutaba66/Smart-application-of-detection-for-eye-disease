import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
img_size = 224
batch_size = 32
epochs = 50
num_classes = 5
train_dir = r'D:\Nazia Apu\Data_nazia-20250516T152449Z-1-001\Data_nazia\Train'
val_dir = r'D:\Nazia Apu\Data_nazia-20250516T152449Z-1-001\Data_nazia\Val'
class_labels = ['diabetic_retinopathy', 'glaucoma', 'healthy', 'pterygium', 'retinal_detachment']

# ---------------- DATA AUGMENTATION ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ---------------- MODEL: DenseNet121 ----------------
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------- TRAINING ----------------
checkpoint = ModelCheckpoint("densenet_eye_disease.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint]
)

print("\n✅ Training complete. Best model saved as 'densenet_eye_disease.h5'.")

# ---------------- FINAL EVALUATION ----------------
print("\n🔍 Loading best saved model for evaluation...")
model = load_model("densenet_eye_disease.h5")

val_generator.reset()
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# ---------------- REPORTS ----------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

# ---------------- CONFUSION MATRIX PLOT ----------------
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j], ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
