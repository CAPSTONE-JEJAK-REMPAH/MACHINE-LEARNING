import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Direktori data
TRAIN_DIR = "C:\\Users\\fadhi\\CNN\\CAPSTONE_PROJECT\\Train_Spices"
TEST_DIR = "C:\\Users\\fadhi\\CNN\\CAPSTONE_PROJECT\\Test_Spices"
VAL_DIR = "C:\\Users\\fadhi\\CNN\\CAPSTONE_PROJECT\\Validate_Spices"

# Data augmentation dan persiapan
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_set = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_set = val_datagen.flow_from_directory(
    VAL_DIR, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

# Load MobileNetV2 sebagai base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layer pre-trained

# Build model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Dropout untuk mencegah overfitting
    tf.keras.layers.Dense(31, activation='softmax')  # Output layer
])

print(model.summary())

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks untuk early stopping dan model checkpoint
checkpoint = ModelCheckpoint(
    'C:/Users/fadhi/CNN/CAPSTONE_PROJECT/Spices_best.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1
)

# Train model
history = model.fit(
    x=train_set,
    validation_data=val_set,
    epochs=5,  # Kurangi jumlah epochs karena transfer learning lebih cepat
    callbacks=[checkpoint, early_stopping]
)

# Evaluasi
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Visualisasi hasil
epoch_range = range(len(acc))

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epoch_range, acc, label="Training Accuracy")
plt.plot(epoch_range, val_acc, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, loss, label="Training Loss")
plt.plot(epoch_range, val_loss, label="Validation Loss")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# Save final model
model.save('C:/Users/fadhi/CNN/CAPSTONE_PROJECT/Spices_final.h5')