
#kütüphaneler
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

#dataseti yükleme
DATADIR = r"desktop/dataset6"
CATEGORIES = ["Acne and Rosacea Photos","Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions","Atopic Dermatitis Photos","Bullous Disease Photos","Cellulitis Impetigo and other Bacterial Infections","Eczema Photos","Exanthems and Drug Eruptions","Hair Loss Photos Alopecia and other Hair Diseases","Light Diseases and Disorders of Pigmentation","Lupus and other Connective Tissue diseases","Melanoma Skin Cancer Nevi and Moles","Nail Fungus and other Nail Disease","Poison Ivy Photos and other Contact Dermatitis","Pox","Psoriasis pictures Lichen Planus and related diseases","Scabies Lyme Disease and other Infestations and Bites","Seborrheic Keratoses and other Benign Tumors","Systemic Disease"]
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        break
    break

#dataseti standartlaştırma ve kümelendirme
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATADIR,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  DATADIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off"),
for image_batch, labels_batch in train_ds:
  break
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#modeli eğitmek

num_classes = len(class_names)

model = Sequential([

  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history["accuracy"]
val_acc = history.history['calidation_accuracy']

loss = history.history["loss"]
val_loss = history.history['validation_loss']

epochs_range = range(epochs)
model.save('derma.h5')


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Kesinliği')
plt.plot(epochs_range, val_acc, label='Doğrulama Kesinliği')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Kesinlikleri')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.show()