MNIST Handwritten Digit Classification

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

📌 Project Overview

This project trains a deep learning model to recognize digits (0-9) from the MNIST dataset. The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels in grayscale.

🚀 Features

Loads and preprocesses the MNIST dataset

Builds a CNN model using TensorFlow/Keras

Trains the model with training data

Evaluates the model on test data

Visualizes predictions with Matplotlib

🛠 Setup & Installation

To run this project, ensure you have Python and TensorFlow installed. You can set up your environment using Anaconda:

conda create --name mnist_env python=3.10
conda activate mnist_env
pip install tensorflow matplotlib numpy

📂 Project Structure

|-- mnist_classification.ipynb  # Jupyter Notebook with full project
|-- README.md                   # Project documentation
|-- model/                       
    |-- saved_model.pb           # Saved trained model

📊 Dataset

The dataset is available in TensorFlow Datasets:

from tensorflow.keras import datasets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

🏗 Model Architecture

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

🎯 Training & Evaluation

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=8, validation_data=(x_test, y_test))

🔍 Testing the Model

import numpy as np
import matplotlib.pyplot as plt

predictions = model.predict(x_test)
num_samples = 5
indices = np.random.choice(len(x_test), num_samples)

fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
for i, idx in enumerate(indices):
    axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[idx])
    true_label = y_test[idx]
    axes[i].set_title(f"Pred: {predicted_label} | True: {true_label}")
    axes[i].axis("off")
plt.show()



