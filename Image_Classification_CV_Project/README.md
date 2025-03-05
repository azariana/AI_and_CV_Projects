Developed by azariana 🚀

Project Overview

This project is a Convolutional Neural Network (CNN) implementation for recognizing handwritten digits using the MNIST dataset. The model is trained on grayscale images of digits (0-9) and predicts the corresponding class with high accuracy.

Project Steps

Dataset Loading & Visualization

Load the MNIST dataset containing 60,000 training images and 10,000 testing images.

Visualize sample images to understand the dataset.

Data Preprocessing

Normalize pixel values to range [0,1] for better convergence.

Reshape images to fit the input format of the CNN.

Building the CNN Model

Define a sequential model with convolutional layers for feature extraction.

Apply max pooling to reduce spatial dimensions.

Use fully connected layers for classification.

Model Compilation & Training

Compile the model with the Adam optimizer and sparse categorical cross-entropy loss function.

Train the model on the training dataset while validating on the test dataset.

Model Evaluation

Evaluate model accuracy on unseen test data.

Identify potential overfitting issues and improve generalization.

Making Predictions

Test the model with new images from the test dataset.

Compare predicted labels with true labels to analyze performance.

Key Learnings

Understanding how convolutional layers extract spatial features.

Importance of normalization and data preprocessing in deep learning.

Training a CNN with TensorFlow and Keras.

Evaluating model performance and handling overfitting.

This project serves as a great introduction to deep learning and CNN-based image classification! 🎯