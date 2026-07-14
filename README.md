# 🐱🐶 Cat vs Dog Image Classifier using Convolutional Neural Networks

## Project Overview

This project focuses on building an end-to-end Deep Learning solution capable of automatically classifying images as either **cats** or **dogs** using a Convolutional Neural Network (CNN). Unlike traditional machine learning algorithms that rely on manually engineered features, Convolutional Neural Networks automatically learn meaningful visual patterns directly from images. Throughout this project, I implemented the complete deep learning pipeline, beginning with dataset acquisition and preprocessing, followed by model development, training, evaluation, and prediction on unseen images.

The objective was not only to achieve accurate classification but also to understand how convolutional neural networks extract features from images and progressively learn complex representations required for image recognition tasks.

---

# Business Problem

Image classification has become an essential component of modern computer vision applications. Platforms dealing with thousands or even millions of images require automated systems capable of recognizing objects without human intervention. Manually categorizing images is not only labor-intensive but also prone to inconsistency and human error.

A simple example of this problem can be seen in pet adoption websites, veterinary clinics, wildlife monitoring systems, and image management platforms. These applications frequently need to distinguish between different animal species before performing downstream tasks such as tagging, searching, organizing images, or triggering specific workflows.

The objective of this project is to develop an automated image classification model capable of identifying whether an image contains a cat or a dog with high accuracy while eliminating the need for manual labeling during prediction.

---

# Dataset

The dataset used in this project was obtained from Kaggle and contains thousands of labeled images belonging to two different classes: **Cats** and **Dogs**. The images exhibit considerable variation in background, lighting conditions, camera angles, poses, and object sizes, making the classification problem more realistic and challenging.

Instead of manually assigning labels to every image, TensorFlow's `image_dataset_from_directory()` utility was used to automatically infer class labels from the directory names. Images stored inside the **cat** folder were assigned one class label, while images inside the **dog** folder were assigned the other. This approach significantly simplifies dataset preparation while reducing the possibility of human labeling errors.

---

# Data Preprocessing

Deep learning models perform considerably better when input data is standardized before training. Since digital images store pixel intensities between **0 and 255**, directly feeding these large values into a neural network can slow down learning and make optimization unstable.

To address this issue, every image was normalized by dividing each pixel value by 255, thereby scaling the entire image into the range of **0 to 1**. This normalization ensures that all input features remain on a comparable scale, allowing gradient descent to converge faster and improving overall numerical stability during training.

TensorFlow's Dataset API was used to efficiently apply preprocessing to every image through the `map()` function, ensuring that normalization occurs automatically while batches are loaded during training.

---

# Model Architecture

The classification model was built from scratch using a Sequential Convolutional Neural Network in TensorFlow and Keras. The network consists of three convolutional blocks followed by fully connected layers responsible for making the final classification.

Each convolutional layer uses **3×3 kernels** to scan the image and learn increasingly complex visual features. During the initial stages, the network focuses on simple characteristics such as edges, corners, and color gradients. As the data passes through deeper convolutional layers, the learned features become progressively more abstract, allowing the model to recognize patterns such as eyes, ears, whiskers, fur textures, facial structures, and eventually the overall appearance of cats and dogs.

Batch Normalization was introduced after every convolution layer to stabilize the distribution of intermediate activations. This technique accelerates convergence, reduces internal covariate shift, and allows the model to train more efficiently.

Following each convolution block, Max Pooling was applied to reduce the spatial dimensions of the extracted feature maps. Pooling decreases computational complexity while preserving the most informative features, making the model more robust to small translations and distortions in the input images.

After feature extraction, the multidimensional feature maps were flattened into a one-dimensional vector so that fully connected layers could perform classification. Two dense layers containing 128 and 64 neurons respectively were used to learn higher-level feature combinations before generating the final prediction.

Dropout layers were incorporated between dense layers as a regularization technique. During training, Dropout randomly deactivates a fraction of neurons, preventing the network from relying excessively on particular neurons and reducing the risk of overfitting.

The final output layer consists of a single neuron with a **Sigmoid activation function**, producing a probability value between 0 and 1. Predictions below 0.5 correspond to the cat class, whereas values greater than or equal to 0.5 indicate the dog class.

---

# Model Training

The network was compiled using the **Adam optimizer**, **Binary Crossentropy loss function**, and **Accuracy** as the evaluation metric.

Adam was selected because it combines adaptive learning rates with momentum, making it one of the most efficient optimization algorithms for deep learning tasks. Binary Crossentropy was chosen since this project involves only two target classes. During training, the optimizer continuously updates the network's weights through backpropagation in an attempt to minimize prediction error.

The model was trained for ten epochs using the prepared training dataset, while validation performance was monitored after every epoch. Recording validation metrics alongside training metrics made it possible to identify whether the model was learning meaningful representations or beginning to overfit the training data.

---

# Model Evaluation

The performance of the trained model was evaluated using both training and validation accuracy and loss curves. Accuracy illustrates the proportion of correctly classified images, while loss measures how far the predicted probabilities deviate from the true labels.

Monitoring these curves throughout training provides valuable insight into model behavior. A continuously decreasing training loss together with a similar validation loss generally indicates effective learning. Conversely, a widening gap between training and validation performance often signals overfitting, suggesting that the network is memorizing training examples instead of learning generalized image representations.

---

# Prediction Pipeline

After completing training, the model was used to classify previously unseen images. Each test image was loaded using OpenCV, resized to **256 × 256 pixels**, normalized using the same preprocessing pipeline employed during training, and finally passed through the trained CNN.

The network produces a probability score representing the likelihood that the input image belongs to the dog class. Based on this probability, the final prediction is generated. If the predicted probability is less than 0.5, the image is classified as a cat; otherwise, it is classified as a dog.

Using identical preprocessing during both training and inference ensures consistency and allows the model to make reliable predictions on new images.

---

# Technologies Used

This project was implemented using **Python**, **TensorFlow**, **Keras**, **NumPy**, **OpenCV**, **Matplotlib**, and **KaggleHub**. TensorFlow and Keras were used to design, train, and evaluate the neural network, while OpenCV handled image loading and preprocessing during inference. Matplotlib was utilized for visualizing training performance, and KaggleHub was used to access the dataset directly from Kaggle.

---

# Future Improvements

Although the current model performs well for binary classification, there are several opportunities for improvement. The model can be further enhanced through extensive data augmentation, hyperparameter tuning, longer training schedules, and transfer learning using pretrained architectures such as ResNet50, MobileNetV2, or EfficientNet. Additionally, deploying the trained model as a web application using Streamlit or converting it to TensorFlow Lite for mobile devices would make the solution more practical for real-world applications.

---
