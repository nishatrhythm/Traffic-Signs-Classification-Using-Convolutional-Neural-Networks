# Traffic Signs Classification Using Convolutional Neural Networks (CNN)

This repository provides a complete implementation of a Traffic Signs Classification model using Convolutional Neural Networks (CNN). The goal is to classify various types of traffic signs for use in applications such as autonomous driving.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results and Visualizations](#results-and-visualizations)
7. [Live Testing](#live-testing)
8. [Android App](#android-app)

---

## Introduction

Traffic sign recognition is crucial for the development of self-driving cars and other intelligent road systems. This project employs a CNN model to classify images of traffic signs into 43 distinct classes. Using image processing and data augmentation techniques, the model achieves robust performance, making it suitable for real-world applications.

![Overview](graphs%20and%20images/terminal_1.png)

![Overview](graphs%20and%20images/overview.png)

---

## Dataset

The project uses the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset, which contains over 35,000 images of 43 different traffic sign classes. The images are of varying sizes, lighting conditions, and angles, making this a challenging classification task.

### Data Split
- **Training Set**: 80% of the dataset
- **Validation Set**: 20% of the training data
- **Test Set**: 20% of the total dataset

![Class Distribution](graphs%20and%20images/distribution.png)

---

## Model Architecture

The CNN model has been built using the Keras framework. The architecture includes:

- **Convolutional Layers**: Extract features from the input images.
- **MaxPooling Layers**: Reduce the spatial dimensions of feature maps.
- **Dropout Layers**: Prevent overfitting by dropping nodes during training.
- **Dense Layers**: Perform final classification using a softmax activation function.

![Overview](graphs%20and%20images/terminal_2.png)

### Model Summary

The network includes:
- Two sets of Convolutional and MaxPooling layers.
- Dropout layers with a 50% drop rate to enhance generalization.
- A fully connected layer followed by a softmax output layer.

---

## Installation and Setup

### Prerequisites
Make sure you have the following packages installed:
- Python (>= 3.6)
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolution-Neural-Networks.git
   cd Traffic-Signs-Classification-Using-Convolution-Neural-Networks
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## Training and Evaluation

### Preprocessing

Images are preprocessed using OpenCV:

- **Grayscale Conversion**: Simplifies the image by reducing color channels.
- **Histogram Equalization**: Normalizes the lighting conditions.
- **Normalization**: Scales pixel values to the [0, 1] range.

### Data Augmentation

To improve model generalization, data augmentation techniques such as width/height shifts, zoom, shear, and rotations are applied.

### Training

To train the model, run:
```bash
python TrafficSigns_main.py
```
The model is trained using the Adam optimizer and categorical cross-entropy loss function. The training history, including accuracy and loss curves, is plotted for analysis.

![Overview](graphs%20and%20images/terminal_3.png)

### Evaluation
The model is evaluated on the test set to measure performance. The final model is saved as `model_trained.p` for later use.

---

## Results and Visualizations

### Accuracy Curves

![Accuracy Curves](graphs%20and%20images/accuracy.png)

### Loss Curves

![Loss Curves](graphs%20and%20images/loss.png)

---

## Live Testing

You can test the model in real-time using a webcam. The model detects and classifies traffic signs with a specified probability threshold.

### Running the Live Test

```bash
python TrafficSigns_test.py
```
- Press `q` to quit the live demo.
- The `TrafficSigns_test.py` script captures images from the webcam, preprocesses them, and classifies the traffic signs using the trained model.

### Live Test Results

Below are some sample screenshots from the live testing:

| ![Live Test 1](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_1.png) | ![Live Test 2](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_2.png) | ![Live Test 3](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_3.png) |
|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
| ![Live Test 4](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_4.png) | ![Live Test 5](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_5.png) | ![Live Test 6](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/Screenshot_6.png) |

---

## Android App

In addition to the Python implementation, an Android app has been developed based on this project. The app uses the trained CNN model to classify traffic signs in real-time, making it suitable for mobile and on-the-go applications.

You can find the Android app and more details in the dedicated repository: [Android App of Traffic Signs Classification using CNN](https://github.com/nishatrhythm/Android-App-of-Traffic-Signs-Classification-using-CNN).

### App Screenshots

| ![App Screenshot 1](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/App_Screenshot_1.PNG) | ![App Screenshot 2](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/App_Screenshot_2.PNG) | ![App Screenshot 3](https://github.com/nishatrhythm/Traffic-Signs-Classification-Using-Convolutional-Neural-Networks/blob/main/live%20test%20images/App_Screenshot_3.PNG) |
|:--------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|

---
