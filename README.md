# Garbage Classification with Deep Learning

This repository contains a deep learning project for classifying images of garbage into categories such as metal, glass, paper, trash, cardboard, plastic, and white-glass using Convolutional Neural Networks (CNNs) and transfer learning with MobileNetV2.

## 📁 Dataset

The dataset is structured into subfolders, each representing a different class:
- metal
- glass
- paper
- trash
- cardboard
- plastic
- white-glass

The dataset used in this project was sourced from Kaggle: [Garbage Classification Dataset](https://www.kaggle.com/datasets/)

## 🧠 Model

The model is built using:
- *TensorFlow* and *Keras*
- *MobileNetV2* for transfer learning
- Image preprocessing with ImageDataGenerator
- Model layers: Conv2D, MaxPooling2D, BatchNormalization, Dropout, etc.

## 📊 Evaluation

Model performance is evaluated using:
- Accuracy and loss plots
- Confusion matrix
- Classification report with precision, recall, and F1-score

## 🔧 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
