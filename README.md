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

The dataset used in this project was sourced from Kaggle: [Garbage Classification Dataset]([https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification))

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

bash
pip install -r requirements.txt


### Key Libraries:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- scikit-learn

## 🚀 Usage

To train and evaluate the model, simply run the Jupyter Notebook:

bash
jupyter notebook meeee.ipynb


## 📈 Results

The notebook includes visualizations for:
- Training/validation accuracy and loss
- Confusion matrix
- Classification metrics for each class

## 📌 Notes

- Uses MobileNetV2 for feature extraction.
- Includes data augmentation to reduce overfitting.
- Designed to help automate waste sorting through image classification.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
