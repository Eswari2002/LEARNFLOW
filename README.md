# LEARNFLOW - Machine Learning Projects

This repository contains machine learning projects developed as part of the LEARNFLOW internship. The projects aim to solve various problems using deep learning techniques with TensorFlow and Keras.

## Table of Contents
- [Facial Emotion Recognition](#facial-emotion-recognition)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)
- [Traffic Sign Recognition](#traffic-sign-recognition)
  - [Problem Statement](#problem-statement)
  - [Installation](#installation-1)
  - [Usage](#usage-1)
  - [Dataset](#dataset-1)
  - [Model Architecture](#model-architecture-1)
  - [Results](#results-1)
  - [Contributing](#contributing-1)
  - [License](#license-1)

## Facial Emotion Recognition

This project aims to develop a model that recognizes facial emotions (e.g., happy, sad, angry) from images or videos. The model is built using deep learning techniques with TensorFlow and Keras.

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Eswari2002/LEARNFLOW.git
    cd LEARNFLOW
    ```

2. Install the required Python libraries:
    ```sh
    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras opencv-python
    ```

### Usage

1. Download and extract the dataset from Kaggle, and place it in the project directory.
2. Run the training script:
    ```sh
    jupyter notebook notebooks/Facial_Emotion_Recognition.ipynb
    ```

### Dataset

The dataset used for this project can be downloaded from Kaggle. It contains images of faces labeled with different emotions.

- [Kaggle Dataset Link](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition?resource=download)

### Model Architecture

The model is built using a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layers
- MaxPooling layers
- Flatten layer
- Dense layers
- Dropout layer

### Results

The model achieves the following results on the validation set:
- Accuracy: 98%
- Loss: 07%
- Precision: 99%
- Recall: 98%
- F1 Score: 98%

### Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

### License

This project is licensed under the MIT License.

## Traffic Sign Recognition

### Problem Statement

Create a model to recognize and classify traffic signs in images for autonomous driving applications.

### Installation

1. Clone the repository (if not already cloned):
    ```sh
    git clone https://github.com/Eswari2002/LEARNFLOW.git
    cd LEARNFLOW
    ```

2. Install the required Python libraries:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Download and extract the dataset from Kaggle, and place it in the `dataset/traffic_sign_recognition` directory.
2. Run the training script:
    ```sh
    jupyter notebook notebooks/Traffic_Sign_Recognition.ipynb
    ```

### Dataset

The dataset used for this project can be downloaded from Kaggle. It contains images of various traffic signs.

- [Kaggle Dataset Link](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification)

### Model Architecture

The model is built using a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layers
- MaxPooling layers
- Flatten layer
- Dense layers
- Dropout layer

### Results

The model achieves the following results on the validation set (update with your results):
- Accuracy: 93%
- Loss: 21%
- Precision: 94%
- Recall: 93%
- F1 Score: 92%

### Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

### License

This project is licensed under the MIT License.
