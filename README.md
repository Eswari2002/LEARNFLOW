# LEARNFLOW - Facial Emotion Recognition

This project aims to develop a model that recognizes facial emotions (e.g., happy, sad, angry) from images or videos. The model is built using deep learning techniques with TensorFlow and Keras.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Eswari2002/LEARNFLOW.git
    cd LEARNFLOW
    ```

2. Install the required Python libraries:
    ```sh
    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras opencv-python
    ```

## Usage

1. Download and extract the dataset from Kaggle, and place it in the project directory.
2. Run the training script:
    ```sh
    Jupyter Notebook Facial_Emotion_Recognition.ipynb
    ```

## Dataset

The dataset used for this project can be downloaded from Kaggle. It contains images of faces labeled with different emotions.

- [Kaggle Dataset Link](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition?resource=download)

## Model Architecture

The model is built using a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layers
- MaxPooling layers
- Flatten layer
- Dense layers
- Dropout layer

## Results

The model achieves the following results on the validation set:
- Accuracy: 98%
- Loss: 07%
- Precision: 99%
- recall: 98%
- F1 score: 98%

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.
