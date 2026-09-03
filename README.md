# Speech Emotion Recognition Using Deep Learning

## Project Overview

This project focuses on recognizing human emotions from speech using **Deep Learning**. The **RAVDESS Emotional Speech Audio Dataset** was used, containing 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

The project was developed using **Python, Librosa, TensorFlow/Keras, and Scikit-learn**.

## Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) Emotional Speech Audio Dataset**.

The speech dataset contains **24 actors** and **8 emotion classes**:

| Code | Emotion |
|---|---|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

The dataset was divided based on actors so that the testing data contains speakers that were not used during training.

**Dataset:**  
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## Feature Extraction

Audio files were preprocessed and meaningful speech features were extracted using **Librosa**.

Different feature extraction approaches were experimented with, including:

- MFCC
- Chroma
- Spectral Contrast
- Tonnetz

Different configurations were tested to identify the best-performing feature representation.

## Models

Two deep learning approaches were explored:

### CNN
A **Convolutional Neural Network (CNN)** was used to learn important patterns from the extracted audio features and classify the speech into 8 emotion categories.

The model includes:
- Convolutional layers
- Max Pooling
- Flatten layer
- Dropout
- Dense layers
- Softmax output layer

### CNN + LSTM
A **CNN + LSTM hybrid model** was also implemented. CNN extracts important features, while LSTM processes these features as sequences to learn temporal relationships.

However, the CNN + LSTM model produced lower accuracy. Several **hyperparameter tuning experiments** were performed, including changes to learning rate, batch size, dropout, and LSTM units, but significant improvement could not be achieved.
## CNN + LSTM Hyperparameter Tuning

The following experiments were performed to improve the CNN + LSTM model:

| Exp. | Learning Rate | Batch | LSTM Units | Dropout | Test Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.001 | 32 | 128 | 0.3 | **40.6%** |
| 2 | 0.001 | 32 | 128 | 0.2 | 36% |
| 3 | 0.001 | 32 | 128 | 0.5 | 37% |
| 4 | 0.0005 | 32 | 128 | 0.3 | 39% |
| 5 | 0.0001 | 32 | 128 | 0.3 | 24% |
| 6 | Best | 16 | 128 | 0.3 | 36% |
| 7 | Best | 64 | 128 | 0.3 | 33% |
| 8 | Best | Best | 64 | 0.3 | 33% |
| 9 | Best | Best | 256 | 0.3 | 35% |
| 10 | 0.0005 | 32 | 64 | 0.3 | 27% |

**Best CNN + LSTM Test Accuracy: 40.6%**

## Optimization

The CNN model was further improved through:

- Hyperparameter tuning
- Different feature extraction configurations
- Model architecture adjustments
- Regularization techniques

After these experiments, the best CNN model achieved approximately **54% test accuracy**, which was the highest accuracy obtained during the project.

## Final Result

| Model | Best Test Accuracy |
|---|---:|
| CNN + LSTM | 40.6% |
| **CNN (Final Model)** | **~54%** |

**Best Model:** CNN  
**Test Accuracy:** ~54%  
**Number of Classes:** 8


## Technologies

- Python
- Google Colab
- Librosa
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
