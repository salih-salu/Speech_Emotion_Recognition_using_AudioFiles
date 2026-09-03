# Speech Emotion Recognition Using Deep Learning

## Project Overview

This project focuses on recognizing human emotions from speech using **Deep Learning**. The **RAVDESS Emotional Speech Audio Dataset** was used, containing 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

The project was developed using **Python, Librosa, TensorFlow/Keras, and Scikit-learn**.

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

### CNN + LSTM
A **CNN + LSTM hybrid model** was also implemented. CNN extracts important features, while LSTM processes these features as sequences to learn temporal relationships.

However, the CNN + LSTM model produced lower accuracy. Several **hyperparameter tuning experiments** were performed, including changes to learning rate, batch size, dropout, and LSTM units, but significant improvement could not be achieved.

Therefore, the CNN model was selected as the final model.

## Optimization

The CNN model was further improved through:

- Hyperparameter tuning
- Different feature extraction configurations
- Model architecture adjustments
- Regularization techniques

After these experiments, the best CNN model achieved approximately **54% test accuracy**, which was the highest accuracy obtained during the project.

## Final Result

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
