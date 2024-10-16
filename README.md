# Hate_speech_detection

This repository contains the code and datasets used in the research paper "Hate Speech Detection through Deep Learning: Dual Approach with Semantic-Sentiment Fusion." The framework integrates semantic analysis using Transformer-based models (BERT) with sentiment analysis tools (VADER and LIWC-22) to develop an effective hate speech detection system.

### Core Libraries 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=NumPy&logoColor=white"> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white"> <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"> 

**Python** == `3.11.5`

**PyTorch** == `2.4.0`

**NumPy** == `1.26.4`

**Pandas** == `2.2.2`

**scikit-learn** == `1.4.2`

## Overview
This framework introduces a method for detecting hate speech by fusing semantic features from BERT with sentiment scores from VADER and LIWC-22, and demonstrates the combination with the best performance.

## Steps to Use the Code
### 1. Data Preprocessing
We utilize four distinct hate speech datasets
- To enhance the model's performance, preprocessing is performed on each dataset using the provided code.
- Raw data is stored in each subfolder of the [`data`](./data) with the format 'dataname_raw'.
- Preprocessed data can be found in each subfolder of the  [`data`](./data) with the format 'dataname_input'. 
Use the code in the [`preprocess`](code/preprocess) for the respective datasets.

### 2. Feature Extraction
After preprocessing, the following steps are conducted on each input data:
- Extract the [CLS] token from BERT for semantic representation.
- Compute sentiment scores using VADER and LIWC-22. (LIWC scored data can be found in each subfolder of the  [`data`](./data) with the format 'dataname_liwc'.
- Combine the BERT [CLS] token with the sentiment scores from VADER and LIWC to create input features.
- Through the process, the input features can be saved in the form of .npy files.
Codes are available by using [`extract_combine_vectors.py`](code/extract_combine_vectors.py).

### 3. Model Training and Evaluation
The labels and features are split into training and testing sets to assess the generalization capabilities of the models: 
- First, train each model with the split training input features/labels using linear, CNN, DNN, and LSTM models.
- During training, the best weight for each model (.pth) will be automatically saved.
- Next, test the models using the best weights with the split testing input features/labels and evaluate each model's performance.
- The best-performing combinations for hate speech detection are identified through these experiments.
Codes are available in [`modeling`](code/modeling).

The [`pth`](./pth) folder contains the trained model weights for each dataset. By applying these weights directly to the test code of the corresponding deep learning models, you can immediately evaluate the model's performance without retraining models.

### Data Source
CMSB: https://search.gesis.org/research_data/SDN-10.7802-2251

SM: https://github.com/Vicomtech/hate-speech-dataset 

TRAC: https://sites.google.com/view/trac2/home

HSOL: https://github.com/t-davidson/hate-speech-and-offensive-language
