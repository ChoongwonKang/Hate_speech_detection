# Hate_speech_detection

This repository contains the code and datasets used in the research paper "Hate Speech Detection through Deep Learning: Dual Approach with Semantic-Sentiment Fusion." The framework integrates semantic analysis using Transformer-based models (BERT) with sentiment analysis tools (VADER and LIWC-22) to develop an effective hate speech detection system.

## Overview
This framework introduces a method for detecting hate speech by fusing semantic features from BERT with sentiment scores from VADER and LIWC-22.

## Steps to Use the Code:
### 1. Data Preprocessing
We utilize four distinct hate speech datasets. To enhance the model's performance, preprocessing is performed on each dataset using the provided code.

### 2. Feature Extraction
After preprocessing, the following steps are conducted:
- Extract the [CLS] token from BERT for semantic representation.
- Compute sentiment scores using VADER and LIWC-22.
- Combine the BERT [CLS] token with the sentiment scores from VADER and LIWC to create input features.

### 3. Model Training and Evaluation
The labels and features are split into training and testing sets to assess the generalization capabilities of the models. We first evaluate linear classifiers using only the BERT [CLS] token.
Next, we combine the input features with deep learning models such as CNN, DNN, and LSTM to measure classification performance. The best-performing combinations for hate speech detection are identified through these experiments.
