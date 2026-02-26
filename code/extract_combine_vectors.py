import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Load tokenizer and model
def load_tokenizer_and_model(model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model.to(device)
    return tokenizer, model, device

# Extract BERT [CLS] tokens
def extract_cls_tokens(texts, tokenizer, model, device, save_path):
    cls_tokens = []
    for text in tqdm(texts, desc="Extracting CLS tokens"):
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=64, padding='max_length', truncation=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        cls_tokens.append(cls_token)
    cls_tokens_array = np.vstack(cls_tokens)
    np.save(save_path, cls_tokens_array)
    return cls_tokens_array

# Extract combined embeddings (BERT [CLS] + VADER)
def extract_combined_embeddings(texts, tokenizer, model, device, analyzer, save_path):
    combined_embeddings = []
    for text in tqdm(texts, desc="Extracting combined embeddings"):
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=64, padding='max_length', truncation=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu()
        sentiment_score = analyzer.polarity_scores(text)['compound']
        sentiment_tensor = torch.tensor([[sentiment_score]], dtype=torch.float32)
        combined_embedding = torch.cat((cls_embedding, sentiment_tensor), dim=1).numpy()
        combined_embeddings.append(combined_embedding)
    combined_embeddings_array = np.vstack(combined_embeddings)
    np.save(save_path, combined_embeddings_array)
    return combined_embeddings_array

# Combine BERT [CLS] with LIWC 
def combine_with_liwc_vectors(liwc_path, cls_vector_path, save_path):
    df = pd.read_csv(liwc_path)
    df2 = np.load(cls_vector_path)
    df = df.drop(columns=['A', 'B', 'Segment'])
    dff = df.index[0]
    df = df.drop(dff)
    df = df.to_numpy()
    final = np.concatenate((df2, df), axis=1)
    np.save(save_path, final)
    return final

# Process each dataset
def process_dataset(file_path, save_cls_path, save_combined_path, liwc_path, save_liwc_combined_path):

    df = pd.read_csv(file_path)
    tokenizer, model, device = load_tokenizer_and_model()
    extract_cls_tokens(df['prepro'], tokenizer, model, device, save_cls_path)
    analyzer = SentimentIntensityAnalyzer()
    extract_combined_embeddings(df['prepro'], tokenizer, model, device, analyzer, save_combined_path)
    combine_with_liwc_vectors(liwc_path, save_cls_path, save_liwc_combined_path)

if __name__ == "__main__":
    tqdm.pandas()
    
    # CMSB Dataset
    process_dataset(
        file_path='cmsb_input.csv',
        save_cls_path='cmsb_cls_vector.npy',
        save_combined_path='cmsb_cls+vader.npy',
        liwc_path='cmsb_liwc.csv',
        save_liwc_combined_path='cmsb_cls+liwc.npy'
    )
    
    # HSOL Dataset
    process_dataset(
        file_path='Davidson_input.csv',
        save_cls_path='Davidson_cls_vector.npy',
        save_combined_path='Davidson_cls+vader.npy',
        liwc_path='Davidson_liwc.csv',
        save_liwc_combined_path='Davidson_cls+liwc.npy'
    )
    
    # SM Dataset
    process_dataset(
        file_path='supremacist_input.csv',
        save_cls_path='supremacist_cls_vector.npy',
        save_combined_path='supremacist_cls+vader.npy',
        liwc_path='supremacist_liwc.csv',
        save_liwc_combined_path='supremacist_cls+liwc.npy'
    )
    
    # TRAC Dataset
    process_dataset(
        file_path='TRAC_input.csv',
        save_cls_path='TRAC_cls_vector.npy',
        save_combined_path='TRAC_cls+vader.npy',
        liwc_path='TRAC_liwc.csv',
        save_liwc_combined_path='TRAC_cls+liwc.npy'
    )
