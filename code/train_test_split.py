import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Train/test split
def train_test_split_and_save(df_path, features_path, train_df_path, test_df_path, train_features_path, test_features_path, stratify_col='encoding'):
    df = pd.read_csv(df_path)
    features = np.load(features_path)
    df_train, df_test, features_train, features_test = train_test_split(
        df, features, test_size=0.2, stratify=df[stratify_col], random_state=42)
    df_train.to_csv(train_df_path, index=False, encoding='utf-8-sig')
    df_test.to_csv(test_df_path, index=False, encoding='utf-8-sig')
    np.save(train_features_path, features_train)
    np.save(test_features_path, features_test)
    print(f"Train and test datasets for {df_path} have been created and saved.")

if __name__ == "__main__":

    train_test_split_and_save(
        df_path='', # input_text
        features_path='', # input_vector
        train_df_path='', # input_text(train)
        test_df_path='', # input_text(test)
        train_features_path='', # input_vector(train)
        test_features_path='' # input_vector(test)
    )