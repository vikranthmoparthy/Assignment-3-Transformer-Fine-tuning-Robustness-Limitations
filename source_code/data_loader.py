"""
This file loads the AG news dataset from Huggingface, converts it to pandas DataFrame and splits the data.
In writing this code, we looked up some documentation, which is listed below:
Sources:
    HuggingFace dataset loading: https://shorturl.at/nLkoe
    HuggingFace dataset pandas conversion: https://shorturl.at/7oBMa
    Stratified Splitting: https://shorturl.at/7FIUj
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_and_split_data(seed=7):
    url = "hf://datasets/sh0416/ag_news/" #File path for the ag news dataset.

    data_files = {"train": f"{url}train.jsonl","test": f"{url}test.jsonl"} 

    #Here we use Huggingface library to load jsonl files
    dataset = load_dataset("json", data_files=data_files)

    #We convert the dataset to pandas dataframe, which we know how to work with.
    df_train_full = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()

    df_train, df_dev = train_test_split( #We use a a 90/10 validation split
        df_train_full,
        test_size=0.1,
        random_state=seed,
        stratify=df_train_full['label']  #Ensures that the proportion of each class in label is preserved in the split.
    )
    return df_train, df_dev, df_test