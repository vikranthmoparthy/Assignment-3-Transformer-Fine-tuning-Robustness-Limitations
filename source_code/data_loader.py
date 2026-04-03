from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_and_split_data(seed=7):
    url = "hf://datasets/sh0416/ag_news/" 
    data_files = {"train": f"{url}train.jsonl","test": f"{url}test.jsonl"} 
    
    dataset = load_dataset("json", data_files=data_files)

    df_train_full = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()

    if 'text' not in df_train_full.columns:
        if 'title' in df_train_full.columns and 'description' in df_train_full.columns:
            df_train_full['text'] = df_train_full['title'] + " - " + df_train_full['description']
            df_test['text'] = df_test['title'] + " - " + df_test['description']
        elif 'Title' in df_train_full.columns and 'Description' in df_train_full.columns:
            df_train_full['text'] = df_train_full['Title'] + " - " + df_train_full['Description']
            df_test['text'] = df_test['Title'] + " - " + df_test['Description']

    df_train, df_dev = train_test_split( 
        df_train_full,
        test_size=0.1,
        random_state=seed,
        stratify=df_train_full['label']  
    )
    return df_train, df_dev, df_test