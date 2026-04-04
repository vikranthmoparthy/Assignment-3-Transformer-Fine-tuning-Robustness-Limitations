"""
In this file, we iteratively fine-tune a new DistilBERT model on three different sizes of training data (25%, 50%, and 100%)
We generate a clear trendline of Accuracy and Macro-F1 scores. Again most of the functionality was taken from the jupyter notebook: 
transformer_finetuning.ipynb. Additionally, the evaluate_model function was taken from another script. We also referred to some 
additional documentation. We only consider 10000 data points, rather than the 108000, since we can save computing power and 
illustrate the same general trend.
Sources:
Fractional Data Subsetting: https://shorturl.at/VdWeH
"""

import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score 
from data_loader import load_and_split_data

def evaluate_model(model, tokenizer, texts, device):  
    model.eval()
    all_preds = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy() 
        all_preds.extend(predicted_labels)
    return all_preds

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_train_full, df_dev, df_test = load_and_split_data(seed=7)

    df_train_full = df_train_full.sample(n=10000, random_state=7)
    
    hf_dev = Dataset.from_pandas(df_dev)
    test_texts = df_test['text'].tolist()
    y_true = df_test['label'].values
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        
    tokenized_dev = hf_dev.map(tokenize_function, batched=True)
    
    #We define the fractions of the training set to test (25%, 50%, 100%)
    fractions = [0.25, 0.50, 1.0]
    results = []
    
    #We loop through each fraction to run a complete and isolated training cycle
    for frac in fractions:
        df_train_sub = df_train_full.sample(frac=frac, random_state=7)
        hf_train = Dataset.from_pandas(df_train_sub)
        tokenized_train = hf_train.map(tokenize_function, batched=True)
        
        #We initialize the model inside the loop, which ensures we start with a blank pre-trained model every time,
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
        
        training_args = TrainingArguments( 
            output_dir=f"./results_{frac}",
            eval_strategy="epoch",
            save_strategy="epoch",             
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=2,
            weight_decay=0.01,
            load_best_model_at_end=True,      
            fp16=True,
            metric_for_best_model="eval_loss"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        
        preds = evaluate_model(model, tokenizer, test_texts, device)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')
        
        results.append({
            "Training_Size": f"{int(frac*100)}%",
            "Accuracy": acc,
            "Macro-F1": f1
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()