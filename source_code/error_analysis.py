"""
In this file, we reload our model and conduct classification on the test set, this time comparing DistillBert's prediction to the ground truth.
This is for the error analysis section.
Most of this code is recycled either from earlier scripts of this assignment or from previous assignments.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_loader import load_and_split_data

def evaluate_transformer(model_path, texts, device): #Same function reused from compare_models.py
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    all_preds = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(predicted_labels)
        
    return np.array(all_preds)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, df_test = load_and_split_data(seed=7) #Same test split
    texts = df_test['text'].values
    y_true = df_test['label'].values
    
    class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}     #Class names for the CSV
    
    model_path = "/content/drive/MyDrive/Colab Notebooks/best_transformer"
    y_pred = evaluate_transformer(model_path, texts.tolist(), device)
    

    errors_mask = y_true != y_pred #Find mismatches
    
    error_texts = texts[errors_mask]
    error_true = y_true[errors_mask]
    error_pred = y_pred[errors_mask]
    
    error_results = []     #Compile into a readable DataFrame

    for i in range(min(50, len(error_texts))):     #We take first 50 errors, so we can choose 10-15 from them
        error_results.append({
            "Text": error_texts[i],
            "True Label": class_names[error_true[i]],
            "Predicted Label": class_names[error_pred[i]]
        })
        
    df_errors = pd.DataFrame(error_results)
    
    output_csv_path = "/content/drive/MyDrive/Colab Notebooks/transformer_errors.csv"     #Save to Google Drive as a CSV
    df_errors.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()