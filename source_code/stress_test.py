"""
In this script, we perform an input field stress test by evaluating the DistilBERT model on two separate slices of test data:
full articles (headline + description) and headline only. All functionality was taken from other existing scripts or from 
transformer_finetuning.ipynb
"""

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_loader import load_and_split_data

def evaluate_transformer(model_path, texts, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    all_preds = []
    batch_size = 32 #Increased to 32 for faster T4 processing
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #We load the exact same test split
    _, _, df_test = load_and_split_data(seed=7)
    y_true = df_test['label'].values

    #Extract the two different data slices
    if 'title' in df_test.columns and 'description' in df_test.columns:
        texts_head = df_test['title'].tolist()
        texts_full = (df_test['title'] + " - " + df_test['description']).tolist()
    else:
        texts_full = df_test['text'].tolist()
        texts_head = [str(t).split(" - ", 1)[0] for t in texts_full]

    model_path = "/content/drive/MyDrive/Colab Notebooks/best_transformer" 
    
    trans_preds_full = evaluate_transformer(model_path, texts_full, device)

    trans_preds_head = evaluate_transformer(model_path, texts_head, device)
    
    results = [     #Final comparison table
        {
            "Model": "DistilBERT",
            "Input Slice": "Headline + Description",
            "Accuracy": accuracy_score(y_true, trans_preds_full),
            "Macro-F1": f1_score(y_true, trans_preds_full, average='macro')
        },
        {
            "Model": "DistilBERT",
            "Input Slice": "Headline Only",
            "Accuracy": accuracy_score(y_true, trans_preds_head),
            "Macro-F1": f1_score(y_true, trans_preds_head, average='macro')
        }
    ]
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()