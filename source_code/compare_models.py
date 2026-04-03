"""
In this file, we evaluate our fine-tuned transformer model by manually processing the test data in batches
and calculating its accuracy and Macro-F1 scores. We then generate a visual confusion matrix and print a final comparison table
against our assignment 2 CNN. Parts of the code were taken from the jupyter notebook: transformer_finetuning.ipynb
However, we wrote the transformer evaluation function, which is tailored specifically to the AG news dataset. For this, we looked up some code documentation.
Sources:
Batched Inference & Tokenization: https://shorturl.at/UizeO
2D tensor argmax: https://shorturl.at/kvR6X
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_loader import load_and_split_data

# This function manually evaluates the data, tailored to the AG news dataset
def evaluate_transformer(model_path, texts, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    all_preds = []
    batch_size = 16
    
    #Manual batching logic
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size] # Slice the list of texts to create the current batch
        
        #We apply padding/truncation during inference to match our training configurations
        inputs = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
        
        with torch.no_grad(): # Disable gradient tracking to save VRAM and speed up the test samples
            outputs = model(**inputs)
            logits = outputs.logits
        
        #By specifying dim=1, we find the highest probability class for each individual sentence within the 16-item batch, rather than the highest probability in the entire matrix.
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
        # Accumulate all predictions in a list to allow for Macro-F1 and confusion matrix calculation
        all_preds.extend(predicted_labels)
        
    return all_preds

# This function was re-used from previous assignments
def generate_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title("Transformer Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #We ensure both models are judged on the exact same test split using seed=7
    _, _, df_test = load_and_split_data(seed=7) 
    texts = df_test['text'].tolist()
    y_true = df_test['label'].values
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    #We updated path to load the model from Google Drive
    model_path = "/content/drive/MyDrive/Colab Notebooks/best_transformer"
    
    #Evaluate the new Transformer
    trans_preds = evaluate_transformer(model_path, texts, device)
    
    results = []
    
    #Append Transformer metrics
    results.append({
        "Model": "DistilBERT",
        "Accuracy": accuracy_score(y_true, trans_preds),
        "Macro-F1": f1_score(y_true, trans_preds, average='macro')
    })
    
    #Path to save the image directly to Google Drive
    output_image_path = "/content/drive/MyDrive/Colab Notebooks/trans_cm.png"
    generate_confusion_matrix(y_true, trans_preds, class_names, output_image_path)
        
    #We hardcode the Assignment 2 CNN Baseline
    results.append({
        "Model": "CNN Baseline",
        "Accuracy": 0.9193,
        "Macro-F1": 0.9191
    })
    
    # Output comparison table
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()