import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_loader import load_and_split_data

class CNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
             for k in kernel_sizes]
        )
        self.rep_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.emb_dropout(self.embedding(x))
        emb_t = emb.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(emb_t))
            p = torch.max(z, dim=2).values
            pooled.append(p)
        rep = torch.cat(pooled, dim=1)
        rep = self.rep_dropout(rep)
        return self.fc(rep)

def evaluate_transformer(model_path, texts, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
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

def evaluate_cnn(model_path, texts, device, vocab):
    vocab_size = len(vocab) if vocab else 30000 
    model = CNNTextClassifier(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds = []
    return all_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, df_test = load_and_split_data(seed=7)
    
    y_true = df_test['label'].values

    if 'title' in df_test.columns and 'description' in df_test.columns:
        texts_head = df_test['title'].tolist()
        texts_full = (df_test['title'] + " - " + df_test['description']).tolist()
    else:
        texts_full = df_test['text'].tolist()
        texts_head = [str(t).split(" - ", 1)[0] for t in texts_full]

    trans_preds_full = evaluate_transformer("../models/best_transformer", texts_full, device)
    trans_preds_head = evaluate_transformer("../models/best_transformer", texts_head, device)
    
    cnn_preds_full = evaluate_cnn("../models/best_cnn.pth", texts_full, device, vocab=None)
    cnn_preds_head = evaluate_cnn("../models/best_cnn.pth", texts_head, device, vocab=None)
    
    results = []
    
    if trans_preds_full and trans_preds_head:
        results.append({
            "Model": "Transformer",
            "Input": "Headline + Description",
            "Accuracy": accuracy_score(y_true, trans_preds_full),
            "Macro-F1": f1_score(y_true, trans_preds_full, average='macro')
        })
        results.append({
            "Model": "Transformer",
            "Input": "Headline Only",
            "Accuracy": accuracy_score(y_true, trans_preds_head),
            "Macro-F1": f1_score(y_true, trans_preds_head, average='macro')
        })
        
    if cnn_preds_full and cnn_preds_head:
        results.append({
            "Model": "CNN (Assignment 2)",
            "Input": "Headline + Description",
            "Accuracy": accuracy_score(y_true, cnn_preds_full),
            "Macro-F1": f1_score(y_true, cnn_preds_full, average='macro')
        })
        results.append({
            "Model": "CNN (Assignment 2)",
            "Input": "Headline Only",
            "Accuracy": accuracy_score(y_true, cnn_preds_head),
            "Macro-F1": f1_score(y_true, cnn_preds_head, average='macro')
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()