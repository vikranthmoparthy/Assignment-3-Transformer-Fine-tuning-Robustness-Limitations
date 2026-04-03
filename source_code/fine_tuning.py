import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from data_loader import load_and_split_data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_train, df_dev, df_test = load_and_split_data(seed=7)

    hf_train = Dataset.from_pandas(df_train)
    hf_dev = Dataset.from_pandas(df_dev)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_seq_length = 256 

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)

    tokenized_train = hf_train.map(tokenize_function, batched=True)
    tokenized_dev = hf_dev.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",             
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,       
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

    trainer.save_model("../models/best_transformer")
    tokenizer.save_pretrained("../models/best_transformer")

if __name__ == "__main__":
    main()