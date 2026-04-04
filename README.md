This repository contains the code for fine-tuning a DistilBERT transformer model on the AG News dataset. 
It has the following structure:

.
```text
├── outputs/
│   ├── trans_cm.png
│   └── transformer_errors.csv
├── source_code/
│   ├── __pycache__/
│   ├── compare_models.py
│   ├── data_loader.py
│   ├── error_analysis.py
│   ├── fine_tuning.py
│   ├── sensitivity_run.py
│   └── stress_test.py
├── .gitignore
├── README.md
└── requirements.txt
```

Each file has separate functionality.

    compare_models.py: Evaluates the fine-tuned model against Assignment 2's CNN and generates a confusion matrix.

    data_loader.py: Mainly for downloading the dataset from Hugging Face, doing some text preprocessing and stratified splitting

    error_analysis.py: Identifies and exports misclassified examples to a CSV.

    sensitivity_run.py: Iteratively trains the model on subsets of the data (25%, 50%, 100%)

    stress_test.py: Tests the model's performance on "Headline + Description" versus "Headline Only"

Each of these files can be separately by navigating into the folder and running:

    python "file_name".py