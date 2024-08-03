import streamlit as st
import requests
from bs4 import BeautifulSoup
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Load the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

# Define tokenizer and model paths
tokenizer_path = "allenai/scibert_scivocab_uncased"
model_path = "./results_scibert/checkpoint-155"  # Replace with the path to your best checkpoint

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Function to tokenize texts
def tokenize_function(texts, tokenizer):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

# Dataset class for handling tokenized data
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Function to classify text
def classify_text(text, tokenizer, model):
    # Preprocess and tokenize the input text
    encodings = tokenize_function([text], tokenizer)
    
    # Create a dataset for the input text
    input_dataset = TextDataset(encodings)
    
    # Create a Trainer instance for prediction
    training_args = TrainingArguments(
        per_device_eval_batch_size=1,  # Single batch size for prediction
        output_dir='./results',
        evaluation_strategy="no",
        do_train=False,
        do_eval=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args
    )
    
    # Predict the label
    predictions = trainer.predict(input_dataset)
    predicted_label_idx = np.argmax(predictions.predictions, axis=1)[0]
    
    # Convert numerical label back to string label
    predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
    
    return predicted_label

# Streamlit app
def main():
    st.title("Text Classification Streamlit App")
    st.write("Enter a URL to classify the text extracted from it.")
    
    # URL input from user
    url = st.text_input("Enter URL", "")
    
    if url:
        with st.spinner("Processing..."):
            try:
                # Extract text from URL
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                page_text = soup.get_text()
                
                # Classify the extracted text
                predicted_label = classify_text(page_text, tokenizer, model)
                st.write(f"Predicted Label: {predicted_label}")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()


# Assuming `label_encoder` is your fitted LabelEncoder instance
np.save('label_classes.npy', label_encoder.classes_)
