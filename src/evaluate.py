from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
from sklearn.metrics import accuracy_score, classification_report
import torch
import pandas as pd

def evaluate_model(model_type='emotion'):
    # Load dataset
    df = pd.read_csv('data/test_lyrics.csv')
    if model_type == 'emotion':
        model_path = 'models/emotion_model'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    else:
        model_path = 'models/genre_model'
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Prepare dataset
    inputs = tokenizer(df['cleaned_lyrics'].tolist(), padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(df['label'].tolist())

    # Evaluate model
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    evaluate_model(model_type='emotion')
    evaluate_model(model_type='genre')
