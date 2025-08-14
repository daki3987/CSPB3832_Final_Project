from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd

class LyricsDataset(Dataset):
    def __init__(self, lyrics, labels, tokenizer, max_len):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, item):
        lyric = str(self.lyrics[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            lyric,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    # Load dataset
    df = pd.read_csv('data/processed_lyrics.csv')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = 128
    batch_size = 16

    # Prepare dataset
    dataset = LyricsDataset(
        lyrics=df['cleaned_lyrics'].to_numpy(),
        labels=df['genre'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
    model.train()

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

    # Save model
    model.save_pretrained('models/genre_model')
    tokenizer.save_pretrained('models/genre_model')

if __name__ == "__main__":
    train_model()
