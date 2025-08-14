## Emotion Detection and Genre Classification in Song Lyrics

This project uses transformer-based NLP models (BERT, RoBERTa, DistilBERT) to perform:
1. **Emotion Detection** — Classifying lyrics into emotions such as joy, sadness, anger, and fear.
2. **Genre Classification** — Categorizing lyrics into music genres like pop, rock, hip-hop, and country.


## Project Structure
```
.  
.
├── src/                  # Python scripts for data processing and model training
├── results/              # Output figures, confusion matrices, and metrics
├── README.md             # Project description and instructions
└── requirements.txt      # Python dependencies

. . .     
 
```
<hr>
     
 
## Requirements

Install dependencies:

pip install -r requirements.txt


*Key packages:

transformers

torch

scikit-learn

pandas

matplotlib

## How to Run
1. **Preprocess the data **
python src/preprocess.py --input data/lyrics.csv --output data/processed.csv

2. **Train the models**
python src/train_emotion.py --data data/processed.csv --model bert-base-uncased
python src/train_genre.py --data data/processed.csv --model roberta-base

3. **Evaluate**
python src/evaluate.py --model_path models/emotion_model --task emotion
python src/evaluate.py --model_path models/genre_model --task genre

4. **Visualize results**
python src/plot_results.py --results results/metrics.json

## Outputs

Accuracy and F1-score tables

Confusion matrices for each task

Heatmap of emotion distribution by genre
