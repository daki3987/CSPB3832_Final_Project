import pandas as pd
import re

def clean_lyrics(lyrics):
    # Remove unwanted characters and extra spaces
    lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove text within square brackets
    lyrics = re.sub(r'\s+', ' ', lyrics)    # Replace multiple spaces with a single space
    lyrics = lyrics.strip()
    return lyrics

def preprocess_data(input_file, output_file):
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Clean lyrics
    df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)
    
    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data('data/lyrics.csv', 'data/processed_lyrics.csv')
