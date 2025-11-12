from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

import pandas as pd
import numpy as np
import re

# Preprocessing data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_data(file_path,num_words=10000,seq_lengths=[25,50,100]):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive':1, 'negative':0})

    # Tokenize the text
    tokenizer = Tokenizer(num_words=num_words, oov_token = "<OOV>")
    tokenizer.fit_on_texts(df['review'])


    data={}
    # Sequencing and padding
    for length in seq_lengths:
        sequences = tokenizer.texts_to_sequences(df['review'])
        padded = pad_sequences(sequences, maxlen = length, padding='post', truncating = 'post')
        data[length] = (padded, df['label'].values)
    return tokenizer,data


if __name__=="__main__":
    tokenizer, data = prepare_data('C:/Users/dharm/sentiment_rnn_project/data/IMDB Dataset.csv')
    print({k: v[0].shape for k,v in data.items()})
