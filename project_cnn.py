import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import Word
nltk.download('wordnet')

from termcolor import colored
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn import set_config
set_config(print_changed_only=False)

print(colored("\nLIBRARIES WERE SUCCESSFULLY IMPORTED...", color="green", attrs=["dark", "bold"]))

# Load datasets
train_set = pd.read_csv("train.csv",
                        encoding="utf-8",
                        engine="python",
                        header=0)

test_set = pd.read_csv("test.csv",
                       encoding="utf-8",
                       engine="python",
                       header=0)

print(colored("\nDATASETS WERE SUCCESSFULLY LOADED...", color="green", attrs=["dark", "bold"]))

# Clean and process train set
train_set["tweet"] = train_set["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
train_set["tweet"] = train_set["tweet"].str.replace('[^\w\s]', '')
train_set['tweet'] = train_set['tweet'].str.replace('\d', '')
sw = stopwords.words("english")
train_set['tweet'] = train_set['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
train_set['tweet'] = train_set['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Clean and process test set
test_set["tweet"] = test_set["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
test_set["tweet"] = test_set["tweet"].str.replace('[^\w\s]', '')
test_set['tweet'] = test_set['tweet'].str.replace('\d', '')
test_set['tweet'] = test_set['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
test_set['tweet'] = test_set['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_set["tweet"])

vocab_size = len(tokenizer.word_index) + 1

max_len = 100  # Max sequence length
embedding_dim = 100

train_sequences = tokenizer.texts_to_sequences(train_set["tweet"])
test_sequences = tokenizer.texts_to_sequences(test_set["tweet"])

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

# Build CNN model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_padded, train_set['label'], epochs=10, batch_size=64, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])

# Generate predictions on test set
predictions = model.predict(test_padded)

# Save predictions
test_set['prediction'] = predictions
test_set.to_csv('test_with_predictions.csv', index=False)

print(colored("\nPredictions saved to 'test_with_predictions.csv'.", color="green", attrs=["bold"]))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training History')
plt.legend()
plt.show()
