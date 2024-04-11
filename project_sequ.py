from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from textblob import Word
from termcolor import colored
from warnings import filterwarnings

# Suppress warnings
filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
train_set = pd.read_csv("train.csv", encoding="utf-8", engine="python", header=0)

# Clean and process dataset
def clean_text(text):
    text = text.lower()
    text = ' '.join(x for x in text.split() if x.isalpha())  # Keep only alphabetic characters
    text = ' '.join(x for x in text.split() if x not in stopwords.words("english"))  # Remove stopwords
    text = ' '.join(Word(word).lemmatize() for word in text.split())  # Lemmatization
    return text

train_set["tweet"] = train_set["tweet"].apply(clean_text)

# Divide datasets
x = train_set["tweet"]
y = train_set["label"]
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=11)

# Vectorize data
vectorizer = CountVectorizer(max_features=5000)  # Limit features to 5000
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x).toarray()
x_val_count = vectorizer.transform(val_x).toarray()

# Build TensorFlow model
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train_count.shape[1],)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(x_train_count, train_y, validation_data=(x_val_count, val_y), epochs=10, batch_size=32, verbose=1)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
