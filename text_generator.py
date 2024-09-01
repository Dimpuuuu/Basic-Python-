# Import necessary libraries
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.callbacks import EarlyStopping
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Sample text data
text = """Deep learning is a subset of machine learning, 
           which is essentially a neural network with three or more layers. 
           These neural networks attempt to simulate the behavior of the human brain 
           to “learn” from large amounts of data."""

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Convert the text into a sequence of integers
sequence_data = tokenizer.texts_to_sequences([text])[0]

# Create input-output pairs
sequence_length = 5
sequences = []
for i in range(sequence_length, len(sequence_data)):
    seq = sequence_data[i-sequence_length:i]
    label = sequence_data[i]
    sequences.append((seq, label))

# Split the sequences into input (X) and output (y)
X, y = zip(*sequences)
X = np.array(X)
y = np.array(y)

# One-hot encode the output labels
vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=sequence_length))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
model.fit(X, y, epochs=50, verbose=1, callbacks=[early_stopping])

# Function to generate text
def generate_text(model, tokenizer, input_text, num_words):
    """
    Generate text based on the input text and the trained model.
    
    :param model: Trained Keras model.
    :param tokenizer: Tokenizer fitted on the text data.
    :param input_text: Starting string to generate text from.
    :param num_words: Number of words to generate.
    :return: Generated text as a string.
    """
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += ' ' + output_word
    return input_text

# Test text generation
seed_text = "neural networks"
generated_text = generate_text(model, tokenizer, seed_text, 20)
print("\nGenerated Text:\n", generated_text)
