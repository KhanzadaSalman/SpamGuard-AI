import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

# 1. Load & Clean
# Make sure spam.csv is in the SAME folder!
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.iloc[:, [0, 1]] # Ensure we only take the first two columns
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Tokenize
vocab_size = 1000
max_len = 50
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

X = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=max_len)
y = df['label'].values

# 3. CNN Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_len),
    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Starting Training...")
model.fit(X, y, epochs=5, verbose=1)

print(f"\nFinal CNN Accuracy: {model.evaluate(X, y)[1]*100:.2f}%") 

# 4. SAVE THE ASSETS
model.save("cnn_spam_model.keras")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nSuccess! Assets saved: cnn_spam_model.keras & tokenizer.pkl")