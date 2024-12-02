# Importing Libraries
import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dropout, Dense, Embedding, LSTM
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Loading datasets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Preprocessing text
def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Decoding reviews
x_train_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in x_train]
x_test_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in x_test]

x_train_text = [preprocess_text(text) for text in x_train_text]
x_test_text = [preprocess_text(text) for text in x_test_text]

# Tokenize and Pad Sequence
maxlen = 200
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train_text)

x_train_seq = tokenizer.texts_to_sequences(x_train_text)
x_test_seq = tokenizer.texts_to_sequences(x_test_text)

x_train = pad_sequences(x_train_seq, maxlen=maxlen)
x_test = pad_sequences(x_test_seq, maxlen=maxlen)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Model Building and Compiling
n_unique_words = 10000
model = Sequential()
model.add(Embedding(n_unique_words, 64, input_length=maxlen))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Plotting accuracy and loss
from matplotlib import pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Loss vs Accuracy')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy', 'Val_Loss', 'Val_Accuracy'], loc='upper right')
plt.show()

# Making Predictions
sample_text = "This is a great movie with fantastic performances!"
sample_text = preprocess_text(sample_text)
tokenized_sample = tokenizer.texts_to_sequences([sample_text])
padded_sample = pad_sequences(tokenized_sample, maxlen=maxlen)
prediction = model.predict(padded_sample)
threshold = 0.5

if prediction[0][0] > threshold:
    print(f"The sample text is predicted as positive with confidence: {prediction[0][0]}")
else:
    print(f"The sample text is predicted as negative with confidence: {1 - prediction[0][0]}")