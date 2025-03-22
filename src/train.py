import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_sequence_len = None
        self.total_words = None

    def train_model(self, predictors, label, total_words, max_sequence_len, epochs=20, model_save_path='my_model.h5'):
        self.model = Sequential()
        self.model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
        self.model.add(GRU(150))
        self.model.add(Dense(total_words, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(predictors, label, epochs=epochs, verbose=1)
        self.model.save(model_save_path)

    def load_model(self, model_save_path='my_model.h5'):
        self.model = keras.models.load_model(model_save_path)

    def generate_text(self, seed_text, next_words, max_sequence_len, tokenizer):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted = np.argmax(self.model.predict(token_list, verbose=0), axis=-1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text