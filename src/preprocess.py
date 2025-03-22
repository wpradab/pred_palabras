import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = None
        self.max_sequence_len = None
        self.total_words = None

    def load_and_preprocess_text(self, file_path, save_path='preprocessing_data.pkl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([text])
        self.total_words = len(self.tokenizer.word_index) + 1

        input_sequences = []
        for line in text.split('\n'):
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        self.max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre'))

        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

        # Save important variables for later use
        preprocessing_data = {
            'tokenizer': self.tokenizer,
            'max_sequence_len': self.max_sequence_len,
            'total_words': self.total_words
        }
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessing_data, f)

        return predictors, label

    def load_preprocessing_data(self, save_path='preprocessing_data.pkl'):
        with open(save_path, 'rb') as f:
            preprocessing_data = pickle.load(f)
        self.tokenizer = preprocessing_data['tokenizer']
        self.max_sequence_len = preprocessing_data['max_sequence_len']
        self.total_words = preprocessing_data['total_words']