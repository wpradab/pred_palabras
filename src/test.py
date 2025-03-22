from preprocess import TextPreprocessor
from train import ModelTrainer

load = TextPreprocessor.load_and_preprocess_text("C:/Users/Dell/Documents/Germana_cap1.txt")

model_trainer = ModelTrainer()
model_trainer.load_model()  # Load the saved model
text_preprocessor = TextPreprocessor()
text_preprocessor.load_preprocessing_data()
generated_text = model_trainer.generate_text("Hola, voy a", 10, text_preprocessor.max_sequence_len, text_preprocessor.tokenizer)
print(generated_text)