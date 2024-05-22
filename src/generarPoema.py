import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def cargar_modelo_y_datos(modelo_path, tokenizer_path, encoder_path):
    modelo = load_model(modelo_path)
    tokenizer = cargar_tokenizer(tokenizer_path)
    encoder = cargar_encoder(encoder_path)
    return modelo, tokenizer, encoder

def generar_poema(modelo, tokenizer, encoder, longitud_poema=50):
    palabra_inicial = random.choice(list(tokenizer.word_index.keys()))
    poema_generado = [palabra_inicial]

    for _ in range(longitud_poema - 1):
        secuencia = tokenizer.texts_to_sequences([poema_generado])
        secuencia_padded = pad_sequences(secuencia, maxlen=modelo.input_shape[1])
        prediccion = modelo.predict(secuencia_padded, verbose=0)
        palabra_idx = np.argmax(prediccion, axis=1)
        palabra_generada = encoder.inverse_transform(palabra_idx)[0]
        poema_generado.append(palabra_generada)

    return ' '.join(poema_generado)

def ajustar_gpt2(texto_inicial, modelo_gpt2):
    pass

def extraer_palabras_relevantes(tokenizer, top_n=20):
    palabras_relevantes = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [palabra for palabra, _ in palabras_relevantes]

def guardar_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as file:
        np.save(file, tokenizer.word_index)

def cargar_tokenizer(file_path):
    with open(file_path, 'rb') as file:
        word_index = np.load(file, allow_pickle=True).item()
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    return tokenizer

def guardar_encoder(encoder, file_path):
    with open(file_path, 'wb') as file:
        np.save(file, encoder.classes_)

def cargar_encoder(file_path):
    with open(file_path, 'rb') as file:
        classes = np.load(file, allow_pickle=True)
    encoder = LabelEncoder()
    encoder.classes_ = classes
    return encoder