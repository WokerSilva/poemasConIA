import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo RNN, el tokenizador y el codificador desde archivos almacenados
def cargar_modelo_y_datos(modelo_path, tokenizer_path, encoder_path):
    modelo = load_model(modelo_path)  # Cargar el modelo RNN entrenado desde el archivo
    tokenizer = cargar_tokenizer(tokenizer_path)  # Cargar el tokenizador desde el archivo
    encoder = cargar_encoder(encoder_path)  # Cargar el codificador desde el archivo
    return modelo, tokenizer, encoder  # Devolver el modelo, el tokenizador y el codificador

# Función para muestreo con top-k, selecciona una palabra basada en las k más probables
def top_k_sampling(prediccion, k=10):
    indices = np.argsort(prediccion)[-k:]  # Obtener los índices de las k palabras más probables
    valores = prediccion[indices]  # Obtener los valores de probabilidad de esas palabras
    valores = valores / np.sum(valores)  # Normalizar los valores para que sumen 1
    return np.random.choice(indices, p=valores)  # Elegir aleatoriamente una palabra basada en las probabilidades

# Función para generar un verso
def generar_verso(modelo, tokenizer, encoder, longitud_verso, k=10):
    palabra_inicial = random.choice(list(tokenizer.word_index.keys()))  # Elegir una palabra inicial aleatoria
    verso_generado = [palabra_inicial]  # Iniciar el verso con la palabra inicial

    for _ in range(longitud_verso - 1):  # Generar palabras hasta alcanzar la longitud del verso
        secuencia = tokenizer.texts_to_sequences([verso_generado])  # Convertir el verso generado en una secuencia de índices
        secuencia_padded = pad_sequences(secuencia, maxlen=modelo.input_shape[1])  # Rellenar la secuencia para que tenga la longitud esperada por el modelo
        prediccion = modelo.predict(secuencia_padded, verbose=0)[0]  # Obtener la predicción del modelo
        palabra_idx = top_k_sampling(prediccion, k)  # Seleccionar la siguiente palabra usando top-k sampling
        palabra_generada = tokenizer.index_word.get(palabra_idx, '')  # Obtener la palabra correspondiente al índice
        if palabra_generada:
            verso_generado.append(palabra_generada)  # Añadir la palabra generada al verso

    return ' '.join(verso_generado).capitalize()  # Devolver el verso generado como una cadena de texto

# Función para generar una estrofa
def generar_estrofa(modelo, tokenizer, encoder, num_versos, longitud_verso, k=10):
    estrofa = [generar_verso(modelo, tokenizer, encoder, longitud_verso, k) for _ in range(num_versos)]  # Generar varios versos
    return '\n'.join(estrofa)  # Unir los versos con saltos de línea para formar una estrofa

# Función para generar un poema
def generar_poema(modelo, tokenizer, encoder, estructura, k=10):
    poema = [generar_estrofa(modelo, tokenizer, encoder, num_versos, longitud_verso, k) for num_versos, longitud_verso in estructura]  # Generar estrofas según la estructura dada
    return '\n\n'.join(poema)  # Unir las estrofas con doble salto de línea para formar el poema completo

# Extraer las palabras más relevantes del tokenizador
def extraer_palabras_relevantes(tokenizer, top_n=20):
    palabras_relevantes = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]  # Ordenar las palabras por frecuencia y tomar las top_n
    return [palabra for palabra, _ in palabras_relevantes]  # Devolver solo las palabras sin las frecuencias

# Guardar el tokenizador en un archivo
def guardar_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as file:
        np.save(file, tokenizer.word_index)  # Guardar el índice de palabras del tokenizador en un archivo

# Cargar el tokenizador desde un archivo
def cargar_tokenizer(file_path):
    with open(file_path, 'rb') as file:
        word_index = np.load(file, allow_pickle=True).item()  # Cargar el índice de palabras desde el archivo
    tokenizer = Tokenizer()  # Crear una instancia del tokenizador
    tokenizer.word_index = word_index  # Asignar el índice de palabras al tokenizador
    tokenizer.index_word = {v: k for k, v in word_index.items()}  # Invertir el índice de palabras para facilitar la conversión de índices a palabras
    return tokenizer  # Devolver el tokenizador

# Guardar el codificador en un archivo
def guardar_encoder(encoder, file_path):
    with open(file_path, 'wb') as file:
        np.save(file, encoder.classes_)  # Guardar las clases del codificador en un archivo

# Cargar el codificador desde un archivo
def cargar_encoder(file_path):
    with open(file_path, 'rb') as file:
        classes = np.load(file, allow_pickle=True)  # Cargar las clases desde el archivo
    encoder = LabelEncoder()  # Crear una instancia del codificador
    encoder.classes_ = classes  # Asignar las clases al codificador
    return encoder  # Devolver el codificador
