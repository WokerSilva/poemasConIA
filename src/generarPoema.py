import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from transformers import BertTokenizer, BertForMaskedLM
import torch
import json

# Inicializamos el tokenizador y el modelo preentrenado de BERT
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Conectores comunes en español
conectores = ["y", "pero", "porque", "aunque", "sin embargo", "además", "entonces", "por lo tanto", "luego", "después"]

# Función para generar texto utilizando BERT
def generar_texto_BERT(verso_inicial, num_palabras):
    tokens = tokenizer_bert.encode(verso_inicial, add_special_tokens=False, return_tensors='pt')
    with torch.no_grad():
        outputs = model_bert.generate(tokens, max_length=num_palabras + len(tokens[0]) - 1, num_return_sequences=1)
    texto_generado = tokenizer_bert.decode(outputs[0][len(tokens[0]) - 1:], skip_special_tokens=True)
    return texto_generado

# Función para cargar poemas por categoría
def cargar_poemas_por_categoria(poemas_limpios_path, categoria_seleccionada):
    with open(poemas_limpios_path, 'r', encoding='utf-8') as file:
        poemas = json.load(file)
    poemas_filtrados = [poema['contenido'] for poema in poemas if poema['categoria'] == categoria_seleccionada]
    return poemas_filtrados

# Función para entrenar el modelo RNN generativo
def entrenar_rnn_generativa(poemas_filtrados):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(poemas_filtrados)
    secuencias = tokenizer.texts_to_sequences(poemas_filtrados)
    longitud_letras = len(tokenizer.word_index) + 1

    X, y = [], []
    for secuencia in secuencias:
        for i in range(1, len(secuencia)):
            n_gram_sequence = secuencia[:i + 1]
            X.append(n_gram_sequence[:-1])
            y.append(n_gram_sequence[-1])

    max_sequence_len = max([len(seq) for seq in X])
    X = np.array(pad_sequences(X, maxlen=max_sequence_len, padding='pre'))
    y = np.array(y)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    model = Sequential()
    model.add(Embedding(longitud_letras, 100, input_length=max_sequence_len - 1))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(150))
    model.add(Dense(longitud_letras, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=1)

    return model, tokenizer, encoder, max_sequence_len

# Función para hacer predicciones generativas con el modelo RNN
def hacer_predicciones_generativas(modelo_rnn, tokenizer_rnn, max_sequence_len, verso_inicial, num_palabras):
    resultado = []
    verso_actual = verso_inicial

    for i in range(num_palabras):
        secuencia = tokenizer_rnn.texts_to_sequences([verso_actual])[0]
        secuencia = pad_sequences([secuencia], maxlen=max_sequence_len - 1, padding='pre')
        prediccion = np.argmax(modelo_rnn.predict(secuencia), axis=-1)
        palabra_predicha = ''
        for palabra, index in tokenizer_rnn.word_index.items():
            if index == prediccion:
                palabra_predicha = palabra
                break
        if i % 5 == 0 and i != 0:  # Añadir un conector cada 5 palabras
            palabra_predicha = random.choice(conectores)
        verso_actual += ' ' + palabra_predicha
        resultado.append(palabra_predicha)

    return ' '.join(resultado)

# Función principal para entrenar el modelo generativo y generar un poema
def ejemplo_entrenamiento_generativo(poemas_limpios_path, categoria_seleccionada, verso_inicial, num_palabras):
    poemas_filtrados = cargar_poemas_por_categoria(poemas_limpios_path, categoria_seleccionada)
    
    if not poemas_filtrados:
        print(f"No se encontraron poemas en la categoría {categoria_seleccionada}")
        return

    modelo_rnn_generativa, tokenizer_rnn, encoder_rnn, max_sequence_len = entrenar_rnn_generativa(poemas_filtrados)
    nuevo_poema = hacer_predicciones_generativas(modelo_rnn_generativa, tokenizer_rnn, max_sequence_len, verso_inicial, num_palabras)
    print(f'Nuevo poema generado: {nuevo_poema}')

# Función main para ejecutar el programa
def main():
    poemas_limpios_path = 'poemas_limpios.json'
    categoria_seleccionada = input("Seleccione una categoría (amor/tristeza): ").strip().lower()
    
    if categoria_seleccionada not in ["amor", "tristeza"]:
        print("Categoría no válida. Por favor seleccione 'amor' o 'tristeza'.")
        return
    
    verso_inicial = input("Ingrese el verso inicial del poema: ").strip()
    num_palabras = int(input("Ingrese el número de palabras a generar: ").strip())

    ejemplo_entrenamiento_generativo(poemas_limpios_path, categoria_seleccionada, verso_inicial, num_palabras)

if __name__ == "__main__":
    main()
