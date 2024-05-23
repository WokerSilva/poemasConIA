import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from transformers import BertTokenizer, BertForMaskedLM
import torch
import json

# Inicializamos el tokenizador y el modelo preentrenado de BERT
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Función para generar texto utilizando BERT
def generar_texto_BERT(verso_inicial, num_palabras):
    # Tokenizamos el verso inicial
    tokens = tokenizer_bert.encode(verso_inicial, add_special_tokens=False, return_tensors='pt')

    # Generamos texto con BERT rellenando los tokens enmascarados
    with torch.no_grad():
        outputs = model_bert.generate(tokens, max_length=num_palabras + len(tokens[0]) - 1, num_return_sequences=1)

    # Decodificamos los tokens generados
    texto_generado = tokenizer_bert.decode(outputs[0][len(tokens[0]) - 1:], skip_special_tokens=True)

    return texto_generado

# Función para cargar poemas por categoría
def cargar_poemas_por_categoria(poemas_limpios_path, categoria_seleccionada):
    with open(poemas_limpios_path, 'r', encoding='utf-8') as file:
        poemas = json.load(file)
    
    poemas_filtrados = [poema['contenido'] for poema in poemas if poema['categoria'] == categoria_seleccionada]
    return poemas_filtrados

def entrenar_rnn_generativa(poemas_filtrados):
    # Inicializamos un tokenizador que convertirá las palabras en números únicos
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(poemas_filtrados)

    # Convertimos los poemas en secuencias de números que representan cada palabra
    secuencias = tokenizer.texts_to_sequences(poemas_filtrados)

    # Calculamos el tamaño del vocabulario basado en el índice de palabras del tokenizador
    longitud_letras = len(tokenizer.word_index) + 1

    # Preparamos los datos de entrenamiento y las etiquetas correspondientes
    X = []  # Lista para almacenar secuencias de entrada
    y = []  # Lista para almacenar la palabra objetivo (etiqueta)
    for secuencia in secuencias:
        for i in range(1, len(secuencia)):
            n_gram_sequence = secuencia[:i + 1]
            # Añadimos la secuencia de entrada.
            X.append(n_gram_sequence[:-1])
            # Añadimos la palabra objetivo.
            y.append(n_gram_sequence[-1])

    # Rellenamos las secuencias para que todas tengan la misma longitud
    max_sequence_len = max([len(seq) for seq in X])
    X = np.array(pad_sequences(X, maxlen=max_sequence_len, padding='pre'))
    y = np.array(y)

    # Utilizamos LabelEncoder para convertir las etiquetas de palabras en números
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Construimos el modelo secuencial de RNN con capas de incrustación (Embedding) y LSTM.
    model = Sequential()
    model.add(Embedding(longitud_letras, 100, input_length=max_sequence_len - 1))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(longitud_letras, activation='softmax'))

    # Compilamos el modelo con una función de pérdida adecuada para clasificación y el optimizador Adam
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenamos el modelo con los datos preparados
    model.fit(X, y, epochs=100, verbose=1)

    # Devolvemos el modelo entrenado junto con el tokenizador, el codificador y la longitud máxima de la secuencia.
    return model, tokenizer, encoder, max_sequence_len

def hacer_predicciones_generativas(modelo_rnn, tokenizer_rnn, max_sequence_len, verso_inicial, num_palabras):
    resultado = []  # Lista para almacenar las palabras generadas.
    verso_actual = verso_inicial

    for _ in range(num_palabras):
        # Convertimos el verso actual en una secuencia de números.
        secuencia = tokenizer_rnn.texts_to_sequences([verso_actual])[0]
        # Rellenamos la secuencia para que tenga la longitud adecuada.
        secuencia = pad_sequences([secuencia], maxlen=max_sequence_len - 1, padding='pre')
        # Realizamos una predicción con el modelo de RNN.
        prediccion = np.argmax(modelo_rnn.predict(secuencia), axis=-1)
        palabra_predicha = ''
        # Buscamos la palabra correspondiente al número predicho.
        for palabra, index in tokenizer_rnn.word_index.items():
            if index == prediccion:
                palabra_predicha = palabra
                break
        # Añadimos la palabra predicha al verso actual y al resultado.
        verso_actual += ' ' + palabra_predicha
        resultado.append(palabra_predicha)

    # Devolvemos el texto generado como una cadena de texto.
    return ' '.join(resultado)

def ejemplo_entrenamiento_generativo(poemas_limpios_path, categoria_seleccionada, verso_inicial, num_palabras):
    # Cargamos los poemas filtrados por la categoría seleccionada
    poemas_filtrados = cargar_poemas_por_categoria(poemas_limpios_path, categoria_seleccionada)
    
    if not poemas_filtrados:
        print(f"No se encontraron poemas en la categoría {categoria_seleccionada}")
        return

    # Entrenamos la RNN generativa con los poemas filtrados
    modelo_rnn_generativa, tokenizer_rnn, encoder_rnn, max_sequence_len = entrenar_rnn_generativa(poemas_filtrados)

    # Utilizamos el modelo entrenado para generar un nuevo poema.
    nuevo_poema = hacer_predicciones_generativas(modelo_rnn_generativa, tokenizer_rnn, max_sequence_len, verso_inicial, num_palabras)
    # Imprimimos el poema generado.
    print(f'Nuevo poema generado: {nuevo_poema}')

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
