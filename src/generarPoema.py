import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


"""
Función para entrenar una red neuronal recurrente (RNN) que genera texto a 
partir de los poemas preprocesados.

Recibe:
    poemas_limpios_path (str): Ruta del archivo que contiene los poemas ya limpios y preprocesados.

Devuelve:
    tuple: Tupla que contiene el modelo RNN entrenado, el tokenizador y el 
     codificador utilizados en el entrenamiento.
"""
def entrenar_rnn_generativa(poemas_limpios_path):
    # Abrimos el archivo que contiene los poemas ya limpios y preprocesados
    with open(poemas_limpios_path, 'r', encoding='utf-8') as file:
        poemas_limpios = file.readlines()

    # Inicializamos un tokenizador que convertirá las palabras en números unicos
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(poemas_limpios)

    # Convertimos los poemas en secuencias de números que representan cada palabra
    secuencias = tokenizer.texts_to_sequences(poemas_limpios)

    # Calculamos el tamaño del vocabulario basado en el indice de palabras del tokenizador
    vocab_size = len(tokenizer.word_index) + 1

    # Preparamos los datos de entrenamiento y las etiquetas correspondientes
    X = []  # Lista para almacenar secuencias de entrada
    y = []  # Lista para almacenar la palabra objetivo (etiqueta)
    for secuencia in secuencias:
        for i in range(1, len(secuencia)):
            n_gram_sequence = secuencia[:i+1]
            # Añadimos la secuencia de entrada.
            X.append(n_gram_sequence[:-1]) 
            # Añadimos la palabra objetivo. 
            y.append(n_gram_sequence[-1])  

    # rellenamos las secuencias para que todas tengan la misma longitud
    max_sequence_len = max([len(seq) for seq in X])
    X = np.array(pad_sequences(X, maxlen=max_sequence_len, padding='pre'))
    y = np.array(y)

    # Utilizamos LabelEncoder para convertir las etiquetas de palabras en números
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Construimos el modelo secuencial de RNN con capas de incrustación (Embedding) y LSTM.
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))

    # Compilamos el modelo con una función de perdida adecuada para 
    #  clasificación y el optimizador Adam
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # se entrena el modelo con los datos preparados
    model.fit(X, y, epochs=100, verbose=1)

    # Devolvemos el modelo entrenado junto con el tokenizador y el codificador.
    return model, tokenizer, encoder


"""
Función para generar texto a partir de un verso inicial y un número de palabras 
utilizando el modelo RNN entrenado

Recibe:
    modelo (tf.keras.Model): Modelo RNN entrenado para generar texto.
    tokenizer (Tokenizer): Instancia del tokenizador utilizado en el entrenamiento.
    verso_inicial (str): Verso inicial que servirá como punto de partida para la generación.
    num_palabras (int): Número de palabras que se generarán después del verso inicial.

Devuelve:
    str: Texto generado como una cadena de palabras.
"""
def hacer_predicciones_generativas(modelo, tokenizer, verso_inicial, num_palabras):    
    resultado = []  # Lista para almacenar las palabras generadas.
    verso_actual = verso_inicial  # Iniciamos con el verso proporcionado por el -usuario-
    for _ in range(num_palabras):
        # Convertimos el verso actual en una secuencia de números.
        secuencia = tokenizer.texts_to_sequences([verso_actual])[0]
        # Rellenamos la secuencia para que tenga la longitud adecuada.
        secuencia = pad_sequences([secuencia], maxlen=len(secuencia), padding='pre')
        # Realizamos una predicción con el modelo.
        prediccion = np.argmax(modelo.predict(secuencia), axis=-1)
        palabra_predicha = ''
        # Buscamos la palabra correspondiente al número predicho.
        for palabra, index in tokenizer.word_index.items():
            if index == prediccion:
                palabra_predicha = palabra
                break
        # Añadimos la palabra predicha al verso actual y al resultado.
        verso_actual += ' ' + palabra_predicha
        resultado.append(palabra_predicha)
    # Devolvemos el texto generado como una cadena de texto.
    return ' '.join(resultado)

"""
Función que entrena un modelo RNN para generar texto y luego utiliza el modelo 
entrenado para generar un nuevo poema a partir de un verso inicial y un 
número de palabras especificado

Args:
    poemas_limpios_path (str): Ruta del archivo que contiene los poemas ya limpios y preprocesados
    verso_inicial (str): Verso inicial que servirá como punto de partida para la generación del nuevo poema
    num_palabras (int): Número de palabras que se generarán después del verso inicial

Returns:
    Se imprime el poema generado en la consola
"""
def ejemplo_entrenamiento_generativo(poemas_limpios_path, verso_inicial, num_palabras):
    # Entrenamos la RNN generativa con el archivo de poemas limpios.
    modelo_rnn_generativa, tokenizer = entrenar_rnn_generativa(poemas_limpios_path)

    # Utilizamos el modelo entrenado para generar un nuevo poema.
    nuevo_poema = hacer_predicciones_generativas(modelo_rnn_generativa, tokenizer, verso_inicial, num_palabras)
    # Imprimimos el poema generado.
    print(f'Nuevo poema generado: {nuevo_poema}')
