import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import LabelEncoder
import numpy as np

def cargar_poemas_limpios(archivo_limpios):
    with open(archivo_limpios, 'r', encoding='utf-8') as archivo:
        poemas_limpios = json.load(archivo)
    return poemas_limpios

def entrenar_rnn(archivo_limpios):
    poemas_limpios = cargar_poemas_limpios(archivo_limpios)

    # Obtener las etiquetas de las categorías
    categorias = [poema["categoria"] for poema in poemas_limpios]
    encoder = LabelEncoder()
    categorias_codificadas = encoder.fit_transform(categorias)
    categorias_codificadas = tf.keras.utils.to_categorical(categorias_codificadas)


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([poema["contenido"] for poema in poemas_limpios])
    secuencias = tokenizer.texts_to_sequences([poema["contenido"] for poema in poemas_limpios])
    secuencias_padded = pad_sequences(secuencias)


    modelo_rnn = Sequential()
    modelo_rnn.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=secuencias_padded.shape[1]))
    modelo_rnn.add(Bidirectional(LSTM(100, return_sequences=True)))
    modelo_rnn.add(Dropout(0.5))
    modelo_rnn.add(Bidirectional(LSTM(100)))
    modelo_rnn.add(Dropout(0.5))
    modelo_rnn.add(Dense(len(encoder.classes_), activation='softmax'))
    modelo_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    modelo_rnn.fit(secuencias_padded, categorias_codificadas, epochs=10, validation_split=0.2)

    modelo_rnn.save('modelo_poemas_rnn.keras')

    evaluar_modelo(modelo_rnn, secuencias_padded, categorias_codificadas)

    return modelo_rnn, tokenizer, encoder


def evaluar_modelo(modelo, secuencias_padded, categorias_codificadas):
    loss, accuracy, precision, recall = modelo.evaluate(secuencias_padded, categorias_codificadas)
    print(f'Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

def hacer_predicciones(modelo, nuevo_poema, tokenizer, encoder):
    secuencia = tokenizer.texts_to_sequences([nuevo_poema])
    secuencia_padded = pad_sequences(secuencia, maxlen=modelo.input_shape[1])
    prediccion = modelo.predict(secuencia_padded)
    categoria_idx = np.argmax(prediccion, axis=1)
    categoria = encoder.inverse_transform(categoria_idx)
    return categoria