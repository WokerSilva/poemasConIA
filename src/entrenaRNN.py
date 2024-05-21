import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Función para cargar los poemas limpios desde un archivo JSON
def cargar_poemas_limpios(archivo_limpios):
    with open(archivo_limpios, 'r', encoding='utf-8') as archivo:
        poemas_limpios = json.load(archivo)
    return poemas_limpios

# Función para entrenar la Red Neuronal Recurrente (RNN)
def entrenar_rnn(archivo_limpios):
    poemas_limpios = cargar_poemas_limpios(archivo_limpios)

    # Obtener las etiquetas de las categorías
    categorias = [poema["categoria"] for poema in poemas_limpios]
    encoder = LabelEncoder()
    categorias_codificadas = encoder.fit_transform(categorias)

    # Tokenizar los textos limpios
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([poema["contenido"] for poema in poemas_limpios])
    secuencias = tokenizer.texts_to_sequences([poema["contenido"] for poema in poemas_limpios])
    secuencias_padded = pad_sequences(secuencias)

    # Definir y compilar el modelo de la RNN
    modelo_rnn = Sequential()
    modelo_rnn.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=secuencias_padded.shape[1]))
    modelo_rnn.add(LSTM(100))
    modelo_rnn.add(Dense(1, activation='sigmoid'))
    modelo_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    modelo_rnn.fit(secuencias_padded, categorias_codificadas, epochs=10, validation_split=0.2)

    # Guardar el modelo entrenado
    modelo_rnn.save('modelo_poemas_rnn.keras')

    # Evaluar el modelo
    evaluar_modelo(modelo_rnn, secuencias_padded, categorias_codificadas)

    return modelo_rnn, tokenizer



# Evaluar el modelo con datos de prueba (opcional)
def evaluar_modelo(modelo, secuencias_padded, categorias_codificadas):
    loss, accuracy = modelo.evaluate(secuencias_padded, categorias_codificadas)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

# Función para hacer predicciones
def hacer_predicciones(modelo, nuevo_poema, tokenizer):
    secuencia = tokenizer.texts_to_sequences([nuevo_poema])
    secuencia_padded = pad_sequences(secuencia, maxlen=modelo.input_shape[1])
    prediccion = modelo.predict(secuencia_padded)
    return prediccion