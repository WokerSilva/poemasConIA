import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Función para cargar los poemas limpios desde un archivo JSON
def cargar_poemas_limpios(archivo_limpios):
    with open(archivo_limpios, 'r', encoding='utf-8') as archivo:
        poemas_limpios = json.load(archivo)
    return poemas_limpios

# Función para entrenar la Red Neuronal Recurrente
def entrenar_rnn(archivo_limpios):
    poemas_limpios = cargar_poemas_limpios(archivo_limpios)

    # Extraer las categorías y codificarlas
    categorias = [poema["categoria"] for poema in poemas_limpios]
    encoder = LabelEncoder()
    categorias_codificadas = encoder.fit_transform(categorias)
    categorias_categorical = tf.keras.utils.to_categorical(categorias_codificadas)

    # Tokenizar los textos limpios
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([poema["contenido"] for poema in poemas_limpios])
    secuencias = tokenizer.texts_to_sequences([poema["contenido"] for poema in poemas_limpios])
    secuencias_padded = pad_sequences(secuencias)

    # Definir y compilar el modelo de la RNN
    modelo_rnn = Sequential()
    modelo_rnn.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=secuencias_padded.shape[1]))
    modelo_rnn.add(Bidirectional(LSTM(100, return_sequences=True)))
    modelo_rnn.add(Dropout(0.5))
    modelo_rnn.add(Bidirectional(LSTM(100)))
    modelo_rnn.add(Dropout(0.5))
    modelo_rnn.add(Dense(len(encoder.classes_), activation='softmax'))
    modelo_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    # Entrenar el modelo
    modelo_rnn.fit(secuencias_padded, categorias_categorical, epochs=10, validation_split=0.2)

    # Guardar el modelo entrenado
    modelo_rnn.save('modelo_poemas_rnn.keras')

    # Evaluar el modelo
    evaluar_modelo(modelo_rnn, secuencias_padded, categorias_categorical, categorias_codificadas, encoder)

    return modelo_rnn, tokenizer, encoder

# Evaluar el modelo con datos de prueba
def evaluar_modelo(modelo, secuencias_padded, categorias_categorical, categorias_codificadas, encoder):
    loss, accuracy, precision, recall = modelo.evaluate(secuencias_padded, categorias_categorical)
    print(f'Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

    # Predecir categorias
    predicciones = modelo.predict(secuencias_padded)
    categorias_predichas = np.argmax(predicciones, axis=1)

    # Matriz de Confusión
    matriz_confusion = confusion_matrix(categorias_codificadas, categorias_predichas)
    print("Matriz de Confusión:")
    print(matriz_confusion)

    # Reporte de Clasificación
    reporte_clasificacion = classification_report(categorias_codificadas, categorias_predichas, target_names=encoder.classes_)
    print("Reporte de Clasificación:")
    print(reporte_clasificacion)

    # Graficar la Matriz de Confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
# Función para hacer predicciones
def hacer_predicciones(modelo, nuevo_poema, tokenizer, encoder):
    secuencia = tokenizer.texts_to_sequences([nuevo_poema])
    secuencia_padded = pad_sequences(secuencia, maxlen=modelo.input_shape[1])
    prediccion = modelo.predict(secuencia_padded)
    categoria_idx = np.argmax(prediccion, axis=1)
    categoria = encoder.inverse_transform(categoria_idx)
    return categoria