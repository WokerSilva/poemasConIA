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

# Función para cargar poemas limpios desde un archivo JSON
def cargar_poemas_limpios(archivo_limpios):
    # Abre el archivo en modo lectura y carga los datos en formato JSON
    with open(archivo_limpios, 'r', encoding='utf-8') as archivo:
        poemas_limpios = json.load(archivo)
    # Devuelve los poemas limpios
    return poemas_limpios

# Función para entrenar el modelo RNN
def entrenar_rnn(archivo_limpios):
    # Carga los poemas limpios utilizando la función definida anteriormente
    poemas_limpios = cargar_poemas_limpios(archivo_limpios)

    # Extrae las categorías de los poemas y las codifica en números enteros
    categorias = [poema["categoria"] for poema in poemas_limpios]
    encoder = LabelEncoder()
    categorias_codificadas = encoder.fit_transform(categorias)
    # Convierte las categorías codificadas en formato one-hot para el entrenamiento
    categorias_categorical = tf.keras.utils.to_categorical(categorias_codificadas)

    # Tokeniza los textos limpios, convirtiendo cada palabra en un número entero único
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([poema["contenido"] for poema in poemas_limpios])
    # Convierte los textos en secuencias de enteros
    secuencias = tokenizer.texts_to_sequences([poema["contenido"] for poema in poemas_limpios])
    # Asegura que todas las secuencias tengan la misma longitud mediante padding
    secuencias_padded = pad_sequences(secuencias)

    # Define y compila el modelo RNN con capas LSTM apiladas
    modelo_rnn = Sequential()
    # Capa de embedding que convierte enteros en vectores densos de tamaño fijo
    modelo_rnn.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=secuencias_padded.shape[1]))
    # Capas LSTM bidireccionales para aprender dependencias en ambas direcciones del texto
    modelo_rnn.add(Bidirectional(LSTM(100, return_sequences=True)))
    # Capa de Dropout para reducir el sobreajuste durante el entrenamiento
    modelo_rnn.add(Dropout(0.5))
    # Otra capa LSTM bidireccional
    modelo_rnn.add(Bidirectional(LSTM(100)))
    # Otra capa de Dropout
    modelo_rnn.add(Dropout(0.5))
    # Capa densa para la clasificación de categorías con activación softmax
    modelo_rnn.add(Dense(len(encoder.classes_), activation='softmax'))
    # Compila el modelo con optimizador adam y función de pérdida para clasificación multiclase
    modelo_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    # Entrena el modelo con los datos preparados y validación del 20%
    modelo_rnn.fit(secuencias_padded, categorias_categorical, epochs=10, validation_split=0.2)

    # Guarda el modelo entrenado en el disco
    modelo_rnn.save('modelo_poemas_rnn.keras')

    # Evalúa el modelo con los datos de entrenamiento y muestra métricas de rendimiento
    evaluar_modelo(modelo_rnn, secuencias_padded, categorias_categorical, categorias_codificadas, encoder)

    # Devuelve el modelo, el tokenizer y el encoder para su uso posterior
    return modelo_rnn, tokenizer, encoder


# Función para evaluar el modelo con datos de prueba
def evaluar_modelo(modelo, secuencias_padded, categorias_categorical, categorias_codificadas, encoder):
    # Evalúa el modelo y obtiene las métricas de pérdida, precisión, precisión y exhaustividad
    loss, accuracy, precision, recall = modelo.evaluate(secuencias_padded, categorias_categorical)
    print(f'Pérdida: {loss}, Precisión: {accuracy}, Precisión por clase: {precision}, Exhaustividad: {recall}')

    # Realiza predicciones con el modelo sobre las secuencias proporcionadas
    predicciones = modelo.predict(secuencias_padded)
    # Convierte las predicciones en índices de categorías
    categorias_predichas = np.argmax(predicciones, axis=1)

    # Crea una matriz de confusión para comparar las categorías verdaderas con las predichas
    matriz_confusion = confusion_matrix(categorias_codificadas, categorias_predichas)
    print("Matriz de Confusión:")
    print(matriz_confusion)

    # Genera un reporte de clasificación con métricas detalladas por clase
    reporte_clasificacion = classification_report(categorias_codificadas, categorias_predichas, target_names=encoder.classes_)
    print("Reporte de Clasificación:")
    print(reporte_clasificacion)

    # Visualiza la matriz de confusión utilizando una gráfica de calor
    plt.figure(figsize=(10, 7))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()

# Función para hacer predicciones
def hacer_predicciones(modelo, nuevo_poema, tokenizer, encoder):
    # Convierte el texto del nuevo poema en una secuencia numérica
    secuencia = tokenizer.texts_to_sequences([nuevo_poema])
    # Ajusta la secuencia al tamaño máximo requerido por el modelo
    secuencia_padded = pad_sequences(secuencia, maxlen=modelo.input_shape[1])
    # Realiza una predicción con el modelo
    prediccion = modelo.predict(secuencia_padded)
    # Obtiene el índice de la categoría con la mayor probabilidad
    categoria_idx = np.argmax(prediccion, axis=1)
    # Transforma el índice numérico de vuelta a la etiqueta de categoría original
    categoria = encoder.inverse_transform(categoria_idx)
    return categoria
