from limpiaPoemas import limpiar_poemas
from entrenaRNN import entrenar_rnn, hacer_predicciones
from generarPoema import generar_poema, extraer_palabras_relevantes, cargar_modelo_y_datos, guardar_tokenizer, guardar_encoder

def main():
    archivo_entrada = 'poemas.json'
    archivo_salida = 'poemas_limpios.json'
    limpiar_poemas(archivo_entrada, archivo_salida)
    
    # Entrenar el modelo RNN para clasificación
    modelo_rnn, tokenizer, encoder = entrenar_rnn(archivo_salida)
    
    # Guardar el tokenizer y el encoder para su uso posterior
    guardar_tokenizer(tokenizer, 'tokenizer.npy')
    guardar_encoder(encoder, 'encoder.npy')
    
    # Imprimir las categorías disponibles
    categorias = encoder.classes_
    print(f'Categorías disponibles: {categorias}')

    # Hacer una predicción con el modelo RNN
    nuevo_poema = "llueve en mi alma una tormenta sin fin..."
    print(f'Predicción para el nuevo poema: {hacer_predicciones(modelo_rnn, nuevo_poema, tokenizer, encoder)}')

    # Generar un nuevo poema
    print("Nuevo poema .. cargando .. ")
    modelo_rnn, tokenizer, encoder = cargar_modelo_y_datos('modelo_poemas_rnn.keras', 'tokenizer.npy', 'encoder.npy')
    nuevo_poema_generado = generar_poema(modelo_rnn, tokenizer, encoder)
    print(f'Nuevo poema generado: {nuevo_poema_generado}')
    
    # Extraer palabras relevantes de los datos de entrenamiento
    palabras_relevantes = extraer_palabras_relevantes(tokenizer)
    print(f'Palabras relevantes: {palabras_relevantes}')

if __name__ == "__main__":
    main()
