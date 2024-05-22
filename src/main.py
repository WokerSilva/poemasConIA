from limpiaPoemas import limpiar_poemas
from entrenaRNN import entrenar_rnn, hacer_predicciones
from generarPoema import ejemplo_entrenamiento_generativo, ejemplo_entrenamiento_generativo

def main():
    archivo_entrada = 'poemas.json'
    archivo_salida = 'poemas_limpios.json'
    limpiar_poemas(archivo_entrada, archivo_salida)
    
    # Entrenar el modelo RNN para clasificación
    modelo_rnn, tokenizer, encoder = entrenar_rnn(archivo_salida)
    
    # Guardar el tokenizer y el encoder para su uso posterior
    #guardar_tokenizer(tokenizer, 'tokenizer.npy')
    #guardar_encoder(encoder, 'encoder.npy')
    
    # Imprimir las categorías disponibles
    categorias = encoder.classes_
    print(f'Categorías existentes: {categorias}')

    # Hacer una predicción con el modelo RNN
    nuevo_poema = "llueve en mi alma una tormenta sin fin..."
    print(f'Predicción para el nuevo poema: {hacer_predicciones(modelo_rnn, nuevo_poema, tokenizer, encoder)}')

    # Generar un nuevo poema
    print("cargando...")
    #modelo_rnn, tokenizer, encoder = cargar_modelo_y_datos('modelo_poemas_rnn.keras', 'tokenizer.npy', 'encoder.npy')

    # Ejemplo de entrenamiento de la RNN generativa y generación de un nuevo poema
    print("Entrenando RNN generativa...")
    ejemplo_entrenamiento_generativo(archivo_salida, 'llueve en mi alma', 50)

if __name__ == "__main__":
    main()