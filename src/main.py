from limpiaPoemas import limpiar_poemas
from entrenaRNN import entrenar_rnn, hacer_predicciones

def main():
    archivo_entrada = 'poemas.json'
    archivo_salida = 'poemas_limpios.json'
    limpiar_poemas(archivo_entrada, archivo_salida)

    modelo, tokenizer, encoder = entrenar_rnn(archivo_salida)

    # Imprimir categorias
    categorias = encoder.classes_
    print(f'Available Categories: {categorias}')

    # Ejemplo de hacer predicciones
    nuevo_poema = "llueve en mi alma una tormenta sin fin lagrimas que caen sin poder resistir el viento susurra historias de dolor y en mi corazon solo queda el rencor las estrellas se esconden la luna no brilla la noche es eterna sin ninguna maravilla en cada rincon se esconde un recuerdo de tiempos felices que ahora pierdo el sol no amanece el dia no llega mi vida es un eco de una voz ciega caminos vacios sin rumbo ni guia perdido en la sombra sin alegria llueve en mi alma una tormenta sin fin mi corazon roto no puede seguir el pasado me ata no hay futuro que vea mi tristeza infinita mi alma lacera"
    modelo_rnn, tokenizer, encoder = entrenar_rnn('poemas_limpios.json')
    categoria_predicha = hacer_predicciones(modelo_rnn, nuevo_poema, tokenizer, encoder)
    print(f'La categoría predicha para el poema es: {categoria_predicha}')

if __name__ == "__main__":
    main()