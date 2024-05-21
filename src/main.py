from limpiaPoemas import limpiar_poemas
from entrenaRNN import entrenar_rnn, hacer_predicciones

def main():
    archivo_entrada = 'poemas.json'
    archivo_salida = 'poemas_limpios.json'
    modelo, tokenizer = entrenar_rnn(archivo_salida)

    # Ejemplo de hacer predicciones
    nuevo_poema = "Una vaga inquietud un misterioso temor como un feliz presentimiento un íntimo y recóndito tormento una pena que acaba en alborozo el sofocante nudo de un sollozo perenne en la garganta el sentimiento de un dolor que se acerca el pensamiento lleno de luz de júbilo de gozo una contradicción honda y obscura que me llena la vida de amargura que mata toda luz y toda idea que turba toda paz toda alegría pero Señor que sabes mi agonía si todo esto es amor bendito sea"
    prediccion = hacer_predicciones(modelo, nuevo_poema, tokenizer)
    print(f'Predicción: {prediccion}')

if __name__ == "__main__":
    main()