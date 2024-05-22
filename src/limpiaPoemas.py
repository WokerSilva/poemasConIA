import json
import re
import unicodedata
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Cargamos los datos desde un archivo JSON
def cargar_poemas(poemas_json):
    with open(poemas_json, 'r', encoding='utf-8') as archivo:
        poemas = json.load(archivo)
    return poemas

# Función para convertir el texto a minúsculas
# Esto es esencial para normalizar el texto y facilitar la comparacion de palabras
def convertir_minusculas(texto):
    return texto.lower()

# Función para eliminar caracteres especiales del texto
# Esto ayuda a limpiar el texto de símbolos no deseados que pueden 
#  interferir con el procesamiento del texto
def eliminar_caracteres_especiales(texto):
    return re.sub(r'[^\w\s]', '', texto)

# Función para eliminar acentos del texto
# Normaliza el texto eliminando acentos para que las palabras con y sin 
#  acentos se traten por igual
def eliminar_acentos(texto):
    forma_nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in forma_nfkd if not unicodedata.combining(c)])

# Función para eliminar espacios extra en el texto
# Reduce múltiples espacios a un solo espacio mejorando la consistencia del texto
def eliminar_espacios_extra(texto):
    return " ".join(texto.split())

# Función para tokenizar el texto
# Divide el texto en una lista de palabras (tokens), lo cual es útil para 
#  el análisis y procesamiento del text
def tokenizar(texto):
    return texto.split()

# Cargar las palabras vacias en español
# Las palabras vacias (stopwords) son palabras comunes que no aportan mucho 
# sigificado y se eliminan en el procesamiento del texto
palabras_vacias = set(stopwords.words('spanish'))

# Función para eliminar palabras vacias de la lista de tokens
# Filtra las palabras vacías de los tokens para centrarse en las palabras significativas
def eliminar_palabras_vacias(tokens):
    return [palabra for palabra in tokens if palabra not in palabras_vacias]

# Función que aplica todas las funciones de limpieza en orden
def limpiar_contenido(contenido):
    contenido = convertir_minusculas(contenido)
    contenido = eliminar_caracteres_especiales(contenido)
    contenido = eliminar_acentos(contenido)
    contenido = eliminar_espacios_extra(contenido)
    tokens = tokenizar(contenido)
    tokens = eliminar_palabras_vacias(tokens)
    return " ".join(tokens)

# Función para guardar los datos limpios en un nuevo archivo JSON
def guardar_poemas(poemas, poemas_json):
    with open(poemas_json, 'w', encoding='utf-8') as archivo:
        json.dump(poemas, archivo, ensure_ascii=False, indent=4)

# Función principal para cargar, limpiar y guardar los poemas
def limpiar_poemas(archivo_entrada, archivo_salida):
    poemas = cargar_poemas(archivo_entrada)
    poemas_limpios = []

    for item in poemas:
        contenido_limpio = limpiar_contenido(item["contenido"])
        poemas_limpios.append({
            "categoria": item["categoria"],
            "autor": item["autor"],
            "titulo": item["titulo"],
            "contenido": contenido_limpio
        })

    guardar_poemas(poemas_limpios, archivo_salida)