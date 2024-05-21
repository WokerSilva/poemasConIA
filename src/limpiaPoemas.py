import json
import re
import unicodedata
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Función para cargar datos desde un archivo JSON
def cargar_poemas(poemas_json):
    with open(poemas_json, 'r', encoding='utf-8') as archivo:
        poemas = json.load(archivo)
    return poemas

# Funciones de limpieza
def convertir_minusculas(texto):
    return texto.lower()

def eliminar_caracteres_especiales(texto):
    return re.sub(r'[^\w\s]', '', texto)

def eliminar_acentos(texto):
    forma_nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in forma_nfkd if not unicodedata.combining(c)])

def eliminar_espacios_extra(texto):
    return " ".join(texto.split())

def tokenizar(texto):
    return texto.split()

palabras_vacias = set(stopwords.words('spanish'))

def eliminar_palabras_vacias(tokens):
    return [palabra for palabra in tokens if palabra not in palabras_vacias]

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