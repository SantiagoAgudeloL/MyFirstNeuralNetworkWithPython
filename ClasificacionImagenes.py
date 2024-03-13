import tensorflow as tf
import tensorflow_datasets as tfds

#descargar datasets de zalando

datos, metadatos = tfds.load('fashion_mnist',as_supervised=True, with_info=True)
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']
nombres_clases = metadatos.features['label'].names
print(nombres_clases)