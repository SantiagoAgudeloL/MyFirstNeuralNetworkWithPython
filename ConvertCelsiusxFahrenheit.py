import tensorflow as tf
import numpy as np 

#Se define arreglo con el que se entrenara la red
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

#Se define las capas.
oculta1 = tf.keras.layers.Dense(units=3 , input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1,oculta2,salida])

#Esto dice como se hará el entrenamiento de la red.
# Optimizador (0.1) adjustara poco a poco la red. 
# loss es la función de perdida.
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Con los array hará 1000 entrenamientos y verbose = false para que no muestre tanta info. 
print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenando!")

#muestrá grafica de entrenamiento
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()


#prueba de predicción
print("Hagamos una predicción!")
resultado = modelo.predict([100.0])
print( "El resultado es " + str(resultado) + " Fahrenheit! ")


#muestrá las varaibles predecidas para resolver el problema lineal (Y=MX+B)
print("Variables internas del modelo")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())