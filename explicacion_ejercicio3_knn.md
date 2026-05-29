# Ejercicio 3: KNN para Regresión

## Explicación simple

En este ejercicio usamos KNN para intentar adivinar cuánto vale un jugador.

La idea de KNN es muy simple: para predecir el valor de un jugador, mira a sus vecinos más parecidos y usa sus valores para hacer una estimación.

## Qué hicimos

Primero creamos un `Pipeline` con dos pasos:

1. `StandardScaler()` para poner todas las variables en la misma escala.
2. `KNeighborsRegressor(n_neighbors=5)` para hacer la predicción.

Luego entrenamos el modelo con los datos de entrenamiento y lo probamos con los datos de test.

Después medimos qué tan bien funcionó usando:

- RMSE
- MAE
- R²

## Por qué usamos escalado

KNN trabaja con distancias.
Si una variable tiene valores mucho más grandes que las demás, puede dominar la comparación.
Por eso primero escalamos los datos.

## Cómo buscamos el mejor K

Probamos varios valores de `K`, desde 1 hasta 30.
Para cada uno:

- entrenamos el modelo
- hicimos predicciones
- calculamos el R²

Después elegimos el `K` que dio el mejor resultado y lo mostramos en una gráfica.

## Bonus con validación cruzada

También probamos la búsqueda del mejor `K` usando validación cruzada.

Eso significa que en vez de mirar solo un split de train/test, el modelo se prueba varias veces con distintos cortes de los datos.

Esto da una respuesta más confiable, pero tarda más.

## Por qué tarda bastante

Tarda porque estamos probando muchos valores de `K` y, además, para cada uno hacemos 5 validaciones.

En total son muchas predicciones, y KNN necesita calcular distancias con muchos jugadores.

## Idea clave

El valor real se modela con `log1p(value_eur)` para que sea más fácil de aprender.
Si después queremos volver a euros, usamos:

```python
np.expm1(y_pred)
```

## Resumen final

En pocas palabras:

- usamos KNN para predecir el valor de un jugador
- escalamos los datos antes de entrenar
- probamos distintos valores de `K`
- y con el bonus usamos validación cruzada para elegir mejor
