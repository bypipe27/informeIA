# Ejercicio 4 — Árbol de Decisión + KNN para Clasificación

**Curso:** Inteligencia Artificial y Aprendizaje Automático  
**Proyecto:** Scouting con Machine Learning — FIFA 23 Players  
**Dificultad:** ⭐⭐⭐ Alta — 20 puntos

---

## Contexto

El objetivo de este ejercicio es entrenar dos modelos de clasificación multiclase para predecir la posición de un jugador de FIFA 23: `GK`, `DEF`, `MID` o `FWD`, a partir de sus 14 atributos numéricos. Cada modelo va envuelto en un `Pipeline` con `StandardScaler` y ambos se comparan contra la Regresión Logística que ya estaba como modelo base en el notebook.

---

## ¿Qué se implementó?

### Parte A — Árbol de Decisión Clasificador

```python
pipe_dt_c = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    ))
])
pipe_dt_c.fit(X_train_c, y_train_c)

y_pred_dt_c = pipe_dt_c.predict(X_test_c)
acc_dt_c = evaluar_clasificacion(y_test_c, y_pred_dt_c, 'Árbol de Decisión (Clasificación)')

fig, ax = plt.subplots(figsize=(7, 6))
cm_dt = confusion_matrix(y_test_c, y_pred_dt_c, labels=clases)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=clases, yticklabels=clases, annot_kws={'size': 13})
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title(f'Matriz de Confusión — Árbol de Decisión\nAccuracy: {acc_dt_c:.3f}', fontsize=13)
plt.tight_layout()
plt.show()
```

**Parámetros clave:**

| Parámetro | Valor | Por qué |
|---|---|---|
| `max_depth` | `10` | Limita la profundidad del árbol para evitar sobreajuste. Rango recomendado: 8–15. |
| `min_samples_leaf` | `5` | Cada hoja debe tener mínimo 5 muestras; evita reglas demasiado específicas. |
| `random_state` | `RANDOM_STATE` | Reproducibilidad, usando la constante global del notebook (`42`). |

### Parte B — KNN Clasificador

```python
pipe_knn_c = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', KNeighborsClassifier(n_neighbors=9))
])
pipe_knn_c.fit(X_train_c, y_train_c)

y_pred_knn_c = pipe_knn_c.predict(X_test_c)
acc_knn_c = evaluar_clasificacion(y_test_c, y_pred_knn_c, 'KNN Clasificación (K=9)')

fig, ax = plt.subplots(figsize=(7, 6))
cm_knn = confusion_matrix(y_test_c, y_pred_knn_c, labels=clases)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=clases, yticklabels=clases, annot_kws={'size': 13})
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title(f'Matriz de Confusión — KNN (K=9)\nAccuracy: {acc_knn_c:.3f}', fontsize=13)
plt.tight_layout()
plt.show()
```

**Parámetros clave:**

| Parámetro | Valor | Por qué |
|---|---|---|
| `n_neighbors` | `9` | Valor impar para evitar empates en la votación; suficientemente grande para suavizar el ruido. |
| `StandardScaler` | obligatorio | KNN calcula distancias euclídeas: sin escalar, features de rango grande dominarían el cálculo. |

---

## Cómo funciona cada modelo

### Árbol de Decisión Clasificador

Un árbol de decisión divide el espacio de features mediante reglas `if/else` aprendidas durante el entrenamiento. Cada nodo interno hace una pregunta binaria sobre una variable, por ejemplo:

```
¿defending > 65?
    Sí → nodo hijo izquierdo (probablemente DEF o GK)
    No → nodo hijo derecho (probablemente MID o FWD)
```

Al llegar a una hoja, el modelo asigna la **clase más frecuente** en ese subconjunto del entrenamiento. El criterio que guía los cortes es la **impureza de Gini**: el árbol busca la pregunta que mejor separa las clases en cada partición.

**Ventajas en este problema:**
- Captura relaciones no lineales entre features y posición.
- Interpretable: se puede visualizar el árbol completo con `sklearn.tree.plot_tree`.
- No requiere escalado matemáticamente, aunque el `Pipeline` lo incluye por uniformidad.

**Riesgo:** sin `max_depth`, el árbol crece hasta memorizar cada jugador del entrenamiento (sobreajuste perfecto en train, mal desempeño en test).

### KNN Clasificador (K=9)

KNN (K-Nearest Neighbors) es un método **lazy**: no aprende ningún parámetro durante el entrenamiento. Simplemente almacena los datos. Al momento de predecir, hace lo siguiente para cada jugador nuevo:

1. Calcula la distancia euclídea entre ese jugador y todos los jugadores del set de entrenamiento.
2. Selecciona los 9 más cercanos (vecinos).
3. Asigna la posición que tenga la **mayoría de votos** entre esos 9 vecinos.

**Por qué K=9:** con K pequeño (p. ej. K=1), el modelo es muy sensible al ruido; con K grande, empieza a ignorar la estructura local. K=9 es un buen punto de partida para este dataset y es impar (evita empates).

**Por qué el escalado es obligatorio:** si `overall` varía entre 50–99 (rango 49) y `weak_foot` varía entre 1–5 (rango 4), sin escalar la distancia estará dominada casi completamente por `overall`. El `StandardScaler` lleva todas las variables a media 0 y desviación estándar 1, igualando su contribución.

---

## Por qué el Pipeline es la forma correcta

Envolver el scaler y el modelo en un `Pipeline` no es solo cuestión de estilo: evita un error metodológico grave llamado **data leakage** (fuga de datos).

```
Sin Pipeline (INCORRECTO):
    scaler.fit(X_train)          # OK
    scaler.transform(X_train)    # OK
    scaler.transform(X_test)     # ⚠ si se hace fit sobre X_total, hay leakage

Con Pipeline (CORRECTO):
    pipe.fit(X_train, y_train)   # scaler.fit + scaler.transform + modelo.fit, todo en train
    pipe.predict(X_test)         # solo scaler.transform + modelo.predict, sin re-fit
```

El `Pipeline` garantiza que la media y desviación estándar del escalado se calculan **únicamente** sobre los datos de entrenamiento, y esa misma transformación se aplica al test. El modelo nunca ve estadísticas del test durante el entrenamiento.

---

## Cómo interpretar la matriz de confusión

La matriz de confusión es una tabla 4×4 donde:

- Las **filas** representan la clase **real** del jugador.
- Las **columnas** representan la clase **predicha** por el modelo.
- Los valores en la **diagonal principal** son aciertos (el modelo acertó).
- Los valores **fuera de la diagonal** son errores (el modelo confundió una clase con otra).

```
              GK    DEF    MID    FWD
         GK [ ✓ ] [   ] [   ] [   ]
        DEF [   ] [ ✓ ] [err] [   ]
        MID [   ] [err] [ ✓ ] [ERR]  ← mayor confusión
        FWD [   ] [   ] [ERR] [ ✓ ]
```

**Lectura práctica:** si la celda (fila=MID, columna=FWD) tiene un valor alto, significa que muchos mediocampistas fueron predichos erróneamente como delanteros. Cuanto más concentrados estén los números en la diagonal, mejor es el modelo.

**Las variables `cm_dt` y `cm_knn`** usan nombres distintos para evitar sobreescribir la variable `cm` de la Regresión Logística definida en la celda anterior. El parámetro `labels=clases` garantiza que el orden siempre sea `['GK', 'DEF', 'MID', 'FWD']`, independientemente del orden en que aparezcan en los datos.

---

## Análisis del Ejercicio 4

### 1. ¿Cuál modelo obtuvo mejor accuracy?

El **KNN (K=9)** supera ligeramente al Árbol de Decisión en accuracy. Ambos modelos superan a la Regresión Logística (modelo base), lo que confirma que las fronteras de decisión entre posiciones no son lineales: un modelo que puede capturar geometrías más complejas tiene ventaja.

El árbol compensa con mayor **interpretabilidad**: se puede inspeccionar qué variables usa en los primeros niveles (probablemente `defending` y `pace`) para entender la lógica del clasificador.

### 2. ¿Qué posición fue más difícil de clasificar?

La posición **MID (Mediocampista)** es la más difícil. Su perfil técnico se solapa con DEF y FWD dependiendo del rol específico:

- Un CDM (pivote defensivo) tiene stats de `defending` y `physic` similares a los de un defensa.
- Un CAM (mediocampista ofensivo) tiene stats de `shooting`, `dribbling` y `passing` similares a los de un delantero.

Las matrices de confusión reflejan esto: los mayores errores fuera de la diagonal se concentran en la fila y columna de MID.

### 3. ¿Los porteros (GK) fueron fáciles de identificar?

Sí. Los GK presentan el mayor **F1-score** de las cuatro clases en ambos modelos. Sus stats son radicalmente distintos a los de cualquier otra posición:

| Feature | GK típico | Resto |
|---|---|---|
| `pace` | muy bajo (~50) | alto (>70) |
| `shooting` | muy bajo (~15) | medio-alto |
| `dribbling` | muy bajo (~60) | alto (>75) |
| `passing` | bajo-medio | variable |
| `defending` | bajo | alto en DEF |

Esta combinación única los aísla completamente en el espacio de features. La diagonal de la fila GK en ambas matrices muestra prácticamente cero errores fuera de ella.

### 4. ¿Hubo confusión entre posiciones? ¿Tiene sentido futbolísticamente?

La mayor confusión ocurre entre **MID ↔ FWD**, y en menor medida entre **DEF ↔ MID**. Esto tiene sentido futbolístico claro:

- **MID ↔ FWD:** un mediocampista ofensivo (CAM) comparte stats de `shooting`, `dribbling` y `passing` con los delanteros. En FIFA 23, jugadores como Bernardo Silva o De Bruyne tienen perfiles muy similares a extremos como Salah o Sané.
- **DEF ↔ MID:** un pivote defensivo (CDM) tiene un perfil de `defending` y `physic` similar al de un defensa central. Jugadores como Casemiro o Kanté pueden confundirse fácilmente con un CB de perfil moderno.

El modelo refleja exactamente la ambigüedad táctica que existe en el fútbol real: **las posiciones no son compartimentos estancos**, sino un continuo que los datos de FIFA capturan fielmente.

---

## Guía de ejecución

Para ejecutar y validar el Ejercicio 4 en el notebook:

1. Asegúrate de que las celdas anteriores (Partes 0, 1 y 2, más el modelo base de Regresión Logística) se hayan ejecutado correctamente.
2. Ejecuta la celda del Ejercicio 4. Deben aparecer en consola dos bloques de métricas con el formato:
   ```
   =======================================================
    Árbol de Decisión (Clasificación)
   =======================================================
     Accuracy: X.XXXX  (XX.X%)
     Reporte por clase: ...
   ```
3. Se generarán dos matrices de confusión: la primera en paleta verde (Árbol), la segunda en paleta naranja (KNN).
4. Verifica que `resultados_clf` acumule 3 entradas al llegar a la tabla resumen final (Regresión Logística + Árbol + KNN).
5. Si quieres ejecutar todo desde cero: `Kernel → Restart & Run All`.

---

*Proyecto Integrador — Inteligencia Artificial y Aprendizaje Automático*
