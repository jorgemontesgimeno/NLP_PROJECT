# Práctica — Clasificación de texto en redes sociales: limpieza, modelos y comparación crítica

## 1. Contexto de la práctica

Hasta ahora hemos trabajado análisis de sentimiento con textos relativamente largos, como reseñas de películas de IMDB. Ese tipo de texto tiene una estructura bastante estable: frases más largas, menos abreviaturas, menos emojis, menos hashtags y menos ruido propio de redes sociales.

En esta práctica vamos a cambiar de escenario: trabajaréis con textos breves procedentes de redes sociales. Este cambio es importante porque obliga a pensar mucho más en el preprocesamiento. En redes sociales aparecen menciones a usuarios, URLs, hashtags, emojis, ironía, abreviaturas, faltas de ortografía, mayúsculas expresivas, repeticiones de letras, mensajes muy cortos y, a veces, etiquetas ambiguas o ruidosas.

El objetivo no es simplemente entrenar muchos modelos, sino construir un estudio comparativo serio: probar distintas formas de preparar el texto, aplicar varios enfoques de NLP y analizar qué funciona, qué no funciona, por qué ocurre y si las conclusiones son generales o dependen del dataset y del tipo de problema.

Como posibles datasets podéis utilizar, por ejemplo, **TweetEval**, que reúne varias tareas de clasificación sobre Twitter/X como sentimiento, emoción, ironía, discurso ofensivo, odio, stance y emoji; **Sentiment140**, basado en tweets etiquetados de forma ruidosa mediante emoticonos; o algún dataset de detección de sexismo/odio en redes sociales, como los de la serie **EXIST**, si se quiere trabajar con una tarea socialmente más delicada.

---

## 2. Objetivo general

Desarrollar un sistema de clasificación de textos de redes sociales comparando, al menos, los siguientes enfoques:

1. Un enfoque de **NLP clásico** basado en limpieza, vectorización y modelos tradicionales de Machine Learning.
2. Un enfoque de **red neuronal** entrenada sobre el dataset, por ejemplo LSTM, GRU, CNN textual o un pequeño Transformer construido con Keras.
3. Opcionalmente, un enfoque basado en **modelo preentrenado o LLM**, ya sea mediante fine-tuning de un modelo tipo BERT/RoBERTa/DistilBERT, o mediante prompting zero-shot/few-shot con un LLM local vía Ollama o mediante API.

La práctica debe centrarse especialmente en la comparación: no basta con obtener una métrica final. Hay que estudiar cómo afecta cada decisión del proceso: limpieza, representación del texto, tipo de modelo, hiperparámetros, tamaño del dataset, balance de clases, longitud de los textos, presencia de emojis, hashtags, negaciones, ironía o ruido.

---

## 3. Elección del problema y del dataset

Cada grupo deberá escoger un dataset de texto procedente de redes sociales. La tarea puede ser, por ejemplo:

- Clasificación de sentimiento: positivo, negativo, neutro.
- Clasificación de emociones.
- Detección de lenguaje ofensivo.
- Detección de odio.
- Detección de ironía.
- Detección de sexismo.
- Clasificación temática de tweets o mensajes.

El dataset elegido debe permitir plantear un problema de clasificación supervisada. Es decir, cada texto debe tener una etiqueta asociada.

Se recomienda que el dataset tenga suficientes ejemplos para poder dividirlo en entrenamiento, validación y test, o que ya proporcione estas particiones. Si el dataset es muy grande, se puede trabajar con una muestra razonable, siempre que se mantenga el equilibrio entre clases y se justifique la decisión.

---

## 4. Preguntas que debe responder el trabajo

La práctica debe estar guiada por preguntas como las siguientes:

- ¿Qué tipo de ruido aparece en los textos?
- ¿Qué elementos parecen irrelevantes y cuáles podrían ser informativos?
- ¿Conviene eliminar emojis o pueden aportar información emocional?
- ¿Conviene eliminar hashtags o pueden resumir el tema del mensaje?
- ¿Qué ocurre si eliminamos menciones, URLs o signos de puntuación?
- ¿Qué pasa con las mayúsculas, las repeticiones de letras o las abreviaturas?
- ¿Los modelos clásicos funcionan sorprendentemente bien?
- ¿La red neuronal mejora realmente al modelo clásico o solo añade complejidad?
- ¿Un modelo preentrenado entiende mejor el lenguaje informal de redes sociales?
- ¿Un LLM zero-shot es competitivo frente a modelos entrenados específicamente?
- ¿Qué modelo recomendaríais si importan la precisión, el coste, la velocidad, la interpretabilidad o la facilidad de despliegue?

---

## 5. Fase 1 — Análisis exploratorio del dataset

Antes de entrenar modelos, se debe realizar un análisis exploratorio del dataset.

Como mínimo, se espera estudiar:

- Número total de ejemplos.
- Número de clases.
- Distribución de clases.
- Ejemplos reales de cada clase.
- Longitud de los textos en caracteres y palabras.
- Presencia de URLs.
- Presencia de menciones a usuarios.
- Presencia de hashtags.
- Presencia de emojis.
- Posibles duplicados.
- Posibles textos vacíos o extremadamente cortos.
- Palabras o tokens frecuentes por clase.
- Desequilibrio entre clases.
- Posibles problemas de etiquetado ambiguo.

No se trata solo de mostrar gráficos, sino de interpretarlos. Cada visualización debe ir acompañada de una explicación: qué muestra, por qué es relevante y qué decisiones puede motivar.

---

## 6. Fase 2 — Limpieza y preprocesamiento

Esta es una de las partes más importantes de la práctica.

Deberéis construir una función o pipeline de limpieza que transforme los textos originales en textos preparados para los modelos. Sin embargo, no debéis asumir que “más limpieza” siempre es mejor.

Se recomienda comparar varias versiones del texto:

### Versión A — Texto casi original

Mantener el texto con los mínimos cambios posibles. Por ejemplo:

- Normalizar espacios.
- Corregir problemas básicos de codificación.
- Mantener emojis, hashtags, signos y mayúsculas.

Esta versión permite comprobar qué ocurre cuando dejamos que el modelo trabaje con la mayor cantidad posible de información original.

### Versión B — Limpieza moderada

Aplicar una limpieza razonable, por ejemplo:

- Pasar a minúsculas.
- Sustituir URLs por un token especial como `<URL>`.
- Sustituir menciones por `<USER>`.
- Separar hashtags o conservarlos sin el símbolo `#`.
- Normalizar repeticiones excesivas.
- Mantener emojis o convertirlos a texto, si se considera útil.
- Eliminar espacios repetidos.

Esta versión suele ser una buena candidata para redes sociales, porque elimina ruido pero conserva señales útiles.

### Versión C — Limpieza agresiva

Aplicar técnicas más fuertes, por ejemplo:

- Eliminar puntuación.
- Eliminar stopwords.
- Aplicar stemming o lematización.
- Eliminar emojis.
- Eliminar hashtags.
- Eliminar tokens poco frecuentes.

Esta versión debe usarse con cuidado. En algunos problemas puede mejorar el rendimiento, pero en otros puede destruir información importante. Por ejemplo, eliminar la palabra “no” puede cambiar por completo el significado de una frase.

La práctica debe comparar al menos dos versiones de limpieza y justificar cuál parece más adecuada para cada modelo.

---

## 7. Fase 3 — Primer enfoque: NLP clásico

Deberéis implementar al menos un enfoque clásico de NLP basado en vectorización y modelos tradicionales de Machine Learning.

Como mínimo, se espera incluir:

- Vectorización con Bag of Words o TF-IDF.
- Uso de unigramas y, al menos, una prueba con bigramas.
- Entrenamiento de un modelo clásico.

Modelos recomendados:

- Regresión logística.
- Naive Bayes.
- Linear SVM.
- Random Forest, aunque no suele ser la primera opción para texto disperso.
- Otro modelo clásico justificado.

Se espera que probéis diferentes configuraciones, por ejemplo:

- `CountVectorizer` frente a `TfidfVectorizer`.
- Unigramas frente a unigramas + bigramas.
- Diferentes valores de `max_features`.
- Uso o no de stopwords.
- Diferentes niveles de limpieza.
- Ajuste básico de hiperparámetros.

El objetivo no es probar combinaciones al azar, sino plantear experimentos con sentido.

Ejemplo de razonamiento esperado:

> “Probamos TF-IDF porque en textos cortos puede ayudar a reducir el peso de palabras muy frecuentes. Sin embargo, también probamos bigramas porque expresiones como ‘not good’, ‘no me gusta’ o ‘very happy’ pueden cambiar el significado respecto a mirar palabras aisladas.”

---

## 8. Fase 4 — Segundo enfoque: red neuronal

Deberéis implementar al menos un modelo de red neuronal entrenado sobre el dataset.

Opciones posibles:

- Embedding + LSTM.
- Embedding + GRU.
- Embedding + CNN 1D.
- Embedding + bloque Transformer sencillo en Keras.
- Otra arquitectura neuronal justificada.

El modelo debe incluir una explicación clara de:

- Cómo se tokeniza el texto.
- Qué tamaño de vocabulario se usa.
- Qué longitud máxima de secuencia se elige.
- Qué ocurre con textos más largos o más cortos.
- Qué hace la capa de embeddings.
- Por qué se usa LSTM, GRU, CNN o Transformer.
- Qué función de pérdida se utiliza.
- Qué métrica se monitoriza.
- Cómo se controla el sobreajuste.

Se espera que uséis técnicas como:

- Separación entrenamiento/validación/test.
- Early stopping.
- Dropout.
- Comparación de curvas de entrenamiento y validación.
- Matriz de confusión.
- Análisis de errores.

No basta con entrenar una red neuronal y decir que “es más avanzada”. Hay que comprobar si realmente mejora el enfoque clásico. En muchos problemas de texto, un modelo clásico bien ajustado puede ser sorprendentemente competitivo.

---

## 9. Fase 5 — Modelo avanzado: fine-tuning o LLM zero-shot

Además de los modelos anteriores, se recomienda incluir una tercera familia de modelos más avanzada. Hay dos caminos posibles.

### Opción A — Fine-tuning de un modelo preentrenado

Podéis descargar un modelo tipo:

- DistilBERT.
- BERT.
- RoBERTa.
- MiniLM.
- Un modelo especializado en tweets, si procede.

La idea es adaptar un modelo que ya ha aprendido representaciones lingüísticas generales a la tarea concreta del dataset.

Se debe explicar:

- Qué modelo se ha elegido.
- Qué tokenizer utiliza.
- Qué longitud máxima de entrada se ha usado.
- Si se entrena todo el modelo o solo la cabeza de clasificación.
- Qué coste computacional tiene.
- Qué ventajas aporta respecto a entrenar una red neuronal desde cero.
- Qué limitaciones tiene.

No se espera necesariamente entrenar durante muchas épocas. Puede bastar con una muestra del dataset o con pocas épocas, siempre que el experimento esté bien planteado y se interprete correctamente.

### Opción B — LLM zero-shot o few-shot

También podéis probar un LLM sin entrenarlo, usando prompting.

Por ejemplo:

- Un modelo local vía Ollama.
- Un modelo accesible por API.
- Un modelo pequeño que pueda ejecutarse razonablemente en local.

En este caso, el LLM recibe el texto y debe devolver una etiqueta. Se debe diseñar un prompt claro, controlar el formato de salida y evaluar el resultado sobre una muestra del dataset.

Aspectos que deben analizarse:

- Calidad de las predicciones.
- Consistencia de las respuestas.
- Errores de formato.
- Sensibilidad al prompt.
- Coste por predicción.
- Tiempo de inferencia.
- Privacidad de los datos.
- Dificultad de automatizar la evaluación.
- Comparación frente a modelos entrenados específicamente.

Ejemplo de prompt:

```text
Clasifica el siguiente texto de red social en una de estas etiquetas:
- negativo
- neutro
- positivo

Devuelve únicamente un JSON con este formato:
{"label": "...", "confidence": 0.0}

Texto:
"..."
```

El uso de LLM no debe sustituir al resto de la práctica. Debe utilizarse como comparación crítica: puede ser potente, pero también más caro, más lento, menos reproducible y más difícil de controlar.

---

## 10. Fase 6 — Evaluación

La evaluación debe ser común para todos los modelos, en la medida de lo posible.

Como mínimo, se espera incluir:

- Accuracy.
- Precision, recall y F1-score.
- Macro-F1, especialmente si hay desequilibrio entre clases.
- Matriz de confusión.
- Comparación entre modelos en una tabla final.
- Análisis de errores cualitativo.

Si el problema es binario, también se pueden incluir:

- Curva ROC.
- AUC.
- Curva precision-recall.
- Análisis de umbral de decisión.

Si el problema es multiclase, es especialmente importante mirar las métricas por clase. Un modelo puede tener buena accuracy global y, aun así, funcionar muy mal en una clase minoritaria.

---

## 11. Comparación obligatoria de experimentos

La práctica debe incluir una tabla comparativa clara.

Un ejemplo de estructura sería:

| Experimento | Texto usado | Representación | Modelo | Accuracy | Macro-F1 | Comentario |
|---|---|---|---|---:|---:|---|
| 1 | Limpieza mínima | TF-IDF | Regresión logística | ... | ... | Buen baseline |
| 2 | Limpieza moderada | TF-IDF + bigramas | Linear SVM | ... | ... | Mejora en negaciones |
| 3 | Limpieza agresiva | TF-IDF | Regresión logística | ... | ... | Pierde información |
| 4 | Texto moderado | Embedding | LSTM | ... | ... | Sobreajuste leve |
| 5 | Texto moderado | Embedding | Transformer Keras | ... | ... | Necesita más datos |
| 6 | Texto original | Tokenizer preentrenado | DistilBERT/RoBERTa | ... | ... | Mejor comprensión contextual |
| 7 | Texto original | Prompt | LLM zero-shot | ... | ... | Buen rendimiento pero lento/caro |

No es obligatorio que esta tabla tenga exactamente estos experimentos, pero sí debe existir una comparación ordenada y razonada.

---

## 12. Análisis de errores

Cada grupo debe revisar manualmente varios errores del modelo.

Se espera seleccionar ejemplos como:

- Falsos positivos.
- Falsos negativos.
- Errores en clases minoritarias.
- Casos donde el modelo clásico falla pero la red neuronal acierta.
- Casos donde la red neuronal falla pero el modelo clásico acierta.
- Casos donde el LLM acierta o falla de forma llamativa.
- Textos ambiguos incluso para una persona.

Para cada ejemplo seleccionado, se debe comentar qué puede haber ocurrido.

Posibles causas:

- Ironía o sarcasmo.
- Uso de emojis.
- Hashtags con carga semántica.
- Negaciones.
- Texto demasiado corto.
- Palabras fuera del vocabulario.
- Lenguaje ofensivo usado en tono irónico.
- Etiqueta discutible.
- Falta de contexto conversacional.
- Ruido en el dataset.
- Diferencias culturales o lingüísticas.

El análisis de errores es una parte fundamental de la práctica. No se trata solo de decir “el modelo se equivoca”, sino de intentar entender qué tipo de fenómeno lingüístico o técnico hay detrás del error.

---

## 13. Discusión esperada

La práctica debe terminar con una discusión crítica.

Algunas cuestiones que deberían aparecer:

- ¿Cuál ha sido el mejor modelo?
- ¿Cuál ha sido el mejor modelo en relación rendimiento/coste?
- ¿Cuál sería más fácil de explicar a una persona no técnica?
- ¿Cuál sería más fácil de desplegar en producción?
- ¿Qué modelo necesita más datos?
- ¿Qué modelo parece más sensible al preprocesamiento?
- ¿Qué modelo parece generalizar mejor?
- ¿Qué modelo falla más en textos ambiguos?
- ¿La limpieza ha ayudado siempre?
- ¿Hay casos donde limpiar demasiado empeora?
- ¿Los embeddings neuronales han aportado algo frente a TF-IDF?
- ¿El modelo preentrenado o LLM justifica su mayor complejidad?
- ¿Qué haríais diferente si tuvierais más tiempo, más datos o más capacidad de cómputo?

Se valorará especialmente que las conclusiones no sean simplistas. Por ejemplo, no basta con decir:

> “BERT es mejor porque tiene más accuracy.”

Una conclusión más interesante sería:

> “El modelo preentrenado obtiene el mejor Macro-F1, especialmente en la clase minoritaria, probablemente porque captura mejor el contexto y las negaciones. Sin embargo, el modelo TF-IDF + Linear SVM queda bastante cerca, entrena mucho más rápido y es más interpretable. Para un prototipo rápido usaríamos el modelo clásico; para un sistema final con más recursos valoraríamos el modelo preentrenado.”

---

# Rúbrica de evaluación

## 1. Comprensión del problema y análisis del dataset — 15%

Se valorará que el grupo entienda bien el problema de clasificación y las características del dataset.

Aspectos esperados:

- Descripción clara de la tarea.
- Justificación del dataset elegido.
- Análisis de distribución de clases.
- Exploración de ejemplos reales.
- Identificación de ruido propio de redes sociales.
- Análisis de longitud de textos, hashtags, menciones, URLs, emojis y duplicados.
- Reflexión sobre posibles problemas de etiquetado o ambigüedad.

Máxima puntuación si el análisis exploratorio sirve realmente para tomar decisiones posteriores.

---

## 2. Limpieza y preprocesamiento — 20%

Se valorará especialmente esta parte.

Aspectos esperados:

- Pipeline de limpieza claro y reproducible.
- Comparación de al menos dos niveles de limpieza.
- Justificación de cada decisión.
- Tratamiento razonado de menciones, URLs, hashtags, emojis, mayúsculas, signos y stopwords.
- Evitar decisiones automáticas sin reflexión.
- Análisis del impacto de la limpieza en los resultados.
- Cuidado con errores típicos como eliminar negaciones importantes.

Máxima puntuación si se demuestra que el grupo entiende que el preprocesamiento puede ayudar o perjudicar dependiendo del modelo y del problema.

---

## 3. Modelo clásico de NLP — 15%

Se valorará la construcción de un baseline sólido.

Aspectos esperados:

- Uso correcto de CountVectorizer o TF-IDF.
- Pruebas con unigramas y bigramas.
- Entrenamiento de al menos un modelo clásico adecuado.
- Comparación de varias configuraciones.
- Evaluación con métricas adecuadas.
- Interpretación de los resultados.
- Análisis de ventajas e inconvenientes del enfoque clásico.

Máxima puntuación si el baseline clásico está bien trabajado y no aparece como un mero trámite.

---

## 4. Modelo de red neuronal — 15%

Se valorará la implementación y explicación de un modelo neuronal.

Aspectos esperados:

- Tokenización adecuada.
- Elección razonada de vocabulario y longitud máxima.
- Uso de embeddings.
- Arquitectura explicada.
- Entrenamiento con validación.
- Control básico del sobreajuste.
- Curvas de entrenamiento.
- Comparación frente al modelo clásico.
- Interpretación de si la red neuronal aporta mejora real o no.

Máxima puntuación si el grupo explica bien no solo el código, sino también qué está aprendiendo el modelo y por qué podría funcionar mejor o peor que TF-IDF.

---

## 5. Modelo avanzado o LLM — 15%

Se valorará la inclusión de un enfoque avanzado, ya sea fine-tuning de un modelo preentrenado o uso de un LLM mediante prompting.

Aspectos esperados si se usa fine-tuning:

- Elección justificada del modelo.
- Uso correcto del tokenizer.
- Preparación adecuada de los datos.
- Entrenamiento razonable.
- Evaluación comparable con el resto de modelos.
- Discusión sobre coste computacional y ventajas del preentrenamiento.

Aspectos esperados si se usa LLM:

- Prompt claro y controlado.
- Salida estructurada.
- Evaluación sobre una muestra representativa.
- Comparación con los modelos entrenados.
- Discusión sobre coste, latencia, privacidad, reproducibilidad y robustez.

Máxima puntuación si el modelo avanzado no se presenta como “magia”, sino como otra herramienta con ventajas y limitaciones.

---

## 6. Evaluación comparativa y análisis de errores — 15%

Se valorará la capacidad de comparar de forma rigurosa.

Aspectos esperados:

- Uso de métricas adecuadas.
- Macro-F1 cuando haya clases desbalanceadas.
- Matrices de confusión.
- Tabla comparativa final.
- Análisis por clase.
- Revisión manual de errores.
- Identificación de patrones de fallo.
- Comparación crítica entre modelos.
- Discusión de por qué un modelo funciona mejor en unos casos y peor en otros.

Máxima puntuación si la práctica demuestra pensamiento crítico y no se limita a ordenar modelos por accuracy.

---

## 7. Claridad, reproducibilidad y comunicación — 5%

Se valorará que el trabajo sea comprensible, ordenado y reproducible.

Aspectos esperados:

- Código organizado.
- Comentarios útiles.
- Explicaciones claras.
- Gráficos legibles.
- Resultados fáciles de seguir.
- Semillas de aleatoriedad cuando proceda.
- Separación clara entre entrenamiento, validación y test.
- Conclusiones conectadas con los experimentos realizados.

Máxima puntuación si una persona externa puede seguir el razonamiento completo del trabajo sin tener que adivinar qué se ha hecho ni por qué.

---

# Resultado esperado

Al terminar la práctica, el grupo debería ser capaz de defender una conclusión razonada del estilo:

> “Para este dataset concreto, el mejor rendimiento lo obtiene el modelo preentrenado, pero el modelo clásico con TF-IDF y Linear SVM ofrece una relación rendimiento/coste muy competitiva. La limpieza moderada mejora los resultados, mientras que la limpieza agresiva empeora el rendimiento porque elimina hashtags y emojis útiles. La LSTM mejora respecto a un baseline simple, pero no supera al modelo clásico bien ajustado. El LLM zero-shot resuelve algunos casos ambiguos, pero es más lento, menos reproducible y sensible al diseño del prompt.”

La práctica no busca únicamente “ganar” con el modelo más potente. Busca aprender a pensar como científicos de datos: formular hipótesis, probarlas, medir, comparar, equivocarse, analizar errores y justificar decisiones.
