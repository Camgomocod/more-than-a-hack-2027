# Fase 1: paper y concurso de posesión

## Alcance y conclusión

El artículo de Borghesi et al., *Using Temporal Convolutional Networks to estimate ball possession in soccer games*, Expert Systems With Applications 223 (2023), 119780, propone una base pertinente para clasificar posesión por equipo y juego detenido. Su resultado no es directamente transferible al concurso: cambian la calidad y la identidad del tracking, la supervisión y, sobre todo, la métrica. Este documento distingue observaciones de hipótesis; no presenta un modelo entrenado ni una mejora medida.

Fuentes: artículo local en `docs/pappers/` (ignorado por Git), secciones 3–5 y tablas 1–4; [README](../../README.md), [datos](../../DATA.md), [reto](../../CHALLENGE.md), [evaluación](../../EVALUATION.md), [reglas](../../RULES.md) y [entrega](../../SUBMISSION.md). Si hay discrepancias en las métricas, manda `evaluation/possession_metrics.py`.

## Comparación de condiciones

| Aspecto | Artículo | Concurso y consecuencia |
|---|---|---|
| Objetivo | Estados `dead`, `home`, `away` por instante. | `D`, `H`, `A` en frames evaluados; `X` es abstención y cubre también tiempos no evaluados. Balón suelto o disputado permanece con el último equipo en control. |
| Etiquetas | Operadores humanos etiquetaron partidos de una liga; unas 40 % de muestras son `dead`. | La referencia procede mecánicamente de tracking corregido. Los cinco minutos públicos solo sirven para verificar alineación, formato y significado, nunca para entrenar, validar o elegir configuraciones. |
| Datos | 35 partidos disponibles; eligen ocho con menos ausencias del balón en juego activo. Cuatro para entrenamiento, dos para validación y dos para test, con equipos y estadios diferentes; 100.000, 5.000 y 5.000 muestras respectivamente. | Un partido de concurso es el test. Datos humanos de otros partidos son admisibles si los organizadores pueden acceder a ellos y se documentan; también pseudoetiquetas automáticas del partido bajo las reglas. Aún no se ha identificado supervisión externa adecuada. |
| Tiempo | 64 fotogramas a una media de 16 fps; objetivo en posición 48 contando desde uno, con 47 anteriores y 16 posteriores. | Tactical es 25 fps: 64 fotogramas cubren 2,56 s, no los aproximadamente 4 s del artículo. Definir el contexto en segundos y no cruzar límites de tiempo. La inferencia acausal es viable al procesar el partido completo. |
| Objetos | Balón, árbitro, 22 titulares y 12 plazas adicionales: 36 slots; trayectoria por objeto, 11 canales. | 46 slots de tracking táctico, sin identidad de jugador estable, con pérdidas y duplicaciones. No asumir que un slot equivale a una persona ni que hay un árbitro o portero identificable como en el paper; verificar cada canal antes de usarlo. |
| Coordenadas | Metros, centradas; Akima para balón, escalado por eje, recorte y valor −2 para ausencias. | Coordenadas normalizadas 0–1; el campo de metadatos es 105 × 68 m y `games.py` utiliza 105,3 × 68 m. Elegir y documentar una convención para distancias; calcular velocidades solo con observaciones válidas y tiempos coherentes. |
| Calidad | Selección de partidos con menor ausencia del balón durante juego activo. | Balón ausente en 40,7 % de los frames evaluados. Interpolar ciegamente huecos largos podría inventar trayectorias; mantener máscaras y distinguir observación de interpolación. |
| Evaluación | Accuracy global y accuracies separadas de detenido/activo y posesión durante juego activo. | `spellF1` es el ranking: compara posesiones por IoU temporal y coincidencia uno a uno; `macroF1` desempata. Fragmentar tramos puede degradar mucho el ranking aunque la clasificación por frame sea buena. |

El paper emplea una TCN compartida por objeto y compara suma, autoatención y convoluciones con ordenación por distancia. Su mejor configuración de la tabla 2 usa dos ramas con autoatención independiente y obtiene **86,39 % de accuracy global**. La tabla 3 informa **89,2 %** para detenido/activo y **86,2 %** para posesión: son subtareas, no accuracy global. La tabla 4 informa **81,59 %** de accuracy global media en 20 partidos para el modelo completo; el experimento de cinco vecinos alcanza 79,41 %. Son protocolos diferentes y no se deben combinar como una sola medición.

El artículo no especifica inequívocamente todos los hiperparámetros necesarios para reproducir una configuración exacta. Tamaño de embedding, kernels, dilataciones, dropout, detalles de atención, ponderación de pérdidas y decisión final deberán justificarse como elecciones propias. La invariancia al orden de objetos tampoco corrige un cambio de identidad dentro de una trayectoria.

## Diagnóstico local reproducible

Ejecutar desde la raíz con `PYTHONPATH=. python docs/experimentos/diagnostico_tracking.py`. El script lee únicamente ATD y metadatos locales y muestra agregados; no consulta etiquetas. Resultados observados:

- ATD táctico: **199.319 registros**, IDs consecutivos y ordenados 1–199.319; `video_frame = frame_id - 1`. `ffprobe` confirma 199.319 frames de vídeo a 25 fps.
- Bloques evaluados publicados: 11.185–90.651 y 110.808–190.197, ambos inclusivos. Se usa el límite publicado, no etiquetas, para calcular los agregados de juego.
- Balón ausente en **85.831/199.319** frames totales y **64.643/158.857** evaluados. En estos últimos hay 150 rachas de ausencia: 17 de hasta 25 frames y 133 más largas; la máxima dura 5.537 frames. Las rachas se cortan en los límites de cada tiempo.
- Media de **20,49 slots visibles** durante los bloques evaluados; 3.749 frames superan 22 slots. Hay 1.219 fragmentos visibles de slots y 43 slots reaparecen. Un fragmento es una secuencia continua visible dentro de un tiempo; estas cifras muestran interrupciones, **no prueban ni cuantifican cambios de identidad**.
- Los vídeos descargados usan los sufijos `-001` y `-002`, mientras que `games.py` espera nombres sin ellos; el diagnóstico ATD no necesita renombrarlos. El ATD y el PDF local están ignorados por Git. No publicar datos del partido ni derivados que permitan reconstruirlos.

## Consecuencias para las siguientes fases

1. **Supervisión primero.** Examinar IDSSE como candidato de datos externos (véase abajo) y verificar su compatibilidad antes de entrenar o comparar modelos. Sin un conjunto independiente adecuado no se podrá afirmar que una configuración supera otra en generalización.
2. **Control del tracking.** Estudiar continuidad e identidad antes de aplicar TCN por objeto. Evaluar ventanas acausales en segundos, máscaras, límites de interpolación y velocidades válidas, sin convertir ausencia del balón automáticamente en `D`.
3. **Experimentos supervisados por fase.** Empezar con agregación por suma como control, después dos ramas con atención independiente y pérdida de posesión enmascarada por la etiqueta verdadera. Fijar configuraciones y particiones por partido antes de comparar. La atención y la decodificación temporal son hipótesis, no mejoras demostradas aquí.
4. **Evaluación y entrega.** En datos externos compatibles calcular `spellF1` y `macroF1` con el evaluador oficial, además de métricas auxiliares; comparar decisión por frame y decodificación temporal sin ajustar con los cinco minutos públicos. `X` cuenta como fallo en `macroF1` y se rellena dentro de cada tiempo para `spellF1`. El baseline publicado de partido completo tiene `spellF1` 0,2433, cifra descriptiva, no una medición del modelo propuesto.

La entrega final requiere CSV `frame,label` con 199.319 filas ordenadas y etiquetas `H/A/D/X`, un repositorio público que regenere exactamente el archivo y un documento técnico que explique activos, datos de entrenamiento, reglas de posesión, limitaciones y reproducción. No se ha generado ninguna entrega en esta fase.

## Candidato externo: IDSSE

El conjunto [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data) contiene siete partidos de Bundesliga 2022/23, tracking a 25 fps y eventos, con licencia CC BY 4.0. El [artículo de Bassek et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/) describe, para el balón, `BallStatus` (0 inactivo, 1 activo) y `BallPossession` (1 local, 2 visitante). Es un candidato especialmente pertinente porque permitiría construir una referencia por frame, no solo aproximarla con eventos aislados.

El mapeo inicial es `BallStatus=0 → D`, `BallStatus=1` y `BallPossession=1 → H`, y `BallStatus=1` y `BallPossession=2 → A`. Tratar valores ausentes o inesperados como desconocidos, sin inventar etiquetas ni incluirlos en métricas hasta definir su tratamiento. Evitar usar `BallStatus` o `BallPossession` como entrada al modelo: son objetivos y provocarían fuga de información. La semántica y la sincronización con el resto de objetos aún necesitan verificación detallada.

Reservar partidos completos para validación y prueba independientes, nunca ventanas solapadas del mismo partido entre particiones. Comparar `spellF1` y `macroF1` con el código oficial adaptado a la longitud y límites de cada partido; no usar los cinco minutos públicos del concurso para seleccionar configuraciones. El tracking de IDSSE puede tener calidad e identidades distintas al ATD automático del concurso, por lo que un buen resultado externo no garantiza transferencia; estudiar ausencias y fragmentación antes de diseñar perturbaciones.

La ficha de Hugging Face enumera dos identificadores (`J03WPF`, `J03WQF`) que no aparecen en los archivos listados; estos contienen `J03WQQ` y `J03WR9`, también presentes en la [documentación de kloppy](https://kloppy.pysport.org/reference/providers/sportec/). Verificar el inventario efectivo al descargar. Registrar procedencia, versión, autoría y atribución CC BY en la documentación; mantener la distinción entre estos datos abiertos y los activos restringidos del concurso.

### Inspección de la descarga local

En `alternative_data/data/idsse/` hay siete juegos completos de archivos: metadatos, eventos y posiciones para `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ` y `J03WR9`. La descarga confirma la discrepancia de identificadores señalada arriba. Los archivos de posiciones totalizan aproximadamente 2,62 GB; el directorio figura como no seguido por Git y no está ignorado, por lo que no debe añadirse accidentalmente.

Un recorrido de las líneas de los XML de posiciones encontró los siguientes recuentos de frames del balón, clasificados según el mapeo anterior:

| Partido | Frames | H activo | A activo | D |
|---|---:|---:|---:|---:|
| J03WMX | 145.967 | 39.111 | 46.258 | 60.598 |
| J03WN1 | 141.561 | 36.421 | 31.956 | 73.184 |
| J03WOH | 137.214 | 39.126 | 32.819 | 65.269 |
| J03WOY | 142.536 | 37.812 | 38.474 | 66.250 |
| J03WPY | 146.211 | 51.192 | 30.268 | 64.751 |
| J03WQQ | 142.345 | 37.030 | 47.322 | 57.993 |
| J03WR9 | 146.810 | 53.239 | 32.077 | 61.494 |

En los 1.002.644 registros del balón aparecen únicamente las cuatro combinaciones esperadas de estado 0/1 y posesión 1/2, y todos llevan atributos X e Y. Los números de frame del balón son consecutivos dentro de cada tiempo; comienzan en 10000 y 100000 respectivamente. Esto no demuestra que las coordenadas sean siempre correctas ni que todos los jugadores estén alineados. La presencia de posesión 1 o 2 durante juego detenido confirma que `BallStatus` debe prevalecer para generar `D`. Los frames están agrupados por objeto y tiempo, no ordenados globalmente por frame.

Cinco de los siete partidos incluyen al Fortuna Düsseldorf: la separación por partido evita fuga directa, pero las particiones siguen compartiendo equipo en muchos casos. Se debe informar esta dependencia y no interpretar siete partidos como siete contextos totalmente independientes. La revisión no ha validado aún cada XML completo, la integridad criptográfica de los archivos, la precisión semántica de los cambios de posesión ni el acoplamiento de todos los jugadores con la línea temporal del balón.
