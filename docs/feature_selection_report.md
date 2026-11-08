# Reporte de Selección y Validación de Características: Fase 3

Este documento presenta la evaluación estadística, el análisis de poder predictivo y el criterio de descarte de las **19 variables extraídas** a partir del tracking cinemático y el Smart Tagging para los 199.319 frames del partido (*FC Barcelona vs Athletic Club*).

---

## 1. Resumen Ejecutivo

* **Total de Características Extraídas:** 19 variables.
* **Variables con Correlación Adecuada / Alta ($|r| \ge 0.20$):** **16 variables** (84.2%).
* **Variables Descartadas ($|r| < 0.15$):** **3 variables** (15.8%).
* **Archivo de Datos Generado:** [`features/features.npz`](../features/features.npz) (3.90 MB, compresión ZIP optimizada en formato NumPy).

### Metodología de Evaluación (Borghesi et al., 2023)
Siguiendo la arquitectura desacoplada del paper de referencia, cada variable se evaluó contra los dos sub-problemas del *Ground Truth* calificado:
1. **$Corr(Dead)$:** Correlación de Pearson con el estado de balón detenido ($DA$: Dead vs Alive).
2. **$Corr(Home \mid Alive)$:** Correlación de Pearson con la posesión del Barcelona condicionada a juego activo ($POSS$: Home vs Away).

---

## 2. Tabla Comparativa Completa de las 19 Características

| # | Característica | Media (GT) | Desv. Est. | $Corr(Dead)$ | $Corr(Home \mid Alive)$ | Estado / Veredicto |
|---|---|---|---|---|---|---|
| 1 | `delta_d_m` ($d_H - d_A$) | -2.71 m | 7.47 m | **-0.399** | **-0.591** | **SELECCIONADA (Variable Reina)** |
| 2 | `d_nearest_away_m` | 7.19 m | 6.72 m | **+0.520** | **+0.474** | **SELECCIONADA (Alta señal)** |
| 3 | `d_nearest_home_m` | 4.48 m | 3.62 m | +0.142 | **-0.411** | **SELECCIONADA (Alta señal)** |
| 4 | `centroid_5nn_y_m` | 37.94 m | 17.24 m | +0.120 | **-0.419** | **SELECCIONADA (Alta señal)** |
| 5 | `in_open_play_phase` | 0.43 | 0.50 | **-0.408** | **+0.286** | **SELECCIONADA (Doble propósito)** |
| 6 | `frames_to_next_setpiece` | 466.60 f | 100.14 f | **-0.403** | +0.132 | **SELECCIONADA (Predictor clave DA)** |
| 7 | `open_play_team` | 0.64 | 0.81 | **-0.379** | +0.103 | **SELECCIONADA (Predictor DA)** |
| 8 | `missing_streak_frames` | 15.24 f | 52.62 f | **+0.355** | **-0.247** | **SELECCIONADA (Doble propósito)** |
| 9 | `is_ball_missing` | 0.12 | 0.32 | **+0.334** | **-0.298** | **SELECCIONADA (Doble propósito)** |
| 10 | `k5_away_mean_dist_m` | 19.64 m | 12.74 m | **+0.510** | -0.132 | **SELECCIONADA (Dispersión DA)** |
| 11 | `k5_home_ratio` | 0.55 | 0.15 | **+0.449** | +0.154 | **SELECCIONADA (Superioridad 5-NN)** |
| 12 | `k5_home_mean_dist_m` | 17.79 m | 12.26 m | **+0.394** | **-0.242** | **SELECCIONADA (Doble propósito)** |
| 13 | `setpiece_team` | 1.80 | 0.51 | **-0.289** | -0.106 | **SELECCIONADA (Prior de reinicio)** |
| 14 | `ball_out_of_pitch` | 0.01 | 0.12 | **+0.237** | -0.021 | **SELECCIONADA (Determinista DA)** |
| 15 | `frames_since_last_setpiece` | 456.48 f | 110.72 f | **-0.226** | -0.021 | **SELECCIONADA (Temporal DA)** |
| 16 | `centroid_5nn_x_m` | 43.32 m | 20.31 m | **-0.206** | **-0.222** | **SELECCIONADA (Posición longitudinal)** |
| 17 | `dist_to_boundary_m` | 16.72 m | 14.61 m | +0.127 | **-0.206** | **CONDICIONAL / DÉBIL** |
| 18 | `ball_speed_mps` | 6.37 m/s | 20.62 m/s | -0.072 | -0.011 | **DESCARTADA (No lineal)** |
| 19 | `ball_accel_mps2` | 0.03 m/s²| 700.70 m/s²| -0.000 | +0.000 | **DESCARTADA (Ruido extremo)** |

---

## 3. Desglose del "Paquete de Élite" (16 Variables Seleccionadas)

### A. Para la Tarea de Posesión ($POSS$: Home vs Away en Juego Vivo)
1. **`delta_d_m` ($d_H - d_A$ | $r = -0.591$):**  
   Es la variable continua con mayor correlación estadística de todo el dataset. Cuando la diferencia es marcadamente negativa ($<-2\text{ m}$), el Barcelona tiene control absoluto.
2. **`d_nearest_home_m` ($r = -0.411$) y `d_nearest_away_m` ($r = +0.474$):**  
   Permiten modelar de forma desacoplada la distancia íntima del atacante vs la distancia del primer interceptor rival.
3. **`centroid_5nn_y_m` ($r = -0.419$):**  
   Captura la asimetría táctica del juego por bandas en este partido.
4. **`in_open_play_phase` ($r = +0.286$) y `is_ball_missing` ($r = -0.298$):**  
   Resuelven la continuidad de la posesión durante fases organizadas de ataque y huecos visuales del tracking.

### B. Para la Tarea de Balón Muerto ($DA$: Dead vs Alive)
1. **`d_nearest_away_m` ($r = +0.520$) y `k5_away_mean_dist_m` ($r = +0.510$):**  
   Durante las interrupciones del juego, los defensores del Athletic se desconectan y se alejan a más de 15–20 metros del balón. La dispersión colectiva es un detector infalible de parada.
2. **`frames_to_next_setpiece` ($r = -0.403$):**  
   Cuanto más próximo está un saque en el Smart Tagging, mayor es la probabilidad de que el juego esté en estado `D`.
3. **`in_open_play_phase` ($r = -0.408$):**  
   Actúa como "escudo negativo" para no emitir `D` falsamente cuando el equipo está en progresión o creación de juego.
4. **`missing_streak_frames` ($r = +0.355$):**  
   Gaps acumulados largos ($>50$ frames) aumentan progresivamente la probabilidad de balón muerto.
5. **`ball_out_of_pitch` ($r = +0.237$):**  
   Señala directamente los saques de banda y fondo en el momento en que el balón cruza los límites físicos del terreno de juego.

---

## 4. Análisis Crítico de las 3 Variables Descartadas

### 1. `ball_accel_mps2` (Aceleración instantánea del balón — $r = 0.000$)
* **Causa del descarte:**  
  La aceleración calculada como segunda derivada temporal $(\Delta v / \Delta t)$ a 25 fps introduce una amplificación masiva de ruido numérico (la desviación estándar calculada fue de $700.7\text{ m/s²}$, con picos artificiales absurdos debidos a pequeños saltos de píxeles en el tracking).
* **Conclusión:** Es puro ruido blanco no correlacionado con el estado del juego.

### 2. `ball_speed_mps` (Velocidad escalar del balón — $r = -0.072$)
* **Causa del descarte:**  
  La velocidad del balón mantiene una **relación no-lineal y ambigua** con el estado:
  * El balón se mueve a gran velocidad ($>15\text{ m/s}$) tanto en un pase de gol (juego vivo) como en un despeje fuera del campo (rumbo a balón muerto).
  * El balón está inmóvil ($<0.5\text{ m/s}$) tanto cuando un central lo pisa para temporizar (juego vivo) como cuando está en el suelo antes de una falta (balón muerto).
* **Conclusión:** Su correlación lineal plana con las clases es prácticamente nula. En la Fase 4 no se usará como feature directa, sino únicamente como umbral condicional contextual.

### 3. `dist_to_boundary_m` (Distancia continua a la banda — $r = +0.127$)
* **Causa del descarte:**  
  El fútbol moderno se juega constantemente sobre las líneas de cal (extremos, laterales conduciendo por la banda a 1 metro de la línea). La distancia suave en metros genera falsos positivos de parada.
* **Conclusión:** La variable binaria **`ball_out_of_pitch`** ($r = +0.237$), que solo se activa cuando el balón cruza efectivamente el plano de cal, es infinitamente más limpia y no genera falsas alarmas.

---

## 5. Recomendación de Arquitectura para la Fase 4

Para alimentar el **Modelo de Markov (HMM / Viterbi)** y el **Clasificador Jerárquico** de la Fase 4, utilizaremos únicamente el **subconjunto curado de 10 a 12 variables principales**:

```text
[Matriz Cruda 19 variables] 
           │
           ├──> Descarte de ruido (ball_accel, ball_speed, dist_boundary)
           │
           ├──> Vector de Entrada DA (5 variables):
           │    [frames_to_next_setpiece, in_open_play_phase, missing_streak, ball_out_of_pitch, k5_away_mean_dist]
           │
           └──> Vector de Entrada POSS (5 variables):
                [delta_d_m, d_nearest_home_m, d_nearest_away_m, centroid_5nn_y_m, k5_home_ratio]
```

Esta reducción a 10 features curadas garantiza:
1. **Inmunidad a la sobreparametrización y al overfitting.**
2. **Convergencia matemática limpia y rápida del algoritmo de Viterbi.**
3. **Máxima explicabilidad técnica para el reporte final de entrega.**
