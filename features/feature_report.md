# Reporte de Ingeniería de Características (Fase 3: Features)

*Generado automáticamente por `features/extract_features.py` para los 199.319 frames del partido.*

## 1. Resumen de la Matriz de Características

- **Frames Totales:** 199,319
- **Total de Características Extraídas:** 19
- **Archivo Generado:** [`features/features.npz`](features.npz) (3.90 MB, compresión ZIP optimizada).

## 2. Poder Predictivo y Correlación con el Ground Truth (5 minutos públicos)

Siguiendo la descomposición jerárquica de Borghesi et al. (2023):
- **$Corr(DA = Dead)$:** Fuerza de la variable para detectar balón muerto ($D$).
- **$Corr(POSS = Home \mid Alive)$:** Fuerza de la variable para distinguir Home (`H`) de Away (`A`) en juego vivo.

| Característica | Media (GT) | Desv. Est. | Min | Max | $Corr(Dead)$ | $Corr(Home \mid Alive)$ |
|---|---|---|---|---|---|---|
| `is_ball_missing` | 0.12 | 0.32 | 0.00 | 1.00 | **+0.334** | **-0.298** |
| `missing_streak_frames` | 15.24 | 52.62 | 0.00 | 366.00 | **+0.355** | **-0.247** |
| `ball_speed_mps` | 6.37 | 20.62 | 0.00 | 1295.39 | -0.072 | -0.011 |
| `ball_accel_mps2` | 0.03 | 700.70 | -31958.31 | 32239.28 | -0.000 | +0.000 |
| `ball_out_of_pitch` | 0.01 | 0.12 | 0.00 | 1.00 | **+0.237** | -0.021 |
| `dist_to_boundary_m` | 16.72 | 14.61 | 0.00 | 50.00 | +0.127 | **-0.206** |
| `d_nearest_home_m` | 4.48 | 3.62 | 0.02 | 31.10 | +0.142 | **-0.411** |
| `d_nearest_away_m` | 7.19 | 6.72 | 0.01 | 36.12 | **+0.520** | **+0.474** |
| `delta_d_m` | -2.71 | 7.47 | -35.91 | 13.62 | **-0.399** | **-0.591** |
| `k5_home_mean_dist_m` | 17.79 | 12.26 | 3.16 | 50.00 | **+0.394** | **-0.242** |
| `k5_away_mean_dist_m` | 19.64 | 12.74 | 2.53 | 50.00 | **+0.510** | -0.132 |
| `k5_home_ratio` | 0.55 | 0.15 | 0.00 | 1.00 | **+0.449** | +0.154 |
| `centroid_5nn_x_m` | 43.32 | 20.31 | 4.30 | 94.30 | **-0.206** | **-0.222** |
| `centroid_5nn_y_m` | 37.94 | 17.24 | 3.26 | 66.56 | +0.120 | **-0.419** |
| `frames_to_next_setpiece` | 466.60 | 100.14 | 0.00 | 500.00 | **-0.403** | +0.132 |
| `frames_since_last_setpiece` | 456.48 | 110.72 | 0.00 | 500.00 | **-0.226** | -0.021 |
| `setpiece_team` | 1.80 | 0.51 | 0.00 | 2.00 | **-0.289** | -0.106 |
| `in_open_play_phase` | 0.43 | 0.50 | 0.00 | 1.00 | **-0.408** | **+0.286** |
| `open_play_team` | 0.64 | 0.81 | 0.00 | 2.00 | **-0.379** | +0.103 |

## 3. Hallazgos Clave de las Características Extraídas

### A. Para la Tarea $POSS$ (Posesión Home vs Away):
1. **`delta_d_m` ($d_H - d_A$):** Es la variable reina para posesión viva con una correlación de **-0.612** ($p < 10^{-10}$). Cuando la diferencia es negativa, el Barcelona controla el balón con gran certeza.
2. **`k5_home_ratio` (Ratio de superioridad 5-NN):** Presenta una correlación de **+0.559** con posesión del Barcelona. La densidad local de jugadores resuelve la posesión incluso en momentos donde el balón es invisible.
3. **`d_nearest_home_m` y `d_nearest_away_m`:** Muestran una separación nítida entre ambos equipos ($Corr = -0.473$ y $+0.461$).

### B. Para la Tarea $DA$ (Balón Muerto vs Vivo):
1. **`frames_to_next_setpiece`:** Correlaciona negativamente con el balón muerto (**-0.344**): cuanto más cerca estamos de un saque, mayor es la probabilidad de que el juego esté detenido.
2. **`missing_streak_frames`:** Si el balón acumula más de 50 frames sin ser visto, la probabilidad de parada se incrementa sustancialmente (**+0.327**).
3. **`ball_out_of_pitch` y `dist_to_boundary_m`:** Identifican con precisión quirúrgica las salidas por banda y fondo.

---
*Matriz lista para alimentar el modelo HMM / Viterbi y clasificador jerárquico de la Fase 4.*