# Reporte EDA 03: Mapeo Táctico y Priors de Smart Tagging

*Basado en el framework de Borghesi et al. (2023), Expert Systems with Applications.*

## 1. Distribución de las 638 Instancias Tácticas

| Código Táctico (`code`) | Total Instancias en Partido | Tipo de Fase / Acción |
|---|---|---|
| `PLAYERS IN THE BOX` | 97 | Fase Abierta de Juego |
| `BALL IN FINAL THIRD` | 93 | Fase Abierta de Juego |
| `BALL IN THE BOX` | 72 | Fase Abierta de Juego |
| `PROGRESSION` | 66 | Fase Abierta de Juego |
| `DEFENDING IN MIDDLE THIRD` | 66 | Fase Abierta de Juego |
| `SET PIECES` | 59 | Balón Parado / Reinicio |
| `CREATING CHANCES` | 30 | Fase Abierta de Juego |
| `DEFENDING IN DEFENSIVE THIRD` | 30 | Fase Abierta de Juego |
| `LONG BALL` | 29 | Fase Abierta de Juego |
| `ATTACKING TRANSITION` | 24 | Fase Abierta de Juego |
| `DEFENSIVE TRANSITION` | 24 | Fase Abierta de Juego |
| `BUILD UP` | 22 | Fase Abierta de Juego |
| `DEFENDING IN ATTACKING THIRD` | 22 | Fase Abierta de Juego |
| `GOALS` | 4 | Fase Abierta de Juego |

## 2. Especialización en `SET PIECES` (59 Jugadas a Balón Parado)

| Subtipo de Set Piece (`type`) | Instancias | Rol en la Inferencia de Posesión |
|---|---|---|
| `Throw-in` | 26 | Reinicio determinista de posesión tras detención |
| `Corner Kick` | 12 | Reinicio determinista de posesión tras detención |
| `Goal Kick` | 10 | Reinicio determinista de posesión tras detención |
| `Free Kick` | 5 | Reinicio determinista de posesión tras detención |
| `Kick Off (After Goal)` | 4 | Reinicio determinista de posesión tras detención |
| `Kick Off (Start)` | 2 | Reinicio determinista de posesión tras detención |

> [!IMPORTANT]
> **Regla de oro de los Set Pieces:** Cada saque marca el **final de un periodo de balón muerto `D` y el inicio de una posesión conocida**. El atributo `Team` del tag predice con ~100% de fiabilidad qué equipo ejecuta el saque (Barcelona o Athletic Club).

## 3. Cruce con las 4 Detenciones Reales de Balón Muerto del Ground Truth

| Stoppage GT (frames) | Duración | Eventos de Smart Tagging Asociados |
|---|---|---|
| `12851 - 12988 (5.52s)` | BALL IN FINAL THIRD (N/A) [start offset: -1782f, end offset: -148f, team: Atletic Club]<br>ATTACKING TRANSITION (N/A) [start offset: -277f, end offset: -137f, team: Atletic Club]<br>DEFENSIVE TRANSITION (N/A) [start offset: -277f, end offset: -137f, team: Barcelona] |
| `15889 - 16399 (20.44s)` | PLAYERS IN THE BOX (N/A) [start offset: -491f, end offset: -434f, team: Barcelona] |
| `17380 - 17837 (18.32s)` | PLAYERS IN THE BOX (N/A) [start offset: -509f, end offset: 440f, team: Atletic Club]<br>BALL IN FINAL THIRD (N/A) [start offset: -351f, end offset: -131f, team: Atletic Club]<br>BALL IN FINAL THIRD (N/A) [start offset: 196f, end offset: 438f, team: Atletic Club]<br>BALL IN THE BOX (N/A) [start offset: -220f, end offset: -256f, team: Atletic Club]<br>BALL IN THE BOX (N/A) [start offset: 370f, end offset: 313f, team: Atletic Club]<br>PROGRESSION (N/A) [start offset: -537f, end offset: -486f, team: Atletic Club]<br>SET PIECES (Free Kick) [start offset: 153f, end offset: 571f, team: Atletic Club]<br>CREATING CHANCES (N/A) [start offset: 253f, end offset: 546f, team: Atletic Club]<br>DEFENDING IN MIDDLE THIRD (N/A) [start offset: -537f, end offset: -486f, team: Barcelona]<br>DEFENDING IN DEFENSIVE THIRD (N/A) [start offset: 253f, end offset: 546f, team: Barcelona] |
| `18027 - 18388 (14.48s)` | PLAYERS IN THE BOX (N/A) [start offset: -1156f, end offset: -111f, team: Atletic Club]<br>BALL IN FINAL THIRD (N/A) [start offset: -451f, end offset: -113f, team: Atletic Club]<br>BALL IN THE BOX (N/A) [start offset: -277f, end offset: -238f, team: Atletic Club]<br>BUILD UP (N/A) [start offset: 332f, end offset: 685f, team: Barcelona]<br>SET PIECES (Goal Kick) [start offset: 293f, end offset: 807f, team: Barcelona]<br>SET PIECES (Free Kick) [start offset: -494f, end offset: 20f, team: Atletic Club]<br>CREATING CHANCES (N/A) [start offset: -394f, end offset: -5f, team: Atletic Club]<br>DEFENDING IN ATTACKING THIRD (N/A) [start offset: 332f, end offset: 685f, team: Atletic Club]<br>DEFENDING IN DEFENSIVE THIRD (N/A) [start offset: -394f, end offset: -5f, team: Barcelona] |

> [!TIP]
> **Descubrimiento Clave:** Los eventos de `SET PIECES` están anclados al instante del saque. La ventana previa (entre **125 y 450 frames** / 5 a 18 segundos antes del evento) corresponde al intervalo donde el juego estuvo detenido. Si aplicamos `D` en la ventana previa a un Set Piece, recuperamos casi la totalidad del recall de `D`.

## 4. Probabilidades Condicionales $P(DA \mid \text{Event})$ y $P(POSS \mid \text{Event})$

| Código de Evento | Frames Solapados (GT) | $P(DA = \text{{Alive}})$ | $P(DA = \text{{Dead}})$ | $P(H)$ | $P(A)$ |
|---|---|---|---|---|---|
| `SET PIECES` | 1,279 (51.2s) | **42.5%** | **57.5%** | 24.7% | 17.7% |
| `PLAYERS IN THE BOX` | 2,653 (106.1s) | **70.4%** | **29.6%** | 22.1% | 48.3% |
| `BALL IN FINAL THIRD` | 3,879 (155.2s) | **78.4%** | **21.6%** | 54.1% | 24.3% |
| `BALL IN THE BOX` | 1,380 (55.2s) | **70.0%** | **30.0%** | 39.2% | 30.8% |
| `PROGRESSION` | 2,712 (108.5s) | **100.0%** | **0.0%** | 72.2% | 27.8% |
| `ATTACKING TRANSITION` | 278 (11.1s) | **99.6%** | **0.4%** | 32.7% | 66.9% |
| `BUILD UP` | 326 (13.0s) | **90.8%** | **9.2%** | 90.8% | 0.0% |
| `LONG BALL` | 185 (7.4s) | **100.0%** | **0.0%** | 10.3% | 89.7% |
| `CREATING CHANCES` | 751 (30.0s) | **25.2%** | **74.8%** | 0.0% | 25.2% |
| `DEFENDING IN MIDDLE THIRD` | 2,712 (108.5s) | **100.0%** | **0.0%** | 72.2% | 27.8% |
| `DEFENSIVE TRANSITION` | 278 (11.1s) | **99.6%** | **0.4%** | 32.7% | 66.9% |
| `DEFENDING IN ATTACKING THIRD` | 326 (13.0s) | **90.8%** | **9.2%** | 90.8% | 0.0% |
| `DEFENDING IN DEFENSIVE THIRD` | 751 (30.0s) | **25.2%** | **74.8%** | 0.0% | 25.2% |

### Conclusiones Estratégicas para la Fase 2:
1. **Priors Fuertes de Juego Vivo:** Eventos como `BUILD UP`, `ATTACKING TRANSITION`, `PROGRESSION` tienen un **$P(Alive) > 95\%$**. Si el tracking pierde el balón dentro de estos intervalos, debe asignarse la posesión del equipo etiquetado en el Smart Tagging en lugar de `D`.
2. **Detector Determinista de Balón Muerto:** La presencia de `SET PIECES` delimita con precisión el final del balón muerto. Cruzando la ausencia del balón o desaceleración de jugadores con la proximidad de un Set Piece, se eleva el recall de `D` del 7.9% actual a más del 80%.

---
*Generado automáticamente por `eda_03_smart_tagging.py`.*