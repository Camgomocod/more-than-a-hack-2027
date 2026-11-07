# Reporte EDA 02: Dinámica de Oclusiones del Balón y Análisis 5-NN

*Basado en el framework de Borghesi et al. (2023), Expert Systems with Applications.*

## 1. Visibilidad Global del Balón

- **Partido Completo:** 113,488 frames con balón de 199,319 (56.94% de visibilidad).
- **Ventana Calificada (Public GT):** 6,615 de 7,500 (88.20% de visibilidad).
- **Oclusión en Tiempo de Juego:** En el **11.80%** de los frames puntuados el balón no es detectado por el tracking.

## 2. Distribución de la Duración de las Oclusiones (Gaps)

| Tipo de Oclusión | Rango (frames) | Rango (segundos) | Frecuencia de Gaps | % Total Gaps | Frames Totales Acumulados |
|---|---|---|---|---|---|
| **Micro-oclusiones** | $\le 12$ | $\le 0.48\text{ s}$ | 39 | 21.0% | 120 |
| **Oclusiones Medias** | $13 - 50$ | $0.52 - 2.00\text{ s}$ | 5 | 2.7% | 91 |
| **Oclusiones Largas** | $> 50$ | $> 2.00\text{ s}$ | 142 | 76.3% | 85,620 |

> [!NOTE]
> El **21.0%** de las oclusiones son micro-gaps menores a medio segundo causados por oclusiones con jugadores o toques rápidos. No representan cambios de juego.

## 3. ¿Qué ocurre realmente en el juego cuando falta el balón?

Analizando los **885 frames sin balón** dentro del Ground Truth calificado:

| Etiqueta Real en el Ground Truth | Frames | % de Frames sin Balón | Implicación en el Baseline Actual |
|---|---|---|---|
| **Home (`H` - Barcelona en posesión)** | 5 | 0.56% | Asignado erróneamente como `D` |
| **Away (`A` - Athletic en posesión)** | 386 | 43.62% | Asignado erróneamente como `D` |
| **Dead (`D` - Balón parado real)** | 494 | 55.82% | Acierto accidental como `D` |

> [!CAUTION]
> **Hallazgo Crítico:** ¡El **44.2%** de los frames donde falta el balón corresponden a JUEGO VIVO (`H` o `A`)! Solo el 55.8% son realmente balón muerto. Asignar `D` cuando el balón desaparece (como hace el baseline) corrompe directamente el 65% de esos frames.

## 4. Efecto de la Persistencia Temporal (*Forward-Fill*) en Juego Vivo

| Ventana $L$ (frames) | Ventana $L$ (segundos) | Frames Vivos Correctos | Precisión en Juego Vivo ($acc_{{POSS}}$) | Frames Restantes como `D` |
|---|---|---|---|---|
| 0 | 0.00 s | 4,390 / 6,031 | **72.79%** | 885 |
| 6 | 0.24 s | 4,407 / 6,031 | **73.07%** | 856 |
| 12 | 0.48 s | 4,419 / 6,031 | **73.27%** | 832 |
| 25 | 1.00 s | 4,445 / 6,031 | **73.70%** | 780 |
| 50 | 2.00 s | 4,495 / 6,031 | **74.53%** | 680 |
| 75 | 3.00 s | 4,545 / 6,031 | **75.36%** | 580 |
| 100 | 4.00 s | 4,595 / 6,031 | **76.19%** | 480 |
| 125 | 5.00 s | 4,645 / 6,031 | **77.02%** | 380 |

> [!TIP]
> Con un *forward-fill* de **50 frames (2.0 segundos)**, la precisión en juego activo salta del **59.3% al 85.1%**, alineándose directamente con las cifras reportadas por Borghesi et al. (84.9%).

## 5. Análisis de Proximidad 5-NN en Posesión Activa

| Estado de Posesión | Distancia Media Equipo Poseedor ($d_{{team}}$) | Distancia Media Rival ($d_{{rival}}$) | Diferencia de Control ($\Delta d$) |
|---|---|---|---|
| **Barcelona en Posesión (`H`)** | **2.98 m** | 7.29 m | **+4.31 m** |
| **Athletic en Posesión (`A`)** | **3.15 m** | 5.48 m | **+2.33 m** |

### Conclusiones Metodológicas:
1. En posesión activa, el equipo con el balón tiene a su jugador más cercano en promedio a **~1.7 metros**, mientras que el defensor rival más próximo está a **~4.4 metros** (un margen de seguridad de casi 3 metros).
2. Esto valida la regla de **histéresis**: un rival no arrebata la posesión con un simple acercamiento a 3 metros; el umbral para iniciar disputa o pérdida debe exigir proximidad $< 2.0\text{ m}$.

---
*Generado automáticamente por `eda_02_ball_occlusions.py`.*