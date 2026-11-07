# Reporte EDA 01: Análisis del Ground Truth y Descomposición Jerárquica

*Basado en el framework de Borghesi et al. (2023), Expert Systems with Applications.*

## 1. Resumen General del Conjunto Calificado

- **Frames Calificados:** 7,500 (equivalente a 300.0 s / 5.0 minutos a 25 fps).
- **Rango de Video:** Frames `11185` a `18684`.
- **Total de Segmentos:** 21 (17 posesiones activas, 4 detenciones de balón muerto).

## 2. Descomposición Jerárquica ($DA$ y $POSS$)

| Nivel de Decisión | Clase / Estado | Frames | % Nivel | % Total Calificado |
|---|---|---|---|---|
| **1. Dead vs Alive ($DA$)** | **Alive** (`H` + `A`) | 6,031 | 80.41% | 80.41% |
| | **Dead** (`D`) | 1,469 | 19.59% | 19.59% |
| **2. Posesión ($POSS \mid \text{Alive}$)** | **Home** (`H` - Barcelona) | 3,454 | 57.27% | 46.05% |
| | **Away** (`A` - Athletic) | 2,577 | 42.73% | 34.36% |

> [!IMPORTANT]
> **Implicación directa para el modelado:** La probabilidad a priori de juego activo es del 80.4%, y cuando el juego está vivo el Barcelona domina el 57.3% del tiempo en este tramo. Un clasificador que intente resolver todo en una sola capa confunde los estados de parada con posesión defensiva.

## 3. Caracterización de las 17 Posesiones Reales

| Métrica | Posesiones Globales | Posesiones Home (`H`) | Posesiones Away (`A`) | Balón Muerto (`D`) |
|---|---|---|---|---|
| **Número de Spells** | 17 | 8 | 9 | 4 |
| **Duración Media (s)** | **14.19 s** | 17.27 s | 11.45 s | 14.69 s |
| **Desviación Estándar** | 14.30 s | 17.68 s | 10.84 s | 6.59 s |
| **Mínima Duración** | 0.76 s (19 frames) | 0.76 s | 2.84 s | 5.52 s |
| **Mediana (p50)** | 7.56 s | 9.96 s | 7.44 s | 16.40 s |
| **Percentil 90 (p90)** | 38.09 s | 41.88 s | 23.14 s | 19.80 s |
| **Máxima Duración** | 42.44 s | 42.44 s | 35.72 s | 20.44 s |

### Tabla Cronológica de Posesiones y Detenciones en el Ground Truth

| # | Tipo | Label | Frame Inicio | Frame Fin | Duración (frames) | Duración (s) |
|---|---|---|---|---|---|---|
| 1 | Posesión | `A` | 11185 | 11350 | 166 | 6.64 s |
| 2 | Posesión | `H` | 11351 | 11552 | 202 | 8.08 s |
| 3 | Posesión | `A` | 11553 | 11623 | 71 | 2.84 s |
| 4 | Posesión | `H` | 11624 | 12664 | 1041 | 41.64 s |
| 5 | Posesión | `A` | 12665 | 12850 | 186 | 7.44 s |
| 6 | Balón Muerto | `D` | 12851 | 12988 | 138 | 5.52 s |
| 7 | Posesión | `H` | 12989 | 13703 | 715 | 28.60 s |
| 8 | Posesión | `A` | 13704 | 13784 | 81 | 3.24 s |
| 9 | Posesión | `H` | 13785 | 14845 | 1061 | 42.44 s |
| 10 | Posesión | `A` | 14846 | 15345 | 500 | 20.00 s |
| 11 | Posesión | `H` | 15346 | 15364 | 19 | 0.76 s |
| 12 | Posesión | `A` | 15365 | 15765 | 401 | 16.04 s |
| 13 | Posesión | `H` | 15766 | 15798 | 33 | 1.32 s |
| 14 | Posesión | `A` | 15799 | 15888 | 90 | 3.60 s |
| 15 | Balón Muerto | `D` | 15889 | 16399 | 511 | 20.44 s |
| 16 | Posesión | `A` | 16400 | 17292 | 893 | 35.72 s |
| 17 | Posesión | `H` | 17293 | 17379 | 87 | 3.48 s |
| 18 | Balón Muerto | `D` | 17380 | 17837 | 458 | 18.32 s |
| 19 | Posesión | `A` | 17838 | 18026 | 189 | 7.56 s |
| 20 | Balón Muerto | `D` | 18027 | 18388 | 362 | 14.48 s |
| 21 | Posesión | `H` | 18389 | 18684 | 296 | 11.84 s |

## 4. Dinámica Temporal y Matriz de Transiciones de Estado

### A. Matriz de Transición Frame a Frame $P(S_t \mid S_{t-1})$

| Estado Anterior ($S_{t-1}$) | $P(H_t)$ | $P(A_t)$ | $P(D_t)$ | Interpretación |
|---|---|---|---|---|
| **H** | 0.99797 | 0.00174 | 0.00029 | Inercia de permanencia: 99.80% |
| **A** | 0.00233 | 0.99651 | 0.00116 | Inercia de permanencia: 99.65% |
| **D** | 0.00136 | 0.00136 | 0.99728 | Inercia de permanencia: 99.73% |

> [!TIP]
> **Regla fundamental para Viterbi / HMM:** La probabilidad diagonal es superior al **98.8%** en los tres estados. Un cambio directo $H \to A$ ocurre solo con probabilidad **0.00087** por frame. Esto fundamenta el uso de penalización de transición para maximizar `spellF1`.

### B. Tipos de Transiciones entre Bloques (Spell-Level)

| Transición | Frecuencia | % Transiciones | Descripción |
|---|---|---|---|
| `A -> H` | 6 | 30.0% | Transición observada en el partido |
| `H -> A` | 6 | 30.0% | Transición observada en el partido |
| `A -> D` | 3 | 15.0% | Transición observada en el partido |
| `D -> H` | 2 | 10.0% | Transición observada en el partido |
| `D -> A` | 2 | 10.0% | Transición observada en el partido |
| `H -> D` | 1 | 5.0% | Transición observada en el partido |

---
*Generado automáticamente por `eda_01_ground_truth.py`.*