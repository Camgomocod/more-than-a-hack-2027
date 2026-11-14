# Validación diagnóstica de `model_consensus`

**Advertencia:** esta validación es *in-sample* sobre la ventana pública de 5 minutos / 17 posesiones. No es una estimación del partido completo y no debe usarse para elegir hiperparámetros.

## Auditoría de procedencia

El historial de git muestra que el score público fue usado como objetivo de selección:

| commit | mensaje |
|---|---|
| `5485e0d` | improve spellF1 to 0.5730 and macroF1 to 0.8309 via dead-ball kinetic rest filter |
| `086d429` | achieve rank 1 spellF1 0.6400 and macroF1 0.8508 via consensus ensemble |
| `64d16b3` | reach all-time macroF1 0.8667 and accuracy 87.15% (spellF1 0.6400) via expanded dead-ball kinetic persistence |

Por tanto, cualquier intervalo calculado aquí sigue siendo optimista: el modelo fue elegido mirando estas mismas 17 posesiones.

## Resumen

| submission | spellF1 | macroF1 | transF1@12 | transF1@25 | poss | dur | frag | bootstrap spellF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `model_tcn_clean` | 0.5943 | 0.8312 | 0.3721 | 0.4186 | 18 | 13.95 s | 1.143 | [0.4229, 0.7543] |
| `model_ensemble` | 0.5730 | 0.8407 | 0.3043 | 0.3913 | 20 | 12.21 s | 1.286 | [0.4216, 0.7135] |
| `model_consensus` | 0.6400 | 0.8667 | 0.2791 | 0.4186 | 18 | 13.45 s | 1.143 | [0.4800, 0.7886] |

## Bootstrap pareado

Mismos resamples de posesiones del GT para todos los modelos. Intervalo percentil 5–95 y fracción de resamples en los que el primero gana.

| comparación | diferencia media | p05 | p95 | fracción A > B |
|---|---:|---:|---:|---:|
| model_ensemble vs model_tcn_clean | -0.0227 | -0.1020 | +0.0581 | 0.334 |
| model_consensus vs model_tcn_clean | +0.0451 | +0.0000 | +0.1143 | 0.879 |
| model_consensus vs model_ensemble | +0.0678 | +0.0278 | +0.1337 | 1.000 |

## `spellF1` por IoU

| submission | IoU 0.3 | IoU 0.4 | IoU 0.5 | IoU 0.6 | IoU 0.7 |
|---|---:|---:|---:|---:|---:|
| `model_tcn_clean` | 0.6857 | 0.6857 | 0.6857 | 0.5143 | 0.4000 |
| `model_ensemble` | 0.7027 | 0.7027 | 0.6486 | 0.4324 | 0.3784 |
| `model_consensus` | 0.7429 | 0.7429 | 0.7429 | 0.5143 | 0.4571 |

## Transiciones

| submission | tol | F1 | precisión | recall | latencia media | matched | missed | spurious |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `model_tcn_clean` | 12 | 0.3721 | 0.3478 | 0.4000 | 5.88 | 8 | 12 | 15 |
| `model_tcn_clean` | 25 | 0.4186 | 0.3913 | 0.4500 | 6.89 | 9 | 11 | 14 |
| `model_ensemble` | 12 | 0.3043 | 0.2692 | 0.3500 | 4.29 | 7 | 13 | 19 |
| `model_ensemble` | 25 | 0.3913 | 0.3462 | 0.4500 | 6.67 | 9 | 11 | 17 |
| `model_consensus` | 12 | 0.2791 | 0.2609 | 0.3000 | 4.17 | 6 | 14 | 17 |
| `model_consensus` | 25 | 0.4186 | 0.3913 | 0.4500 | 7.56 | 9 | 11 | 14 |

## Sensibilidad a una posesión

Jackknife: se elimina una posesión del GT y se recalcula `spellF1`. Sirve para ver cuánto depende el resultado de una sola posesión.

| submission | completo | mínimo | máximo | peor posesión |
|---|---:|---:|---:|---|
| `model_tcn_clean` | 0.5943 | 0.5529 | 0.6118 | #3 (H, 7.75 min) |
| `model_ensemble` | 0.5730 | 0.5333 | 0.5889 | #3 (H, 7.75 min) |
| `model_consensus` | 0.6400 | 0.6000 | 0.6588 | #3 (H, 7.75 min) |

## Lectura

- El bootstrap pareado mide si la diferencia se mantiene al remuestrear posesiones, pero **no corrige** el sesgo de haber elegido el modelo en esta misma ventana.
- El jackknife muestra cuánto cambia el resultado al quitar una posesión; si el rango es grande, el 0.64 no es estable.
- La decisión correcta para la submission final no debería basarse solo en esta tabla.

## Conclusión operativa

`model_consensus` es el mejor candidato en la ventana pública: `spellF1` 0.6400 frente a 0.5943 de `model_tcn_clean`, con una diferencia pareada media de +0.0451 (p05 +0.0000, p95 +0.1143) y 87.9% de resamples a favor.

Pero la ventana pública ya se usó para seleccionar y ajustar el modelo (ver auditoría de commits). El propio scoreboard avisa de que las diferencias pequeñas quedan dentro del ruido con 17 posesiones. El jackknife lo confirma: `model_consensus` baja a 0.6000 al quitar una sola posesión, mientras que `model_tcn_clean` llega a 0.6118 en su mejor caso.

**Recomendación:** no cambiar `los_chuchos.csv` a `model_consensus.csv` solo con esta evidencia. Mantener `model_tcn_clean.csv` como submission final y usar `model_consensus` como candidato experimental hasta tener una validación held-out o de partido completo. Si el equipo decide asumir el riesgo de sobreajuste para el scoreboard público, `model_consensus` es la mejor opción en esta ventana, pero debe etiquetarse como tal.
