# Guía de ejecución y comparación — `ball_clean_ensemble`

Esta carpeta contiene el trabajo de la sesión para mejorar la detección de posesiones
usando el balón limpio (`ball_clean.npz`) y un TCN causal entrenado con pseudo-etiquetas.

El objetivo principal era **subir `spellF1`**, sin usar el GT público para entrenar.
El GT público se usa únicamente para evaluar y diagnosticar.

---

## 1. Resultado final

Candidato principal:

```text
evaluation/results/model_tcn_clean.csv
```

Es una copia de:

```text
models/ball_clean_ensemble/results/tcn_viterbi.csv
```

Resultados públicos actuales:

| submission | spellF1 | macroF1 | transF1 @25 | poss | dur | frag |
|---|---:|---:|---:|---:|---:|---:|
| **model_tcn_clean** | **0.5943** | **0.8312** | **0.4186** | 18 | 13.9 s | 1.14 |
| model_ensemble | 0.5444 | 0.8148 | 0.3636 | 19 | 13.4 s | 1.19 |
| clean_ensemble | 0.5444 | 0.8220 | 0.4091 | 19 | 13.2 s | 1.19 |

Mejora frente a `model_ensemble`:

- `spellF1`: **+0.0498**
- `macroF1`: **+0.0165**
- `transF1 @25`: **+0.0550**
- una posesión menos y menos fragmentación

Intervalos bootstrap de `spellF1`:

- `model_tcn_clean`: `0.4343 .. 0.7434`
- `model_ensemble`: `0.4000 .. 0.7000`

Los intervalos se solapan. La mejora es consistente, pero **no es estadísticamente
concluyente** con solo 17 posesiones públicas.

`model_tcn_clean` y `tcn_viterbi` son el mismo resultado; el primero es el nombre
usado en el scoreboard principal y el segundo el usado dentro de esta carpeta.

---

## 2. Requisitos

Se asume Windows PowerShell y el entorno virtual del repo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si la activación falla por política de ejecución, usar directamente:

```powershell
.venv\Scripts\python.exe --version
```

Dependencias usadas:

- Python 3.15
- `numpy`
- `pandas`
- `matplotlib`
- `jupyter`
- `torch` CPU

Si falta `torch`:

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Para reconstruir features desde el ATD crudo también hacen falta los datos del partido.
Ver `DATA.md` y la variable `MTAH_DATA_DIR`.

---

## 3. Ruta rápida: solo evaluar y comparar

Si ya existen los artefactos de esta carpeta, no hace falta reentrenar.

### 3.1 Evaluar el resultado principal

```powershell
.venv\Scripts\python.exe evaluation\run_evaluation.py --gt public --bootstrap 1000
```

Esto puntúa todos los CSV de `evaluation/results/`, escribe:

```text
evaluation/scoreboard.csv
evaluation/reports/
```

### 3.2 Evaluar las variantes limpias

```powershell
.venv\Scripts\python.exe evaluation\run_evaluation.py `
  --gt public `
  --results models\ball_clean_ensemble\results `
  --out models\ball_clean_ensemble\evaluation `
  --bootstrap 1000
```

Esto escribe:

```text
models/ball_clean_ensemble/evaluation/scoreboard.csv
models/ball_clean_ensemble/evaluation/reports/
```

### 3.3 Comparar variantes

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\compare_metrics.py
```

Muestra tabla ordenada por `spellF1` y deltas contra `baseline_ensemble`.

### 3.4 Abrir el notebook de revisión

```powershell
.venv\Scripts\jupyter-notebook.exe models\ball_clean_ensemble\review_results.ipynb
```

El notebook incluye scoreboards, deltas, `spellF1` por IoU, transiciones, errores por
posesión, timeline y comparación entre semillas del TCN.

### 3.5 Abrir el notebook de ablación

```powershell
.venv\Scripts\jupyter-notebook.exe models\ball_clean_ensemble\clean_vs_noclean.ipynb
```

Compara HMM y TCN con limpieza frente al control sin limpieza. Es una ablación
pequeña de una seed, con las limitaciones explicadas en el propio notebook.
Si faltan `features_noclean.npz` o los resultados de control, ejecutar los
comandos de la sección 6.

### 3.6 Analizar errores de una submission concreta

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\analyze_spell_errors.py `
  --pred models\ball_clean_ensemble\results\tcn_viterbi.csv `
  --gt evaluation\gt\BAR-ATH_possession_gt_public.csv
```

---

## 4. Ruta completa: reconstruir todo

### 4.1 Construir features limpias

Requiere `models/akima_with_data_prep/ball_clean.npz` y los datos crudos del ATD.

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\build_features_clean.py --all
```

Escribe:

```text
models/ball_clean_ensemble/features_clean_long_lastseen.npz
models/ball_clean_ensemble/features_clean_rawmissing.npz
models/ball_clean_ensemble/features_clean_centroidref.npz
models/ball_clean_ensemble/features_clean_meta.json
```

Las tres variantes son:

| modo | significado |
|---|---|
| `long_lastseen` | gaps largos + referencia `last_seen`; es la usada por el TCN |
| `rawmissing` | missing crudo del ATD + referencia `last_seen` |
| `centroidref` | gaps largos + referencia al centroide de los 5 jugadores cercanos |

Contadores vistos en la sesión:

```text
observed = 113488
imputed = 185
missing_long = 85646
missing_raw = 85831
```

### 4.2 Reproducir baseline y variantes HMM

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\run_clean_pipelines.py
```

Escribe en `models/ball_clean_ensemble/results/`:

```text
baseline_ensemble.csv
clean_viterbi.csv
clean_bundesliga.csv
clean_ensemble.csv
clean_ensemble_rawmissing.csv
clean_ensemble_centroidref.csv
```

El baseline debe coincidir con `evaluation/results/model_ensemble.csv`.
El script imprime `Reproducción del baseline: OK` cuando coincide.

### 4.3 Entrenar el TCN base

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\train_tcn_clean.py `
  --seed 7 `
  --skip-variants
```

Es el modelo base guardado como:

```text
models/ball_clean_ensemble/tcn_clean.pt
models/ball_clean_ensemble/tcn_clean_logits.npz
```

En CPU tarda aproximadamente 6 minutos por seed.

### 4.4 Entrenar las semillas del ensemble

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\train_tcn_clean.py --seed 1  --tag seed1  --skip-variants
.venv\Scripts\python.exe models\ball_clean_ensemble\train_tcn_clean.py --seed 2  --tag seed2  --skip-variants
.venv\Scripts\python.exe models\ball_clean_ensemble\train_tcn_clean.py --seed 11 --tag seed11 --skip-variants
.venv\Scripts\python.exe models\ball_clean_ensemble\train_tcn_clean.py --seed 23 --tag seed23 --skip-variants
.venv\Scripts\python.exe models\ball_clean_ensemble\train_tcn_clean.py --seed 42 --tag seed42 --skip-variants
```

Cada seed escribe:

```text
tcn_clean_seedX.pt
tcn_clean_logits_seedX.npz
```

El ensemble final usa 6 modelos: `seed7`, `seed1`, `seed2`, `seed11`, `seed23`, `seed42`.

### 4.5 Promediar los logits del ensemble

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\ensemble_tcn_logits.py `
  --inputs `
    models\ball_clean_ensemble\tcn_clean_logits.npz `
    models\ball_clean_ensemble\tcn_clean_logits_seed1.npz `
    models\ball_clean_ensemble\tcn_clean_logits_seed2.npz `
    models\ball_clean_ensemble\tcn_clean_logits_seed11.npz `
    models\ball_clean_ensemble\tcn_clean_logits_seed23.npz `
    models\ball_clean_ensemble\tcn_clean_logits_seed42.npz `
  --out models\ball_clean_ensemble\tcn_clean_logits_ensemble.npz
```

### 4.6 Generar las submissions TCN

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\run_tcn_variants.py
```

Escribe:

```text
results/tcn_viterbi.csv
results/tcn_hmm_10.csv
results/tcn_hmm_20.csv
results/tcn_hmm_30.csv
```

`tcn_viterbi` es el candidato final.

### 4.7 Promover el candidato al scoreboard principal

```powershell
Copy-Item `
  models\ball_clean_ensemble\results\tcn_viterbi.csv `
  evaluation\results\model_tcn_clean.csv `
  -Force
```

Después volver a ejecutar:

```powershell
.venv\Scripts\python.exe evaluation\run_evaluation.py --gt public --bootstrap 1000
```

---

## 5. Qué hace cada pieza

| archivo | función |
|---|---|
| `build_features_clean.py` | construye las tres variantes de features limpias |
| `build_features_noclean.py` | construye el control sin limpieza para la ablación |
| `run_clean_pipelines.py` | ejecuta baseline y variantes HMM sobre features limpias |
| `tcn_model.py` | define el TCN causal dilatado |
| `train_tcn_clean.py` | entrena, carga y genera variantes TCN |
| `ensemble_tcn_logits.py` | promedia probabilidades de varios TCN |
| `run_tcn_variants.py` | decodifica TCN/HMM con Viterbi |
| `compare_metrics.py` | compara scoreboards y calcula deltas |
| `analyze_spell_errors.py` | muestra errores por posesión |
| `tune_clean_hmm.py` | grid de diagnóstico para HMM |
| `review_results.ipynb` | notebook de revisión y comparación |
| `clean_vs_noclean.ipynb` | notebook pequeño de ablación con/sin limpieza |

### Modelo TCN

Configuración usada:

```text
features: 37 canales
channels: 48
levels: 6
dropout: 0.10
epochs: 12
steps por epoch: 120
seq_len: 2048
batch_size: 8
soft_weight: 0.5
tv_weight: 0.005
```

Entrenamiento:

- etiquetas duras: `clean_ensemble.csv`
- objetivos suaves: mezcla de emisiones HMM
- peso extra en ventanas de transición
- pérdida: cross-entropy + soft loss + TV loss

Decodificación final:

```text
tcn_weight = 1.0
min_spell = 62
scale = 0.8
boundary refinement = desactivado
```

### Por qué se eligió Viterbi

El TCN solo, con `argmax` y `debounce`, no supera al HMM. La mejora aparece al usar
los logits del TCN como emisiones dentro de Viterbi con `min_spell=62`.

### Por qué se eligió el ensemble

Las semillas individuales dieron `spellF1` entre `0.5600` y `0.6118`.
El ensemble de 6 da `0.5943`, más estable y menos dependiente de una seed concreta.

---

## 6. Limpieza vs sin limpieza

### Linaje de las features

```text
models/akima_with_data_prep/ball_cleaning.ipynb
        -> models/akima_with_data_prep/ball_clean.npz
        -> models/ball_clean_ensemble/build_features_clean.py
        -> models/ball_clean_ensemble/features_clean_long_lastseen.npz
```

`features_clean_long_lastseen.npz` sale de la limpieza propuesta en
`ball_cleaning.ipynb`: `build_features_clean.py` carga `ball_clean.npz` y el ATD
crudo (para las features de jugadores). El meta lo deja registrado:

```text
features_clean_meta.json -> ball_clean_path / raw_atd_path / counts
```

### Control sin limpieza

`build_features_noclean.py` construye `features_noclean.npz` desde las features
antiguas (`features/features.npz`) y el balón crudo. El control:

- no aplica Akima,
- no usa `ball_gap_*`,
- no usa `last_seen` (rellena con el centro),
- usa `missing` crudo,
- conserva el mismo esquema de 37 canales del TCN.

El HMM de control reproduce exactamente `baseline_ensemble`, así que la
comparación es válida.

### Resultados de la ablación

| pipeline | spellF1 | macroF1 | transF1@25 | poss |
|---|---:|---:|---:|---:|
| HMM baseline / control sin limpieza | 0.5444 | 0.8148 | 0.3636 | 19 |
| HMM con limpieza | 0.5444 | 0.8220 | 0.4091 | 19 |
| TCN seed7 sin limpieza | 0.4973 | 0.8012 | 0.3556 | 20 |
| TCN seed7 con limpieza | 0.5789 | 0.8222 | 0.4783 | 21 |
| TCN ensemble6 con limpieza | 0.5943 | 0.8312 | 0.4186 | 18 |

Diferencia TCN seed7 `clean - noclean`:

- `spellF1`: **+0.0817**
- `macroF1`: **+0.0210**
- `transF1@25`: **+0.1227**

Lectura: la limpieza no cambia el `spellF1` del HMM, pero sí parece aportar al
TCN. Es una sola seed y los intervalos bootstrap se solapan, así que la
conclusión es direccional, no estadísticamente concluyente.

Notebook: `clean_vs_noclean.ipynb`.

Comandos:

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\build_features_noclean.py

.venv\Scripts\python.exe models\ball_clean_ensemble\run_clean_pipelines.py `
  --skip-baseline `
  --clean-features models\ball_clean_ensemble\features_noclean.npz `
  --rawmissing-features models\ball_clean_ensemble\features_noclean.npz `
  --centroidref-features models\ball_clean_ensemble\features_noclean.npz `
  --results-dir models\ball_clean_ensemble\results_noclean

.venv\Scripts\python.exe models\ball_clean_ensemble\train_tcn_clean.py `
  --seed 7 --tag noclean `
  --features models\ball_clean_ensemble\features_noclean.npz `
  --labels models\ball_clean_ensemble\results_noclean\clean_ensemble.csv `
  --skip-variants

.venv\Scripts\python.exe models\ball_clean_ensemble\run_tcn_variants.py `
  --logits models\ball_clean_ensemble\tcn_clean_logits_noclean.npz `
  --features models\ball_clean_ensemble\features_noclean.npz `
  --prefix noclean_

.venv\Scripts\python.exe models\ball_clean_ensemble\run_tcn_variants.py `
  --logits models\ball_clean_ensemble\tcn_clean_logits.npz `
  --features models\ball_clean_ensemble\features_clean_long_lastseen.npz `
  --prefix clean_seed7_
```

---

## 7. Comparaciones disponibles

### `tcn_viterbi` vs `clean_ensemble`

| métrica | tcn_viterbi | clean_ensemble |
|---|---:|---:|
| spellF1 | 0.5943 | 0.5444 |
| macroF1 | 0.8312 | 0.8220 |
| transF1 @25 | 0.4186 | 0.4091 |
| poss | 18 | 19 |
| dur | 13.9 s | 13.2 s |
| frag | 1.14 | 1.19 |

### Interpretación

- `spellF1` manda en el scoreboard.
- `macroF1` es secundaria; puede favorecer modelos que no mejoran la estructura.
- `poss`, `dur` y `frag` explican el fallo mejor que un F1 aislado.
- El bootstrap evita sobreinterpretar diferencias pequeñas.

### Errores restantes

Los errores principales están en:

- posesiones muy cortas, de 1 a 4 segundos
- transiciones alrededor de balón muerto
- spells que el TCN fusiona o parte en dos

La siguiente vía razonable es un decoder con modelo de duración explícito
(HSMM o Viterbi semi-Markov), no subir `min_spell` sin control.

---

## 8. Diagnóstico HMM

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\tune_clean_hmm.py
```

Es una herramienta de diagnóstico. Usa el GT público para puntuar, no para entrenar.
En la sesión no encontró una mejora clara sobre la configuración por defecto.

---

## 9. Problemas frecuentes

### `ModuleNotFoundError: No module named 'torch'`

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### `No se encontró features_clean_*.npz`

Ejecutar:

```powershell
.venv\Scripts\python.exe models\ball_clean_ensemble\build_features_clean.py --all
```

### `No se encontró ball_clean.npz`

Verificar que exista:

```text
models/akima_with_data_prep/ball_clean.npz
```

### Faltan los datos crudos del ATD

Configurar `MTAH_DATA_DIR` o colocar los datos en `./data`. Ver `DATA.md`.
Si solo se quieren evaluar resultados ya generados, no hacen falta.

### El scoreboard principal no muestra `model_tcn_clean`

Copiar de nuevo:

```powershell
Copy-Item models\ball_clean_ensemble\results\tcn_viterbi.csv evaluation\results\model_tcn_clean.csv -Force
```

### El notebook no encuentra la raíz del repo

Abrirlo desde la raíz del repo o desde esta carpeta. El notebook sube directorios
hasta encontrar `evaluation/possession_metrics.py`.

---

## 10. Reglas y notas de la sesión

- No se usó el GT público para entrenar.
- El TCN usa pseudo-etiquetas generadas por `clean_ensemble`.
- El GT público solo se usa para evaluación y diagnóstico.
- No se modificaron archivos del compañero.
- Los archivos nuevos están dentro de `models/ball_clean_ensemble/`.
- `evaluation/results/model_tcn_clean.csv` es una submission local; no hace falta commitearla.
- No hay `data/idsse`, así que no se hizo entrenamiento supervisado adicional con Bundesliga.
- Se mantuvo la semántica antigua de `ball_speed_mps` para que la comparación con el
  baseline no se contamine por el cambio de features.

---

## 11. Orden recomendado

Si vienes de cero:

```text
1. build_features_clean.py --all
2. run_clean_pipelines.py
3. train_tcn_clean.py (seed 7)
4. train_tcn_clean.py (seeds 1, 2, 11, 23, 42)
5. ensemble_tcn_logits.py
6. run_tcn_variants.py
7. copiar tcn_viterbi.csv a evaluation/results/model_tcn_clean.csv
8. run_evaluation.py --gt public --bootstrap 1000
9. compare_metrics.py
10. abrir review_results.ipynb
```

Si solo quieres revisar lo ya hecho:

```text
1. run_evaluation.py --gt public --bootstrap 1000
2. compare_metrics.py
3. review_results.ipynb
```
