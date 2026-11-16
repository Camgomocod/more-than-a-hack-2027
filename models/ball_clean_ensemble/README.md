# `ball_clean_ensemble` — la cadena del envío

Esta carpeta construye el modelo que se envía como `los_chuchos.csv`: un TCN
causal que decodifica posesiones sobre el balón limpio y un ensemble consensus
que combina el TCN con el HMM bayesiano.

**Si solo quieres ver el rendimiento, no hace falta nada de esta carpeta.** Desde
la raíz del repo:

```powershell
pip install -r requirements.txt
python reproduce.py            # puntúa los_chuchos.csv con el scorer oficial
python reproduce.py --verify   # reconstruye el CSV desde los artefactos commiteados
```

Los comandos completos, con los SHA256 esperados, están en
`TECHNICAL_DOCUMENT.md` (sección 6). Esta guía es el detalle de implementación.

---

## 1. Qué produce esta carpeta

| artefacto | contenido | SHA256 |
|---|---|---|
| `features_clean_long_lastseen.npz` | 37 features sobre el balón limpio | `925ED02E…` |
| `results/clean_ensemble.csv` | pseudo-etiquetas H/A/D/X para el TCN | `C8B99D47…` |
| `tcn_clean*.pt` | 6 checkpoints (seed 7 + 5 semillas) | — |
| `tcn_clean_logits*.npz` | logits por seed + media | `E89EDFF0…` (media) |
| `evaluation/results/model_tcn_clean.csv` | TCN decodificado | `39711AC6…` |
| `los_chuchos.csv` (raíz) | consensus TCN + HMM | `852AA5FB…` |

`model_ensemble.csv` (el HMM) lo genera `models/hmm_ensemble.py`, no esta
carpeta: `8EB5FBA0…`.

---

## 2. Linaje de las features

```text
models/akima_with_data_prep/ball_cleaning.ipynb
        -> models/akima_with_data_prep/ball_clean.npz
        -> build_features_clean.py
        -> features_clean_long_lastseen.npz
```

`build_features_clean.py` carga `ball_clean.npz` y el ATD crudo (para las
features de jugadores) y recorre los frames una sola vez. Mientras el balón está
invisible usa `last_seen` como referencia de distancias. El sidecar
`features_clean_meta.json` deja registradas las rutas relativas y los conteos:

```text
observed = 113488, imputed = 185, missing_long = 85646, missing_raw = 85831
```

Requiere `models/akima_with_data_prep/ball_clean.npz` y los datos crudos del ATD
(ver `DATA.md` y `MTAH_DATA_DIR`).

```powershell
python models/ball_clean_ensemble/build_features_clean.py
```

---

## 3. Pseudo-etiquetas del TCN

```powershell
python models/ball_clean_ensemble/run_clean_pipelines.py
```

Decodifica el ensemble HMM sobre las features limpias y escribe
`results/clean_ensemble.csv`, las pseudo-etiquetas duras del TCN. No lee el
ground truth en ningún momento.

Ojo: este decodificador **no** incluye el `DEAD_BOOST` que `hmm_ensemble.py`
aplica a la submission del HMM. Es intencionado: el TCN se entrenó con estas
etiquetas y el SHA de arriba lo comprueba.

---

## 4. Entrenar el ensemble de 6 TCN

```powershell
python models/ball_clean_ensemble/train_tcn_clean.py --seed 7
python models/ball_clean_ensemble/train_tcn_clean.py --seed 1  --tag seed1
python models/ball_clean_ensemble/train_tcn_clean.py --seed 2  --tag seed2
python models/ball_clean_ensemble/train_tcn_clean.py --seed 11 --tag seed11
python models/ball_clean_ensemble/train_tcn_clean.py --seed 23 --tag seed23
python models/ball_clean_ensemble/train_tcn_clean.py --seed 42 --tag seed42
```

En CPU, unos 6 minutos por seed. Cada ejecución escribe `tcn_clean{tag}.pt` y
`tcn_clean_logits{tag}.npz`.

Configuración: 37 canales de entrada, `channels=48`, `levels=6`,
`dropout=0.10`, 12 epochs × 120 steps, `seq_len=2048`, `batch_size=8`,
`soft_weight=0.5`, `tv_weight=0.005`. Pérdida: cross-entropy ponderada por clase
+ soft targets de las emisiones HMM + TV loss.

---

## 5. Promediar logits y decodificar

```powershell
python models/ball_clean_ensemble/ensemble_tcn_logits.py `
  --inputs models/ball_clean_ensemble/tcn_clean_logits.npz `
           models/ball_clean_ensemble/tcn_clean_logits_seed1.npz `
           models/ball_clean_ensemble/tcn_clean_logits_seed2.npz `
           models/ball_clean_ensemble/tcn_clean_logits_seed11.npz `
           models/ball_clean_ensemble/tcn_clean_logits_seed23.npz `
           models/ball_clean_ensemble/tcn_clean_logits_seed42.npz `
  --out models/ball_clean_ensemble/tcn_clean_logits_ensemble.npz

python models/ball_clean_ensemble/run_tcn_variants.py
```

`run_tcn_variants.py` usa la configuración congelada del envío
(`tcn_weight=1.0`, `min_spell=62`, `scale=0.8`, sin refinado de fronteras) y
escribe directamente `evaluation/results/model_tcn_clean.csv`. El score que
imprime al final es un diagnóstico sobre la ventana pública, no un criterio de
selección: no hay barridos de hiperparámetros en este script.

---

## 6. Piezas

| archivo | función |
|---|---|
| `build_features_clean.py` | construye `features_clean_long_lastseen.npz` |
| `run_clean_pipelines.py` | decodifica el ensemble HMM y escribe las pseudo-etiquetas |
| `tcn_model.py` | define el TCN causal dilatado |
| `train_tcn_clean.py` | entrena una seed y guarda checkpoint + logits |
| `ensemble_tcn_logits.py` | promedia probabilidades de varias seeds |
| `run_tcn_variants.py` | decodifica los logits con Viterbi y escribe `model_tcn_clean.csv` |
| `validate_consensus.py` | auditoría in-sample: bootstrap pareado y jackknife |

---

## 7. Problemas frecuentes

**`ModuleNotFoundError: No module named 'torch'`**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**`No se encontró features_clean_long_lastseen.npz`** → ejecuta la sección 2.

**`No se encontró ball_clean.npz`** → debe existir
`models/akima_with_data_prep/ball_clean.npz`.

**Faltan los datos crudos del ATD** → configura `MTAH_DATA_DIR` (ver `DATA.md`).
Para solo evaluar resultados ya generados no hacen falta.

---

## 8. Reglas

- El GT público no se usa para entrenar ni para elegir hiperparámetros.
- El TCN usa pseudo-etiquetas del HMM propio, nunca labels humanos.
- El GT público solo se usa como diagnóstico, y esa ventana ya se usó para
  fijar las constantes del consensus: el riesgo se declara en
  `TECHNICAL_DOCUMENT.md` y en `evaluation/consensus_validation.md`.
