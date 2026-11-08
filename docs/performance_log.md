# Registro Histórico de Desempeño y Benchmark de Modelos

**Última actualización:** 2026-11-08 16:32:04  
**Reto:** More than a Hack 2027 — Phase 1: Ball Possession (Metrica Sports / FC Barcelona / MWCapital)  
**Archivo de datos estructurados:** [`evaluation/performance_log.json`](file:///home/crwy/Projects/Barcelona/more-than-a-hack-2027/evaluation/performance_log.json)  

---

## 1. Tabla Oficial de Clasificación (Ground Truth Público: FC Barcelona vs. Athletic Club)

| Rank | Submission | Fase / Enfoque | `spellF1` (Rank) | `macroF1` | Exactitud | $\kappa$ | Posesiones (GT=17) | Duración (GT=14.2s) | F1 D (Paradas) | `shareE` | `transF1` ($\tau=25$) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **`model_ensemble`** | Fase 5.3 (Ensemble Bayesiano Unificado) | **0.5444** | **0.8148** | 83.1% | 0.7279 | 19 | 13.4s | 0.7549 | 0.005 | 0.3636 |
| 2 | `model_viterbi_refined` | Fase 5.1 (Modelo Refinado) | 0.5371 | 0.8003 | 81.8% | 0.7067 | 18 | 14.1s | 0.7368 | 0.002 | 0.2857 |
| 3 | `model_viterbi` | Fase 4 (HMM Viterbi Base) | 0.5371 | 0.7938 | 81.3% | 0.6985 | 18 | 14.2s | 0.7244 | 0.003 | 0.2857 |
| 4 | `model_bundesliga_trained` | Fase 5.2 (Entrenado en Bundesliga) | 0.5000 | 0.7959 | 81.9% | 0.7071 | 19 | 13.7s | 0.7069 | 0.011 | 0.3256 |
| 5 | `baseline_smooth125` | Baseline Oficial | 0.4242 | 0.5812 | 72.7% | 0.5355 | 16 | 18.5s | 0.1464 | 0.052 | 0.1667 |
| 6 | `baseline_v2` | Fase 2 (Heurístico Contextual) | 0.4235 | 0.7169 | 76.6% | 0.6157 | 17 | 15.8s | 0.5474 | 0.020 | 0.2326 |
| 7 | `baseline_smooth50` | Baseline Oficial | 0.3256 | 0.6207 | 71.3% | 0.5202 | 26 | 10.8s | 0.3225 | 0.048 | 0.0816 |
| 8 | `baseline_smooth25` | Baseline Oficial | 0.2868 | 0.6200 | 69.1% | 0.4878 | 36 | 7.6s | 0.3857 | 0.056 | 0.2034 |
| 9 | `baseline_raw` | Baseline Oficial | 0.0378 | 0.5972 | 65.1% | 0.4282 | 131 | 2.0s | 0.4197 | 0.046 | 0.1410 |

---

## 2. Progresión Métrica a lo largo de las Fases del Proyecto

| Hito / Versión | Descripción Clave | `spellF1` | `macroF1` | Recall D | Posesiones | Estado |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Inicial (`baseline_raw`)** | Asignación pura al más cercano frame a frame | 0.0378 | 0.5972 | 33.6% | 131 (2.0s) | Superado |
| **Filtro Suavizado (`smooth125`)** | Ventana móvil ciega de 5 segundos | 0.4242 | 0.5812 | 7.9% | 16 (18.5s) | Superado |
| **Fase 2 (`baseline_v2`)** | Histeresis 2.5m + colchón 5-NN + Set Pieces | 0.4235 | 0.7169 | 41.9% | 17 (15.8s) | Superado |
| **Fase 4 (`model_viterbi`)** | HMM Jerárquico + Viterbi Global + 10 features Fase 3 | **0.5371** | **0.7938** | **63.4%** | **18 (14.2s)** | **#1 Scoreboard** |

---

## 3. Pruebas de Generalización Fuera de Dominio (Benchmark Externo)

Evaluación en el dataset público **IDSSE** (*Nature Scientific Data*, 2025) con tracking óptico Sportec Solutions TRACAB (25 fps) de la Bundesliga alemana:

| Partido Evaluado | Frames | Duración | `macroF1` | Exactitud | $\kappa$ | F1 Balón Parado ($D$) | Error Reparto Posesión |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. FC Köln vs. FC Bayern München (DFL-MAT-J03WMX)** | 145,967 | 97.3 min | **0.8227** | **83.1%** | **0.7420** | **0.8940** | **0.76%** |
| **1. FC Köln vs. FC Bayern München (DFL-MAT-J03WMX)** | 145,967 | 97.3 min | **0.8317** | **84.2%** | **0.7554** | **0.8899** | **1.10%** |
| **1. FC Köln vs. FC Bayern München (DFL-MAT-J03WMX)** | 145,967 | 97.3 min | **0.8319** | **84.3%** | **0.7570** | **0.8970** | **1.10%** |

### Conclusiones del Benchmark Externo
* El modelo funciona directamente sobre otro sistema de tracking y otra liga sin reentrenamiento.
* La detección de balón parado ($D$) alcanza un F1 de **0.8970**, confirmando la fiabilidad de la arquitectura jerárquica $DA$.
* La cuota de posesión acumulada tiene un error de apenas **1.1%** a lo largo de un partido completo de 97.3 minutos.

---

## 4. Instrucciones de Reproducción

```bash
# 1. Generar la submission oficial #1
.venv/bin/python models/hmm_viterbi.py

# 2. Evaluar en el scoreboard oficial
run-eval

# 3. Ejecutar el benchmark sobre IDSSE Bundesliga
.venv/bin/python evaluation/benchmark_idsse.py

# 4. Actualizar este registro de desempeño
.venv/bin/python evaluation/update_performance_log.py
```
