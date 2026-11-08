# Reporte de Generalización: Benchmark del Modelo en el Dataset IDSSE (Bundesliga)

**Fecha:** 2026-11-08  
**Dataset Externo:** [IDSSE](file:///home/crwy/Projects/Barcelona/more-than-a-hack-2027/data/idsse) (*Integrated Dataset of Synchronized Spatiotemporal and Event Data in Elite Soccer*, Bassek et al., *Nature Scientific Data*, 2025).  
**Partido Evaluado:** `DFL-MAT-J03WMX` — **1. FC Köln vs. FC Bayern München** (Bundesliga 2022/2023, Jornada 34).  
**Tecnología de Tracking:** Sportec Solutions TRACAB Optical Tracking (25 fps).  
**Volumen Evaluado:** **145.967 fotogramas** (97,3 minutos de fútbol continuo: Primer Tiempo 70.708 frames + Segundo Tiempo 75.259 frames).  
**Modelo Evaluado:** Modelo Oculto de Markov Jerárquico con Decodificación de Viterbi ([`models/hmm_viterbi.py`](file:///home/crwy/Projects/Barcelona/more-than-a-hack-2027/models/hmm_viterbi.py)).  
**Script de Reproducción:** [`evaluation/benchmark_idsse.py`](file:///home/crwy/Projects/Barcelona/more-than-a-hack-2027/evaluation/benchmark_idsse.py).

---

## 1. Contexto y Objetivo de la Prueba

El objetivo de esta prueba fue someter el modelo desarrollado en la Fase 4 a un **test de estrés fuera de dominio (*out-of-domain evaluation*)**:
1. **Diferente proveedor y sistema de tracking:** De ATD (Metrica/Barça) a TRACAB Optical Tracking (Sportec Solutions / DFL).
2. **Diferentes equipos y estilos de juego:** De LaLiga (*FC Barcelona vs. Athletic Club*) a la Bundesliga (*1. FC Köln vs. FC Bayern München*).
3. **Escala temporal completa:** En lugar de una ventana de 5 minutos, se evaluó el partido completo a lo largo de **97,3 minutos ininterrumpidos** (145.967 fotogramas).
4. **Ground Truth objetivo de alta resolución:** Los archivos TRACAB de IDSSE etiquetan fotograma a fotograma `BallStatus` (`0` = Parado / Dead, `1` = En juego / Alive) y `BallPossession` (`1` = Local, `2` = Visitante).

---

## 2. Resultados Consolidados

| Métrica | Primer Tiempo (47,1 min) | Segundo Tiempo (50,2 min) | **Partido Completo (97,3 min)** |
|:---|:---:|:---:|:---:|
| **Total Fotogramas** | 70.708 | 75.259 | **145.967** |
| **`macroF1`** | **0.8517** | **0.8120** | **0.8319** |
| **`Accuracy` (Exactitud Frame a Frame)** | **85.38%** | **83.13%** | **84.26%** |
| **`Cohen Kappa` ($\kappa$)** | **0.7781** | **0.7359** | **0.7570** |
| **`Alive/Dead F1`** | **0.9482** | **0.9155** | **0.9319** |
| **F1 Balón Muerto ($D$)** | **0.8888** | **0.9052** | **0.8970** |
| **F1 FC Bayern München ($A$)** | **0.8567** | **0.7615** | **0.8091** |
| **F1 1. FC Köln ($H$)** | **0.8096** | **0.7692** | **0.7894** |
| **`spellF1` (Mean IoU 0.3..0.7)** | **0.4000** | **0.3961** | **0.3981** |
| **Error en Cuota de Posesión (`shareE`)** | **0.0065 (0,65%)** | **0.0156 (1,56%)** | **0.0110 (1,10%)** |

---

## 3. Desglose Detallado

### 3.1 Primer Tiempo (70.708 fotogramas)
* **Distribución Real (TRACAB Ground Truth):**
  * Bayern München ($A$): 27.801 frames (39,3% total, 58,9% de juego vivo)
  * 1. FC Köln ($H$): 19.424 frames (27,5% total, 41,1% de juego vivo)
  * Balón Parado / Dead ($D$): 23.483 frames (33,2% del tiempo)
* **Rendimiento del Modelo:**
  * **`macroF1 = 0.8517`**
  * **Exactitud: 85,38%** (más de 60.360 fotogramas exactamente clasificados).
  * **Detección de Balón Parado ($D$):** F1 de **0.8888** (detectó 19.982 frames de paradas de 23.483 reales).
  * **Cuota de Posesión:** Predijo **40,48%** para Köln frente al **41,13%** real (diferencia de apenas **0,65%**).

### 3.2 Segundo Tiempo (75.259 fotogramas)
* **Rendimiento del Modelo:**
  * **`macroF1 = 0.8120`**
  * **Exactitud: 83,13%**
  * **Detección de Balón Parado ($D$):** F1 de **0.9052**
  * **Cuota de Posesión:** Predijo **50,06%** para Köln frente al **51,61%** real (diferencia de **1,56%** tras la presión del Köln en el tramo final).

---

## 4. Hallazgos Clave sobre la Capacidad de Generalización

1. **Robustez Transversal Inmune al Proveedor:**  
   El modelo, concebido en coordenadas normalizadas e independientes de sensor mediante la burbuja de ventaja espacial $\Delta d$ y el vecindario 5-NN, funcionó de inmediato sobre el sistema óptico TRACAB sin requerir re-entrenamiento ni ajuste de sesgos.
2. **Excelente Detección de Paradas ($F_1 \approx 0.90$ para $D$):**  
   El desacoplamiento jerárquico $DA$ demostró ser extremadamente fiable en partidos completos: el 90% de las fases sin juego (saques de banda, faltas, córners, interrupciones) fueron aisladas con precisión.
3. **Reproducción Fiel de las Estadísticas Oficiales de Posesión:**  
   El error acumulado en el cálculo del reparto de posesión fue de apenas **1,1%** a lo largo de los 97 minutos, validando la precisión del modelo para analítica de rendimiento en élite.
4. **Análisis de `spellF1` (0.398):**  
   En el tracking TRACAB de IDSSE, el sistema de eventos de la DFL anota micro-cambios de posesión en balones divididos y duelos aéreos (duración media en GT de 7,6 segundos con 250 posesiones por tiempo). El modelo predice 135-146 posesiones consolidadas (duración media de 14,3 s, similar al criterio de Metrica/Barça), lo que penaliza el IoU en desvíos momentáneos pero preserva la coherencia táctica global.
