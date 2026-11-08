# Fase 4: Diseño del Modelo Secuencial (HMM / Viterbi Jerárquico)

**Proyecto:** More than a Hack 2027 — Phase 1: Ball Possession  
**Script Principal:** [`models/hmm_viterbi.py`](file:///home/crwy/Projects/Barcelona/more-than-a-hack-2027/models/hmm_viterbi.py)  
**Submission Oficial:** [`evaluation/results/model_viterbi.csv`](file:///home/crwy/Projects/Barcelona/more-than-a-hack-2027/evaluation/results/model_viterbi.csv)  
**Posición en Scoreboard:** **#1 Absoluto** (`spellF1 = 0.5371`, `macroF1 = 0.7938`)

---

## 1. Fundamentos Teóricos y Arquitectura

La Fase 4 supera las limitaciones inherentes a los umbrales rígidos de `baseline_v2` mediante un **Modelo Oculto de Markov (HMM)** con decodificación global de **Viterbi**, estructurado de forma jerárquica siguiendo el marco conceptual de Borghesi et al. (2023) (*Expert Systems with Applications*).

### 1.1 Descomposición Jerárquica Desacoplada

En lugar de clasificar directamente entre $\{H, A, D\}$, el espacio de estados se desacopla en dos fases condicionales:

```text
               Vector de Características X_t (10 variables curadas)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
          Etapa 1: Rama DA                   Etapa 2: Rama POSS
       (Dead vs Alive Game)               (Home vs Away | Alive)
                  │                                   │
                  ├─ ball_out_of_pitch                ├─ delta_d_m (Burbuja 5-NN)
                  ├─ frames_to_next_setpiece          ├─ k5_home_ratio
                  ├─ in_open_play_phase               ├─ open_play_team
                  ├─ missing_streak_frames            └─ d_nearest_home / away
                  └─ min(d_home, d_away)
                  │                                   │
                  ▼                                   ▼
          P(Dead), P(Alive)                   P(H|Alive), P(A|Alive)
                  └─────────────────┬─────────────────┘
                                    ▼
                 Probabilidades de Emisión Conjunta:
                       P(D_t | X_t) = P(Dead)
                       P(H_t | X_t) = P(Alive) · P(H | Alive)
                       P(A_t | X_t) = P(Alive) · P(A | Alive)
                                    │
                                    ▼
                 Decodificador Global de Viterbi con A (3x3)
```

---

## 2. Formulación Matemática

### 2.1 Modelo de Emisión ($P(S_t \mid X_t)$)

#### Rama 1: Estado del Juego ($DA$)
El logit $z_{DA}$ modela la probabilidad de interrupción / balón muerto:
$$z_{DA} = -2.8 + 6.0 \cdot \mathbb{I}_{\text{out}} + 3.5 \cdot \max\left(0, \frac{180 - \Delta_{\text{SP}}}{100}\right) - 3.0 \cdot \mathbb{I}_{\text{open}} + 1.2 \cdot \min\left(\frac{L_{\text{missing}}}{50}, 5\right)(1 - \mathbb{I}_{\text{open}}) + 1.2 \cdot \max(0, \min(d_H, d_A) - 3.0)$$

$$P(\text{Dead}_t) = \begin{cases} 0.999 & \text{si } \mathbb{I}_{\text{out}} = 1 \\ \sigma(z_{DA}) & \text{en otro caso} \end{cases}$$
$$P(\text{Alive}_t) = 1 - P(\text{Dead}_t)$$

#### Rama 2: Posesión en Juego Vivo ($POSS \mid \text{Alive}$)
El logit $z_{POSS}$ modela la dominancia territorial y la cercanía al balón:
$$z_{POSS} = - 1.5 \cdot \text{clip}(\Delta d, -8, 8) + 2.0 \cdot (\rho_{\text{5NN}}^H - 0.5) + 0.8 \cdot \text{Prior}_{\text{smart}}$$

$$P(H_t \mid \text{Alive}) = \sigma(z_{POSS})$$
$$P(A_t \mid \text{Alive}) = 1 - P(H_t \mid \text{Alive})$$

#### Emisiones en Espacio Logarítmico con Temperatura
Con un hiperparámetro de temperatura $\alpha_{\text{emit}} = 0.80$:
$$\log \mathcal{E}(S_t) = \alpha_{\text{emit}} \cdot \log P(S_t \mid X_t)$$

---

### 2.2 Matriz de Transición Temporal de Markov ($A$)

El fútbol presenta una inercia temporal extrema a $25\text{ fps}$ ($40\text{ ms}$ por muestra). Los cambios de posesión ocurren con frecuencia $< 0.002$/frame:

$$A = \begin{pmatrix} 
0.9950 & 0.0004 & 0.0046 \\
0.0004 & 0.9950 & 0.0046 \\
0.0025 & 0.0025 & 0.9950 
\end{pmatrix}$$

* **Auto-transición ($A_{ii} = 0.9950$):** Fomenta posesiones continuas y estables (duración media esperada $\sim 14-16\text{ s}$).
* **Transición Cruzada Directa ($A_{HA} = A_{AH} = 0.0004$):** Impone una penalización severa a los cambios instantáneos $H \leftrightarrow A$ sin una disputa previa.
* **Transiciones a/desde Balón Parado ($A_{HD}, A_{AD} = 0.0046$; $A_{DH}, A_{DA} = 0.0025$):** Permite transiciones fluidas hacia saques o faltas.

---

### 2.3 Decodificación Global de Viterbi

Para una secuencia de $T$ frames (en cada mitad del partido), el algoritmo de Viterbi halla la secuencia de estados $S_{1:T}^* \in \{H, A, D\}^T$ que maximiza la verosimilitud conjunta:

$$\text{Paso hacia adelante: } V_t(j) = \max_{i \in \{H,A,D\}} \Big( V_{t-1}(i) + \log A_{i, j} \Big) + \log \mathcal{E}_t(j)$$
$$\text{Puntero de vuelta: } B_t(j) = \arg\max_{i \in \{H,A,D\}} \Big( V_{t-1}(i) + \log A_{i, j} \Big)$$
$$\text{Backtracking: } S_T^* = \arg\max_j V_T(j), \quad S_t^* = B_{t+1}(S_{t+1}^*)$$

---

## 3. Resultados Oficiales en el Scoreboard

Evaluación oficial ejecutada con `/home/crwy/.local/bin/run-eval` sobre el Ground Truth público:

| Posición | Modelo | `spellF1` (Rank) | `macroF1` | Posesiones | Duración Media | Error Cuota (`shareE`) | Latencia Media |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | **`model_viterbi` (Fase 4)** | **0.5371** | **0.7938** | **18** | **14.2 s** | **0.003** | **-3.8 frames** |
| #2 | `baseline_smooth125` | 0.4242 | 0.5812 | 16 | 18.5 s | 0.052 | -7.7 frames |
| #3 | `baseline_v2` (Fase 2) | 0.4235 | 0.7169 | 17 | 15.8 s | 0.020 | -6.4 frames |
| #4 | `baseline_smooth50` | 0.3256 | 0.6207 | 26 | 10.8 s | 0.048 | +4.0 frames |
| #5 | `baseline_smooth25` | 0.2868 | 0.6200 | 36 | 7.6 s | 0.056 | +8.2 frames |
| #6 | `baseline_raw` | 0.0378 | 0.5972 | 131 | 2.0 s | 0.046 | -3.1 frames |

### 3.1 Análisis de Mejoras Clave

1. **`spellF1` (+0.113, +27% respecto al líder anterior):**  
   El salto de 0.4242 a 0.5371 confirma que Viterbi preserva la integridad de los hechizos de posesión (*possession spells*) sin truncarlos prematuramente.
2. **Duración Media Exacta (14.2 s vs 14.2 s GT):**  
   El modelo predice una duración media que calca al décimo de segundo el promedio real del partido.
3. **Calibración de la Cuota de Posesión (`shareE = 0.003`):**  
   El reparto de posesión previsto (56.9% Barça vs 43.1% Athletic) difiere en apenas un 0.3% del Ground Truth real (57.3% vs 42.7%).
4. **Desempeño en Balón Muerto ($D$):**  
   El F1 de la clase $D$ alcanza **0.724** (con 932 frames de $D$ correctamente detectados de 1469), comparado con apenas 116 frames (recall 7.9%, F1 0.146) en el baseline anterior.
5. **Puntualidad en Transiciones ($\tau = 25$):**  
   La latencia media es de apenas **-3.8 frames (-0.15 s)**, demostrando que la regularización temporal de Markov no introdujo retraso perjudicial (*lag*).

---

## 4. Eficiencia Computacional y Escalabilidad

* **Tiempo de Extracción de Features (Fase 3):** 7.2 segundos para los 199.319 frames.
* **Tiempo de Decodificación Viterbi (Fase 4):** 1.4 segundos para los 199.319 frames.
* **Memoria RAM:** $< 150\text{ MB}$ (utilizando arreglos contiguos en NumPy float32/int8).
* **Dependencias:** 100% Python estándar y NumPy (sin PyTorch ni librerías pesadas externas).
