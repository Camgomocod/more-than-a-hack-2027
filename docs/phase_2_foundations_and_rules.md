# Fundamentos, Reglas y Justificación Técnica: Fase 2 (Baseline v2)

Este documento recopila las bases factuales extraídas de los datos del partido, los principios teóricos del paper de referencia (*Borghesi et al., 2023*) y la justificación técnica de las decisiones adoptadas para la **Fase 2: Baseline Heurístico Mejorado & Fusión Contextual**.

---

## 1. Bases Factuales Extraídas de los Datos del Reto (Fase 1 EDA)

Nuestros análisis empíricos sobre los 199.319 frames del partido y el ground truth público de 5 minutos establecieron 4 verdades matemáticas inapelables:

1. **La paradoja del balón perdido:**
   * En todo el partido, el balón está ausente (`NaN`) en el **43.06%** de los frames.
   * En los 5 minutos del Ground Truth calificado, cuando el tracking pierde el balón (885 frames), el **44.2% del tiempo el juego sigue en vivo** (386 frames en posesión del Athletic Club y 5 del Barcelona).
   * *Base para el diseño:* Asignar `D` automáticamente cuando falta el balón (como hace el baseline oficial) es un error sistemático que fractura las posesiones legítimas.
2. **Dinámica de paradas reales (`D`):**
   * En el GT público solo hay **4 paradas de balón muerto**, con una duración media de **14.69 segundos** (entre 5.5 s y 20.4 s). No existen detenciones de 0.5 o 1 segundo en el fútbol real.
   * El baseline oficial apenas tiene un **Recall del 7.9% en `D`** porque solo detecta la parada si el tracking pierde de vista el balón. Si el balón se detiene visible en el suelo o en las manos de un jugador antes de un saque, el baseline sigue asignando posesión a `H` o `A`.
3. **El anclaje determinista de los `SET PIECES`:**
   * En el archivo `BAR-ATH_tactical_smart_tagging.xml` existen **59 jugadas a balón parado** (26 saques de banda, 12 córners, 10 saques de puerta, 5 faltas y 6 saques de centro).
   * Cada Set Piece define el momento exacto de la reanudación del juego, y la etiqueta `Team` predice con fiabilidad casi del 100% qué equipo saca.
   * La ventana previa al saque (entre 125 y 450 frames antes del reinicio) coincide exactamente con los tramos donde el juego estuvo detenido.
4. **Burbuja de seguridad espacial (Margen 5-NN):**
   * En posesión activa del Barcelona, el jugador de `H` más cercano al balón está en promedio a **2.98 m**, mientras que el defensor más próximo de `A` está a **7.29 m** (un colchón de ventaja de **+4.31 m**).
   * En posesión del Athletic, su atacante está a **3.15 m** y el defensor del Barça a **5.48 m** (ventaja de **+2.33 m**).
   * *Base para el diseño:* Un cambio de posesión nunca ocurre por un acercamiento circunstancial a 4 o 5 metros.

---

## 2. Principios del Paper de Referencia (Borghesi et al., 2023)

El paper *"Using Temporal Convolutional Networks to estimate ball possession in soccer games"* aporta los siguientes fundamentos metodológicos:

1. **Descomposición Jerárquica ($DA$ vs. $POSS$):**
   * El paper demuestra que abordar el problema como dos sub-tareas desacopladas es superior a una clasificación plana:
     * Tarea 1 ($DA$): Decidir si el juego está activo o detenido.
     * Tarea 2 ($POSS$): Condicionado a que el juego esté activo, decidir si pertenece a Home o Away.
   * *Aplicación en Fase 2:* Resolver primero la máscara de balón muerto (`D`) usando Smart Tagging y cinemática, y sobre los tramos activos resolver la posesión del equipo.
2. **Relevancia del entorno local (5-NN):**
   * El estudio de ablación del paper (Sección 5.3.1, Tabla 4) probó que considerar únicamente los **5 jugadores más cercanos al balón (5-NN)** retiene prácticamente el 100% de la precisión de posesión ($84.98\%$, idéntico a usar los 22 jugadores). Los 17 jugadores lejanos son puro ruido para determinar la posesión instantánea.
3. **Interpolación y Manejo de Datos Faltantes:**
   * El paper emplea **Akima spline** para reconstruir trayectorias de balón y banderas de dato faltante (`missing_flag`).
   * Para gaps cortos ($< 0.5\text{ s}$), la interpolación es óptima; para gaps medios y largos, la persistencia temporal de estado previene el parpadeo.

---

## 3. Justificación de los Componentes Seleccionados para la Fase 2

La **Fase 2** no salta a una red neuronal de caja negra todavía, sino que construye un **Baseline v2 determinista y fundamentado**. Cada componente ataca una debilidad matemática específica:

| Componente en Baseline v2 | Debilidad que ataca | Mecanismo Físico / Algorítmico |
|---|---|---|
| **1. Persistencia de Posesión (*Forward-fill* condicional)** | Los 386 frames vivos perdidos por ausencia de balón en el baseline. | Si el balón desaparece durante $\le 50$ frames (2.0 s) en juego abierto, la posesión se mantiene en el equipo que la tenía en lugar de colapsar a `D`. |
| **2. Inyección Retrógrada de `D` por Set Pieces** | El recall del 7.9% en la clase `D`. | Desde el frame de saque de cada uno de los 59 Set Pieces, retroceder una ventana (típicamente 5 a 15 segundos donde la velocidad del balón cae o sale del campo) y forzar la etiqueta `D`. Al instante del saque, asignar la posesión al equipo del tag `Team`. |
| **3. Filtro de Histéresis Espacial 5-NN** | La alternancia rápida e hiperfragmentación de `baseline_raw` (131 posesiones en 5 min). | Exigir que para que un rival "robe" la posesión deba colocarse a $< 2.5\text{ m}$ del balón de forma sostenida durante al menos 10–12 frames (0.4 s), evitando transiciones espurias por cruces o despejes. |
| **4. Post-procesado de Debounce / Posesión Mínima** | Posesiones artificiales de 0.5 segundos que destruyen el IoU temporal de `spellF1`. | Absorber cualquier posesión aislada de duración $< 1.0\text{ s}$ (< 25 frames) en el estado dominante adyacente, ya que en el GT el 90% de las posesiones superan los 3.5 segundos. |

---

## 4. Matriz de Ventajas y Desventajas del Enfoque de la Fase 2

### Ventajas:
1. **Salto masivo e inmediato en `spellF1`:**  
   Al resolver la clase `D` y evitar la fragmentación en huecos sin balón, la métrica de ranking subirá sustancialmente frente al 0.4242 del baseline oficial sin necesidad de entrenamientos pesados.
2. **Interpretabilidad total (Caja Blanca):**  
   Cada decisión está vinculada a una regla física y futbolística verificable. Esto es vital para el **documento técnico de entrega** que exige la competición.
3. **Cero riesgo de sobreajuste complejo:**  
   Al ser un modelo basado en principios cinemáticos y eventos de campo reales, generaliza a todo el partido mucho mejor que una red neuronal sobreajustada a los 5 minutos de muestra.
4. **Ejecución ultrarrápida:**  
   Se ejecuta en menos de 1 minuto sobre los 199.319 frames completos con numpy.

### Desventajas / Limitaciones:
1. **Dependencia de la calidad del Smart Tagging:**  
   Si el Smart Tagging omitió una falta menor que no derivó en tarjeta o saque claro, la ventana previa podría no marcarse como `D` con esta regla.
2. **Ambigüedad en balones divididos largos:**  
   En balones aéreos disputados entre dos jugadores donde ninguno tiene control claro durante varios segundos, la regla de histéresis mantendrá la posesión del último poseedor, lo cual es correcto según las bases del reto pero puede tener un pequeño desfase de frames en la frontera de cambio.
3. **Puntos de mejora para las Fases 3 y 4:**  
   Una máquina de estados probabilística (HMM con Viterbi) o un clasificador por ventanas (LightGBM) calibrará los umbrales de distancia y probabilidad de forma óptima y matemática en las fases posteriores.
