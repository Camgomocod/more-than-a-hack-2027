# Ball possession detection — Los Chuchos

**Submission:** `los_chuchos.csv` (199,319 rows, `frame,label`)
**Public score:** spellF1 **0.6400**, macroF1 **0.8667**, accuracy 0.8715, alive/dead F1 0.9665

> Upload this document under the team name (`los_chuchos.md` or `los_chuchos.pdf`).

---

## 1. Approach

We built two independent possession models on the tactical ATD and combined them.

**A causal TCN over ball and player geometry.** Thirty-seven features per frame
(ball kinematics, gap length/age, out-of-pitch flag, nearest-player distances for
both teams, 5-nearest mean distances and centroid, set-piece phase and team).
A dilated causal TCN turns them into frame-level H/A/D probabilities. Six seeds
are trained and their probabilities averaged.

**A Bayesian HMM with learned emissions.** The same features feed a Viterbi
decoder over {home, away, dead} with a sticky transition prior
(p_stay = 0.995, i.e. ~8 s expected dwell) and a weak Bundesliga-trained side
signal. A dead-ball "kinetic boost" raises the dead emission when the ball is
slow and play is not in open play.

**The consensus ensemble.** The TCN path is the spine. The HMM is allowed to
veto: wherever it decodes dead and the TCN says home or away, the frame becomes
dead. The result is then re-snapped twice: a boundary refinement that moves
each change to the first frame where the receiving player is genuinely within
2.5 m, and a de-bounce of spells shorter than 25 frames.

Why this and not something else: the ranking metric is possession-spell F1, an
interval measure, not frame accuracy. The HMM contributes what a hand-tuned
emission cannot — a duration prior and a hard dead/alive decision — while the
TCN contributes what the emission models miss. Both are cheap, deterministic and
reproducible from committed artefacts.

The public window score with this ensemble is 0.6400 spellF1. Section 5 states
plainly why that number is optimistic.

---

## 2. Assets used, and assets dropped

**Used**

| asset | how |
|---|---|
| Tactical ATD (`BAR-ATH_tactical_atd.txt/.xml`) | 199,319 frames at 25 fps, pitch 105.3 × 68 m; player slots with team mapping, per-frame ball position and speed |
| Smart Tagging (in the ATD XML) | set-piece / open-play flags, time to the next and since the last set piece, set-piece team |
| Ball cleaning (Akima, pure NumPy) | gaps ≤ 25 frames are micro-interpolated; longer gaps stay `NaN` and expose `last_seen`, gap age/length/remaining and the 5-nearest-player reference |
| IDSSE Bundesliga, match `DFL-MAT-J03WMX` | two logistic emission models (dead/alive, possession side) in `models/bundesliga_weights.json` |

**Dropped**

| asset | why |
|---|---|
| Broadcast video | not frame-aligned to the tactical timeline, and the scorer is frame-exact; a one-frame error moves every transition it scores |
| Homography from video | no reliable calibration; the ATD already provides metric coordinates |
| Video-derived pose / appearance | cost and complexity with no verified gain over the ATD alone |
| Private full-match ground truth | not available, and not permitted |

---

## 3. Training data and provenance

| dataset | source and licence | who labelled it | size | how we used it |
|---|---|---|---|---|
| IDSSE Bundesliga `J03WMX` | `pysport/idsse-data`, CC BY 4.0 | the IDSSE dataset authors (human event annotations) | 145,967 frames | trained the two logistic emission models |
| BAR-ATH tactical ATD | provided bundle (not redistributed) | **nobody** — labels are our own HMM's pseudo-labels | 199,319 frames | TCN hard targets and soft targets |
| IDSSE `J03WN1`, `J03WPY` | same dataset, CC BY 4.0 | the IDSSE dataset authors | 2 matches, 287,772 graded frames | held-out evaluation only |

No human annotation of the provided match was used, and the private ground truth
was never touched. The TCN never sees the public ground truth: its targets are
the HMM's output on the same match (`clean_ensemble.csv`), an allowed
algorithmic label. The public 5-minute window was used to *diagnose* the model;
it was also, historically, used to fix the consensus constants — see
Limitation 1, which quantifies exactly what that cost.

---

## 4. Possession rules implemented

Labels: `H` home in possession, `A` away, `D` dead ball (stoppage), `X`
abstention.

**Where possession starts and ends.** A frame is alive when the ball is in play
and no stoppage is active. The side is decided by combining the physical
emission (`delta_d_m`, 5-nearest home ratio) with the Bundesliga logistic side
signal at α = 0.35 physical / 0.65 learned.

**Loose balls and challenges.** The possession side is the team whose nearest
player is closer to the ball; a change of side is only accepted when the new
team's nearest player is within 2.5 m and closer than the rival. Boundary
refinement searches ±15–20 frames around the decoded change for that first
qualifying frame.

**Stoppages and restarts.** Outside open play the dead emission is boosted
(`DEAD_BOOST = 2.5` when ball speed < 4 m/s), and the two-pass consensus lets
that veto override the TCN. A dead→alive transition must find the receiving
player within 2.5 m; a live→dead transition is moved back to the first frame
where the ball leaves the pitch, or stops (< 0.6 m/s) with no player within 2 m.

**Spells.** Viterbi with p_stay 0.995 (≈ 200 frames ≈ 8 s expected dwell).
Minimum spell 62 frames (2.5 s) for the TCN decode, then a 25-frame de-bounce in
the consensus. Halftime (frames 110,808–113,742) is `D`; frames before kickoff,
after the final whistle and outside the tracked halves are `X`.

| constant | value | where |
|---|---|---|
| emission blend α | 0.35 | `models/hmm_ensemble.py` |
| dead-ball boost / speed threshold | 2.5 / 4.0 m/s | `models/hmm_ensemble.py` |
| Viterbi p_stay | 0.995 | `models/hmm_ensemble.py` |
| TCN decode scale / min spell | 0.8 / 62 frames | `run_tcn_variants.py` |
| consensus de-bounce / snap window | 25 / 20 frames | `models/consensus_ensemble.py` |
| Akima micro-interpolation limit | 25 frames | `models/akima_with_data_prep/ball_cleaning.ipynb` |

---

## 5. Results and limitations

**Public 5-minute window** (17 possessions, 7,500 graded frames, 1,000 bootstrap draws):

| metric | value | 5–95% |
|---|---|---|
| spellF1 | **0.6400** | 0.4914 – 0.7886 |
| macroF1 | **0.8667** | 0.7718 – 0.9103 |
| per class F1 (H / A / D) | 0.9044 / 0.8347 / 0.8612 | |
| transition F1 @12 / @25 | 0.2791 / 0.4186 | |
| alive/dead F1 | 0.9665 | |
| possessions | 18 predicted vs 17 true (13.4 s vs 14.2 s mean) | |

**Held-out IDSSE benchmark** (two unseen matches, pooled, 921 possessions). Panel A
is the fair IDSSE-only comparison; Panel B transfers the BAR-ATH models with
set-piece features neutralised, so read it as indicative.

| panel | model | spellF1 | 5–95% | macroF1 | predicted spells |
|---|---|---|---:|---:|---:|
| A | HMM, physical emissions | 0.3836 | 0.3555 – 0.4109 | 0.7567 | 617 |
| A | HMM ensemble | 0.3642 | 0.3384 – 0.3871 | 0.7572 | 619 |
| A | nearest-player baseline | 0.1244 | 0.1168 – 0.1307 | 0.6727 | 4,074 |
| B | TCN (submission input) | **0.3415** | 0.3140 – 0.3682 | 0.6966 | 564 |
| B | consensus (submitted) | 0.3252 | 0.2991 – 0.3490 | 0.7302 | 577 |

Paired bootstrap, consensus − TCN: **−0.0163**, 95% CI [−0.0271, −0.0043];
consensus wins 0.6% of resamples.

**Limitations**

1. **The 0.6400 is in-sample.** The consensus constants (α, dead-ball boost,
   25-frame de-bounce, 20-frame snap window) were fixed while looking at this
   same 5-minute window. The honest out-of-sample estimate is the held-out row
   above: 0.3252 for the submitted consensus, 0.3415 for the TCN. The shipped
   file is still the consensus because the ranking is the public scoreboard.
2. **Pseudo-label dependence.** With no human labels for this match, the TCN
   distils our HMM's decisions. Systematic HMM errors are inherited, not fixed.
3. **The ball is invisible in 43.1% of frames** (11.8% inside the graded
   window). Most of the signal is player geometry around the last seen ball, so
   long occlusions degrade the side decision.
4. **Under-segmentation on unseen matches.** We predict 577 possessions where
   the truth has 921, and 564 for the TCN: spells are merged, so spellF1 is
   bounded by recall. On the public window the count is right (18 vs 17), which
   is part of why the two numbers differ so much.
5. **Dead-ball bias.** The consensus converts 13,379 frames the TCN called home
   or away into dead. Inside the graded window that costs 192 alive frames of
   6,031 (3.2%); outside it the public ground truth cannot audit the effect.
6. **Small sample.** 17 possessions give a ±0.30 bootstrap band. Differences
   below that are not evidence, and every constant here sits inside one.

**With more time:** a duration-aware decoder (semi-Markov / HSMM) to attack
limitation 4 directly, a held-out-selected consensus rather than a public-window
one, and a second IDSSE training match to stabilise the emission weights.

---

## 6. Reproduction

Requirements: Python 3.8+, `pip install -r requirements.txt` (numpy, pandas).
Torch is only needed to retrain the TCN; the data bundle is only needed to
rebuild features. Expected SHA-256 of the submission:
`852AA5FBD8AAFB0881FE0EBA41655E301873453EF39F0A08A96F44655306DA53`.

**See the score and the SHA check (no data, seconds):**

```bash
python reproduce.py            # official scorer on los_chuchos.csv + held-out table
python reproduce.py --verify   # rebuilds los_chuchos.csv from committed artefacts, checks SHA
```

**Full rebuild from the data bundle** (`MTAH_DATA_DIR`, CPU, ~45 min):

```bash
python features/extract_features.py                      # -> features/features.npz
python models/hmm_ensemble.py                            # -> evaluation/results/model_ensemble.csv
python models/ball_clean_ensemble/build_features_clean.py
python models/ball_clean_ensemble/run_clean_pipelines.py  # -> results/clean_ensemble.csv
python models/ball_clean_ensemble/train_tcn_clean.py --seed 7
python models/ball_clean_ensemble/train_tcn_clean.py --seed 1  --tag seed1
python models/ball_clean_ensemble/train_tcn_clean.py --seed 2  --tag seed2
python models/ball_clean_ensemble/train_tcn_clean.py --seed 11 --tag seed11
python models/ball_clean_ensemble/train_tcn_clean.py --seed 23 --tag seed23
python models/ball_clean_ensemble/train_tcn_clean.py --seed 42 --tag seed42
python models/ball_clean_ensemble/ensemble_tcn_logits.py --inputs \
  models/ball_clean_ensemble/tcn_clean_logits.npz \
  models/ball_clean_ensemble/tcn_clean_logits_seed1.npz \
  models/ball_clean_ensemble/tcn_clean_logits_seed2.npz \
  models/ball_clean_ensemble/tcn_clean_logits_seed11.npz \
  models/ball_clean_ensemble/tcn_clean_logits_seed23.npz \
  models/ball_clean_ensemble/tcn_clean_logits_seed42.npz \
  --out models/ball_clean_ensemble/tcn_clean_logits_ensemble.npz
python models/ball_clean_ensemble/run_tcn_variants.py     # -> evaluation/results/model_tcn_clean.csv
python models/consensus_ensemble.py                       # -> los_chuchos.csv
```

Intermediate artefacts are committed, with their hashes in
`models/ball_clean_ensemble/README.md`, so a stage can be checked on its own:

| artefact | SHA-256 (first 8) |
|---|---|
| `models/ball_clean_ensemble/features_clean_long_lastseen.npz` | `925ED02E` |
| `models/ball_clean_ensemble/results/clean_ensemble.csv` | `C8B99D47` |
| `models/ball_clean_ensemble/tcn_clean_logits_ensemble.npz` | `E89EDFF0` |
| `evaluation/results/model_tcn_clean.csv` | `39711AC6` |
| `evaluation/results/model_ensemble.csv` | `8EB5FBA0` |
| `los_chuchos.csv` | `852AA5FB` |

**Other checks:** `python evaluation/validate_submission.py los_chuchos.csv`
(official format check) and `python evaluation/run_evaluation.py --results .
--gt public --bootstrap 1000` (the scorer on its own). The held-out benchmark
is reproducible with `notebooks/heldout_idesse.ipynb`; its frozen output is in
`evaluation/heldout_idesse/`. The frame-by-frame animation is
`visual_validation/barca_atd_animation.ipynb`. The in-sample audit (paired
bootstrap, jackknife) is `models/ball_clean_ensemble/validate_consensus.py`.
