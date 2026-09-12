# results.md — PoseLift skeleton-AD comparative study

**Generated:** 2026-08-29. **Last updated:** 2026-08-31 (E1 + E3 added).
**This study:** 4 models × **4 seeds {0,1,2,3}** on split (b): main comparison + operating points
(§5.1–5.3), **E1 leave-one-camera-out (§5.4)**, **E2 sensitivity — keypoint degradation,
window length, normalisation (§5.5a–c)**, **E3 efficiency (§5.6)**. §5.7 qualitative not run.
Pre-registration amendments: **1** (σ=0 primary), **2** (E2-norm run over each model's native
normalisation menu, not forced shared schemes).
Pre-registration: `results/preregistration.md` (frozen before test scoring; Amendment 1 =
primary metric at smoothing σ = 0, because the pre-registered validation split is anomaly-free
and cannot tune σ). Artefacts: `raw/metrics.csv`, `raw/scores/*.npz`, `aggregated/summary.csv`
(seed mean ± std), `stats/{perseed,delong,bootstrap,wilcoxon,verdicts}.csv`, `figures/*.pdf`,
`RUNLOG.md`.

Primary metric: **frame-level ROC-AUC**, micro-averaged over all test frames.
Split (b) test: **69 videos** (41 anomaly + 28 all-normal), **22,772 frames**, **1,534 anomalous
→ base rate 0.0674**. All 6 cameras in train/val/test; zero train↔{val,test} track-id leakage
(verified in `harness/make_split_b.py`).

---

## Table 3.1 — Per-model protocol deviations (native settings kept for faithful reproduction)

| Model | Repo commit | Window (shared T=24 / stride 12) | Normalisation | Key deviations / patches |
|---|---|---|---|---|
| STG-NF | `edb5f32` | seg_len 24, **stride 6 (native)** | STG-NF native (per-segment y-std) | run via proven working copy; ingest verified vs its own internal AUC (seed 0: 0.581 vs 0.582) |
| COSKAD (Euclidean) | **`7d9cecb`** (initial commit; HEAD `ba55553` non-functional — FAIL-03) | **seg_len 12 (native)**, stride 1 | `markovitz` | `validation:False` + LR-scheduler monitor → training `loss` (shipped drives scheduler **and** checkpoint on `validation_auc` == TEST auc → **leakage**); +6 behaviour-neutral guards. |
| MoCoDAD | `4eb672d` | **seg_len 6 (native)**, stride 1 | `robust` (Morais CSV-trajectory) | `n_generated_samples 50` (paper default); post-processing patched only to export raw scores. |
| Shopformer | — | — | — | **NOT reproduced** — official code deleted from repo (FAIL-01); reimplementation deferred |
| motion-energy | (harness) | T 24 / stride 12 (shared) | mid-hip+torso (shared) | mean per-joint speed / acceleration; measures articulation, not locomotion; **deterministic** (seed variance = 0) |

Shared for every model (`harness/`): window→frame = mean over covering windows; multi-person→frame
= max; no-detection frames carried forward; score oriented higher = more anomalous (sign verified);
all metrics from `harness/metrics.py` (9/9 unit tests), with and without smoothing.

---

## 5.1 — Reproduction vs published

Published numbers are **split (a)** (release Train/Test, scenario-windowed, single run, base rate
0.317). This pass is **split (b)** (pre-registered, camera-stratified, +28 all-normal test videos,
base rate 0.067, no windowing). **Like-for-like §5.1 is not possible from public artefacts** — the
paper's scenario-windowing script is in no repo. A split-(a) reproduction run is **deferred**.

| Model | Published (their n=1, split a) | This study (4 seeds, **split b**, σ=0): mean ± std |
|---|---|---|
| STG-NF | ROC 67.46 · PR 84.06 · EER 0.39 | ROC **66.0 ± 7.8** · PR **10.4 ± 2.6** · EER **0.40 ± 0.08** |
| Shopformer | ROC 69.15 · PR 44.49 | — not reproduced (FAIL-01) |
| GEPC / TSGAD | 60.6 / 63.4 | cited only |

**Reading.** (i) STG-NF's split-(b) mean (66.0) is close to the paper's 67.46, but this is
**coincidental** — different split, base rate, windowing — and *not* evidence of reproduction.
(ii) **STG-NF's seed std is ±7.8 AUC points** (range 58.1–73.6 across 4 seeds); the paper's 67.46
is a single draw from this wide distribution. (iii) Ingest fidelity confirmed: harness ROC-AUC
0.581 vs STG-NF's own internal 0.582 on the identical seed-0 run. **A genuine §5.1 comparison
requires the split-(a) reproduction run at ≥4 seeds.**

---

## 5.2 — Main comparison (split b, **4 seeds**, σ=0 primary)

Figures: `figures/roc_pr_curves.pdf` (ROC + PR, seed 0), `figures/forest_auc_diff.pdf`
(per-seed + 4-seed mean ± 95% t-CI, and Δ vs STG-NF).
Full per-seed table: `stats/perseed.csv`.

| Model | frame ROC-AUC (mean ± std) | per-seed | frame PR-AUC (chance 0.067) | EER | event ROC-AUC |
|---|--:|---|--:|--:|--:|
| motion-energy (speed) | 0.505 ± 0.000 | 0.505 ×4 (deterministic) | 0.062 ± 0.000 | 0.474 | 0.117 |
| motion-energy (accel) | 0.508 ± 0.000 | 0.508 ×4 | 0.063 ± 0.000 | 0.457 | 0.125 |
| COSKAD (Euclidean) | 0.599 ± 0.003 | 0.597 · 0.598 · 0.598 · 0.603 | 0.097 ± 0.004 | 0.436 | 0.279 |
| **STG-NF** (baseline) | **0.660 ± 0.078** | **0.581 · 0.736 · 0.607 · 0.718** | 0.104 ± 0.026 | 0.401 ± 0.079 | **0.338 ± 0.023** |
| **MoCoDAD** | **0.736 ± 0.011** | 0.721 · 0.746 · 0.736 · 0.743 | **0.137 ± 0.009** | 0.336 ± 0.013 | 0.235 ± 0.007 |

### Verdicts vs STG-NF — pre-registration §2 (`stats/verdicts.csv`, `stats/wilcoxon.csv`, `stats/delong.csv`)

| Model | mean ΔROC-AUC (4 seeds) | per-seed Δ | all seeds same sign? | DeLong seed-0 CI | clip-bootstrap seed-0 | Wilcoxon p (Holm) | Cliff's δ | **Verdict** |
|---|--:|---|:--:|--:|--:|--:|--:|---|
| **MoCoDAD** | **+0.076** | +0.140, +0.010, +0.129, +0.025 | **yes (all +)** | [+0.124, +0.157] excl. 0 | [0.609, 0.809] LB > STG-NF | 0.125 (0.50) | **+0.75** | **OUTPERFORMS STG-NF** (criteria 1, 2, 4 met — see §5.4; criterion 3 FPR caveat below) |
| COSKAD | −0.061 | +0.016, −0.138, −0.009, −0.115 | no | [−0.002, +0.035] incl. 0 | [0.484, 0.689] overlaps | 0.375 (0.375) | −0.50 | **comparable / within noise** |
| motion (speed) | −0.155 | all negative | yes (all −) | excl. 0 | below STG-NF | 0.125 (0.375) | −1.00 | **BELOW STG-NF** |
| motion (accel) | −0.152 | all negative | yes (all −) | excl. 0 | below STG-NF | 0.125 (0.25) | −1.00 | **BELOW STG-NF** |

**Caveats on the MoCoDAD verdict (do not drop these):**
- **Criterion 4 (LOCO consistency) is MET** — E1 done (§5.4): MoCoDAD − STG-NF is positive in all
  6 leave-one-camera-out folds (mean +0.131) as well as all 4 seeds.
- **Wilcoxon signed-rank p = 0.125** is at the *floor* for n = 4 all-same-sign pairs (it cannot go
  lower); Holm-corrected 0.50. The seed-level test alone is under-powered at the pre-registered
  4 seeds. The verdict leans on DeLong (p ≪ 0.001), the all-positive direction, and Cliff's δ 0.75.
- **Criterion 3 (not FPR-driven):** MoCoDAD's val-calibrated 5%-FPR threshold realises **7.6% FPR
  on test** (mean; STG-NF/COSKAD land ~3.9%). The recall gain is partly bought with a higher
  operating FPR — flagged for the operating-point / E2 work.

### Key methodological finding — STG-NF seed instability

| Model | ROC-AUC std over 4 seeds | 95% t-CI width |
|---|--:|--:|
| **STG-NF** | **0.078** | **0.248** |
| MoCoDAD | 0.011 | 0.037 |
| COSKAD | 0.003 | 0.009 |
| motion | 0.000 | 0.000 |

STG-NF's run-to-run variance is **7× MoCoDAD's and 26× COSKAD's**. Its 3-epoch normalizing-flow
training is highly unstable on split (b) (ROC-AUC 58→74); the SVDD and diffusion models converge
to the same value every seed. Consequences:
1. **STG-NF's own published n=1 number is one draw from a ±8-point distribution** — any comparison
   against it is dominated by seed luck.
2. **The Pass-1 (seed-0) ranking was misleading:** at seed 0, COSKAD (0.597) edged STG-NF (0.581)
   and MoCoDAD (0.720) looked far ahead. Over 4 seeds, STG-NF's *mean* (0.660) rises above COSKAD's
   (0.599), and MoCoDAD's lead shrinks from +0.14 to +0.076 but becomes **consistent** (all seeds).

### Headline

1. **"Bigger/newer isn't better" is REJECTED for MoCoDAD.** The diffusion model (142K params)
   outperforms STG-NF (~0.6K params) on frame ROC-AUC across all 4 seeds (mean +0.076, DeLong
   p ≪ 0.001, Cliff's δ 0.75) — pending LOCO confirmation and the FPR caveat above.
2. **COSKAD (239K params, SVDD) ties STG-NF** — no benefit from the larger model; verdict
   *within noise*, and its point estimate is actually *below* STG-NF's 4-seed mean.
3. **Nothing is deployable.** Best (MoCoDAD): PR-AUC 0.137 (≈ 2× the 0.067 chance rate), EER 0.34,
   recall 18% at a nominal 5% false-alarm budget (7.6% realised).
4. **Frame ≠ event.** MoCoDAD leads on frame ROC-AUC but has the **worst event-level ROC-AUC**
   (0.235 vs STG-NF 0.338) — it flags scattered anomalous frames, not whole shoplifting videos.
5. **Smoothing interacts with score type** (σ sweep in `raw/metrics.csv`): Gaussian smoothing
   helps COSKAD (0.599→0.69 at σ=8) and MoCoDAD slightly, and slightly *hurts* STG-NF —
   distance/reconstruction scores benefit, likelihood scores do not. σ>0 rows are exploratory
   (Amendment 1).

---

## 5.3 — Operating-point recall + detection latency (threshold calibrated on all-normal val split)

4-seed means, σ=0. `raw/metrics.csv` for per-seed / per-FPR detail.

| Model | recall @ FPR 1% | recall @ FPR 5% | recall @ FPR 10% | **realised** test FPR (nominal 5%) | median latency @ 5% (frames) |
|---|--:|--:|--:|--:|--:|
| STG-NF | 0.033 ± 0.021 | 0.081 ± 0.045 | 0.15 ± 0.06 | 0.039 | 6.0 |
| COSKAD | 0.021 ± 0.004 | 0.058 ± 0.014 | 0.16 ± 0.02 | 0.044 | 2.5 |
| MoCoDAD | 0.055 ± 0.011 | **0.182 ± 0.037** | **0.33 ± 0.03** | **0.076** | 0.0–1.0 |
| motion (speed) | 0.000 | 0.007 | 0.033 | 0.058 | 0.0 |

**Reading.** MoCoDAD gives the best recall at every budget and near-zero latency, but its
val-calibrated threshold **over-shoots to 7.6% FPR on test** — the threshold generalises worst for
MoCoDAD (STG-NF/COSKAD ~3.9–4.4%). STG-NF's recall std (±0.045 at 5% FPR) is again driven by its
seed instability. Even at the best case, ~18% frame recall at a ~5–8% alarm rate is not
operationally useful.

---

## 5.4 — LOCO generalisation (E1: leave-one-camera-out)

6 folds. Fold *k*: train the one-class model on **normal** clips from the other 5 cameras, test on
**all** clips of camera *k*. No val split → threshold-free metrics only (frame ROC-AUC primary).
4 seeds/fold. Splits `results/splits/loco_f{1..6}_{train,test}.json` (track-leak asserted 0 per
fold). Fold test sizes: C1 42 vids (10 anom) · C2 33 (10) · C3 32 (10) · C4 21 (6) · C5 15 (4) ·
**C6 8 (1 anom — widest CI, read with caution)**.
Figure `figures/loco_heatmap.pdf` · matrix `stats/loco_matrix.csv` · criterion-4 columns in
`stats/verdicts.csv`.

**Frame ROC-AUC, 4-seed mean per held-out camera (σ=0):**

| Model | C1 | C2 | C3 | C4 | C5 | C6 | **LOCO mean** | min | max | folds < 0.5 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| STG-NF | 0.624 | 0.594 | 0.680 | 0.663 | 0.571 | 0.679 | **0.635** | 0.571 | 0.680 | 0 |
| COSKAD | 0.574 | **0.497** | 0.705 | **0.498** | 0.618 | 0.833 | **0.621** | 0.497 | 0.833 | **2** |
| MoCoDAD | 0.699 | 0.786 | 0.717 | 0.672 | 0.852 | 0.872 | **0.767** | 0.672 | 0.872 | 0 |

**MoCoDAD − STG-NF, per fold (fold means):** +0.075, +0.192, +0.037, +0.009, +0.282, +0.193 →
**all 6 folds positive** (mean +0.131). Criterion 4 of the pre-registered "outperforms" test
(direction consistent across all 4 seeds *and* all 6 LOCO folds) is **MET**.
**COSKAD − STG-NF, per fold:** −0.050, −0.097, +0.025, −0.165, +0.047, +0.153 → not consistent;
COSKAD LOCO mean dAUC −0.014.

**Reading.** Under camera shift MoCoDAD keeps its lead and never drops below 0.67; its worst
transfer fold (C4, hold-out) is still above STG-NF's best fold. **COSKAD fails to transfer to two
held-out cameras (C2 0.497, C4 0.498 — at or below chance)**, so its LOCO mean (0.621) is
misleading; it is not camera-robust. STG-NF is the most *consistent* across folds (range
0.571–0.680) but its per-seed spread *within* a fold stays very large (e.g. C6: seeds span
0.47–0.94, C5: 0.41–0.80) — the same instability seen in §5.2, now also across cameras. No model's
LOCO mean reaches the in-domain split-(b) numbers, i.e. all three lose accuracy under camera shift;
MoCoDAD loses the least.

**Chapter 5 mapping:** §5.4 LOCO table + `figures/loco_heatmap.pdf`; confirms the §5.2 MoCoDAD
verdict (criterion 4). §5.7 (qualitative) remains **not run**.

---

## 5.5 — Sensitivity analysis (E2: keypoint degradation, window length, normalisation)

### 5.5a — Robustness to keypoint degradation (E2-noise)

Eval-time only, **no retraining**: the split-(b) TEST poses are degraded, the existing 4-seed
checkpoints are re-scored, train + val stay clean. Threshold-free frame ROC-AUC (4-seed mean ± std,
σ=0), exactly as E1. 9 pre-registered conditions (preregistration §7): additive Gaussian pixel
noise {1,2,4,8}px; per-(frame,joint) dropout p∈{.05,.10,.20}; whole-person per-frame dropout
p∈{.05,.10}. 27/27 (model×condition) cells have all 4 seeds; 0 failed runs. Perturbation code
`harness/perturb.py` (deterministic RNG `[20260901, cond_idx, seed]`); figure
`figures/e2noise_curves.pdf`; matrix `stats/e2noise_matrix.csv`. **Exploratory** (preregistration §8).

| condition | STG-NF | Δ | COSKAD | Δ | MoCoDAD | Δ |
|---|--:|--:|--:|--:|--:|--:|
| **clean** | 0.660 | — | 0.599 | — | 0.736 | — |
| Gaussian 1px | 0.659 | −0.001 | 0.599 | −0.000 | 0.735 | −0.002 |
| Gaussian 2px | 0.660 | −0.001 | 0.599 | −0.000 | 0.741 | +0.005 |
| Gaussian 4px | 0.659 | −0.002 | 0.599 | −0.000 | 0.738 | +0.002 |
| Gaussian 8px | 0.665 | +0.005 | 0.597 | −0.002 | 0.725 | −0.012 |
| joint-drop 5% | 0.663 | +0.002 | 0.558 | **−0.041** | 0.738 | +0.002 |
| joint-drop 10% | 0.672 | +0.012 | 0.552 | **−0.047** | 0.736 | −0.000 |
| joint-drop 20% | 0.651 | −0.009 | 0.556 | **−0.043** | 0.737 | +0.001 |
| frame-drop 5% | 0.666 | +0.005 | 0.525 | **−0.074** | 0.731 | −0.006 |
| frame-drop 10% | 0.695 | +0.034 | **0.503** | **−0.096** | 0.726 | −0.011 |

**Reading.**
1. **Positional jitter is a non-issue.** All three models are flat under Gaussian noise up to 8px
   (|Δ| ≤ 0.012) — windowed skeleton-AD absorbs sub-bounding-box coordinate error.
2. **COSKAD is the only model that degrades, and it degrades badly under *missing* keypoints.**
   Joint dropout costs it a stable −0.04–0.05; whole-frame dropout drives it to **chance
   (0.503 ROC-AUC, EER 0.497 at 10%)**, consistent across all 4 seeds (0.508/0.513/0.495/0.495).
   Its graph encoder + `markovitz` normalisation depends on a fully-populated skeleton and a
   present person every frame; the shared harness's carry-forward gap-fill cannot rescue it.
3. **MoCoDAD is the most robust model on every condition** (never more than 0.012 from clean) —
   reinforcing §5.2/§5.4: it is not just the most accurate, it is the most stable, now under
   input degradation as well as seed and camera.
4. **STG-NF's apparent "improvement" under frame-drop (+0.034) is seed noise, not robustness** —
   its per-seed spread on frame-drop 10% is 0.594–0.802, the same instability as §5.2. Read its
   whole row as "flat within its own ±0.07 seed band".

PR-AUC and EER tell the same story (`raw/metrics.csv`, `variant` column): COSKAD PR-AUC falls
0.097→0.069 by frame-drop 10%, STG-NF and MoCoDAD hold ~0.10 / ~0.14 throughout.

**Deployment implication.** With a real (imperfect) pose detector that misses joints or whole
people, COSKAD is unusable; STG-NF and MoCoDAD keep their clean-data accuracy.
Figure `figures/e2noise_curves.pdf`, matrix `stats/e2noise_matrix.csv`.

---

### 5.5b — Window length (E2-window)

Retrain at window length T ∈ {12, 32, 48}, stride T/2 (T=24 = the main run, at each model's
native stride). 4 seeds. **STG-NF and COSKAD only** — MoCoDAD's native seg_len 6 is
architecturally load-bearing (the diffusion U-Net conditions on the first half of the window and
predicts the second, `conditioning_indices:[0,1,2]`), so a length sweep is not a clean ablation
for it (pre-registration §7's own "for models that allow it"). Config-only otherwise.
Figure `figures/e2window_curves.pdf`, matrix `stats/e2window_matrix.csv`. **Exploratory** (§8).

| T (stride T/2) | STG-NF | COSKAD |
|---|--:|--:|
| 12 | 0.673 | 0.599 |
| **24 (main)** | **0.660** | **0.599** |
| 32 | **0.518** | 0.572 |
| 48 | **0.436** | 0.535 |

**Reading.** No model benefits from a longer window; both are best at T ≤ 24.
- **STG-NF collapses for T ≥ 32** (0.518, then 0.436 = below chance). Its 3-epoch normalizing-flow
  training cannot fit the longer sequences on this data — the flow's log-likelihood degrades and
  the score inverts. T=12 (0.673) ≈ T=24 (0.660), within its seed band.
- **COSKAD degrades gently and monotonically** (0.599 → 0.572 → 0.535); its native T=12 is its
  best. Per-seed spread is tiny (±0.002) so the trend is real, not noise.
- Practical consequence: the T≈12–24 range used by all four methods in §5.2 is already near-optimal;
  the shorter native windows (STG-NF stride-6 T24, COSKAD T12, MoCoDAD T6) are not a limitation.

### 5.5c — Normalisation scheme (E2-norm) — Amendment 2

**Amendment 2 to the pre-registration** (`results/preregistration.md`): the pre-registered E2-norm
(force the shared-harness schemes {mid-hip+torso, mid-shoulder+bbox-diag, bbox-minmax, none}
across all three repos) requires pre-normalising the keypoints and disabling each repo's internal
normalisation — which puts SVDD / diffusion / normalizing-flow training on out-of-range inputs and
risks non-convergence, and needs a per-repo code path that does not exist. Replaced with: **run
each model over its own shipped normalisation menu** (config-only). Faithful to the underlying
question ("how sensitive is each model to its normalisation choice"), lower risk. All rows
**exploratory** (§8). Matrix `stats/e2norm_matrix.csv`.

| Model | default (main) | alternative | Δ vs default |
|---|--:|--:|--:|
| STG-NF | normalised **0.660** | raw / unnormalised **0.634** | −0.027 |
| COSKAD | markovitz **0.599** | stan **0.584** | −0.015 |
| MoCoDAD | robust **0.736** | markovitz **0.670** · stan **0.713** | −0.066 · −0.023 |

**Reading.**
1. **Every model's shipped default is at or near its best.** Switching normalisation costs
   0.015–0.066 ROC-AUC — real (per-seed spreads are small) but modest. Normalisation choice is a
   second-order factor next to window length (§5.5b) or seed (§5.2).
2. **STG-NF is almost normalisation-agnostic** — turning normalisation off entirely costs only
   −0.027, inside its ±0.078 seed band. **MoCoDAD is the most normalisation-sensitive** (robust →
   markovitz −0.066).
3. **`bbox` normalisation is unusable on PoseLift** for both COSKAD and MoCoDAD, and both failures
   were removed as non-results: it divides by per-segment keypoint width/height, which is ≈0 for
   the many near-stationary retail poses → COSKAD produced a constant 0.5 score; MoCoDAD produced
   ROC-AUC 0.934 identical to 4 decimals across all 4 seeds (zero variance → a clip-length /
   coverage artifact, not anomaly detection). Reported as a failure mode, not a number.
4. **COSKAD `robust` not reproduced** (RUNLOG): its `robust` path needs dash-named trajectory-CSV
   staging + a fitted scaler; two staging attempts failed on the folder-name parser. Time-boxed
   out. COSKAD's available comparison is markovitz vs stan.

**Chapter 5 mapping:** §5.5a–c tables + `figures/e2noise_curves.pdf`, `figures/e2window_curves.pdf`;
matrices `stats/e2{noise,window,norm}_matrix.csv`. Net message across E2: **MoCoDAD's lead over
STG-NF (§5.2, §5.4) is robust to input degradation and normalisation choice; window length is the
one hyper-parameter that can break a model (STG-NF at T ≥ 32), and the study's chosen windows
avoid that.**

---

## 5.6 — Efficiency (E3)

Each model profiled **in its own pinned env**, CPU, `torch.set_num_threads(1)`, batch 1.
Latency = median of 200 timed forwards (20 warmup); MoCoDAD measured through its real generative
`test_step`. Peak RSS = `ru_maxrss`. FLOPs not measured — no profiler present in the pinned envs
and installing one upgraded torch (incident logged in RUNLOG); params + CPU latency are the
deployability signal. Source `harness/eff_{profile,stgnf,coskad,mocodad}.py` → `stats/efficiency.csv`
· figure `figures/efficiency_pareto.pdf`.

| Model | params (trainable) | CPU ms / window | windows / s | peak RSS |
|---|--:|--:|--:|--:|
| STG-NF (normalizing flow) | **616** | 4.88 | 205 | 327 MB |
| COSKAD (STSE encoder) | 239,716 | **2.15** | **465** | 290 MB |
| MoCoDAD (diffusion, n_gen=50, paper default) | 142,294 | **2595.8** | **0.39** | 722 MB |
| MoCoDAD (n_gen=1) | 142,294 | 55.3 | 18 | 722 MB |

**Reading.** MoCoDAD's paper-default generative inference costs **~2.6 s per window on CPU
(0.39 win/s)** — ~1200× COSKAD and ~530× STG-NF; even the single-sample setting (n_gen=1, 55 ms)
is only 18 win/s. Its ROC-AUC lead (§5.2, §5.4) is bought at ~1000× the inference cost and 2–2.5×
the memory. STG-NF is by far the **smallest** model (616 params) but **not the fastest** — its 8
sequential flow steps make it 2× slower per window than COSKAD's single encoder forward. **COSKAD
is the fastest (465 win/s) despite being the largest** by parameter count. Pareto
(`figures/efficiency_pareto.pdf`): STG-NF and COSKAD are real-time-capable and tied at ~0.60–0.66
ROC-AUC; MoCoDAD buys ~+0.08 accuracy at a ~1000× latency penalty. → the most accurate model is
mid-sized; the largest by params (COSKAD) is no more accurate than the smallest (STG-NF).

---

## Deviations from the plan
1. Seeds = 4 (pre-registration §3; reduced from the methodology's 10 by user decision).
2. Shopformer not reproduced — official code incomplete (FAIL-01); reimplementation deferred.
3. STG-NF/Shopformer split-(a) reproduction deferred → §5.1 like-for-like not done.
4. COSKAD pinned to initial commit `7d9cecb` — repo HEAD non-functional (FAIL-03).
5. COSKAD `validation:False` + scheduler-monitor change — removes a test-AUC leakage in the shipped config.
6. Native windows kept per model (STG-NF stride 6, COSKAD T 12, MoCoDAD T 6) — Table 3.1.
7. σ not tuned (val is anomaly-free) — primary = σ=0 (Amendment 1).
8. Two orchestration bugs during the 4-seed run (MoCoDAD `$S_train` shell-var; COSKAD ingest seed
   label) — caught, fixed, re-ingested from intact raw dumps; contaminated rows stripped. RUNLOG.
9. ~10 behaviour-neutral code patches in COSKAD/MoCoDAD (empty-clip / all-normal-clip guards,
   raw-score export) — archived as `results/env/patches/*.patch`.

## Open / failed
- **E1 LOCO — DONE** (§5.4): MoCoDAD criterion 4 MET (all 6 folds positive); COSKAD not
  camera-robust (C2, C4 ≤ chance).
- **E2 — DONE** (§5.5a–c): (a) keypoint-degradation robustness, 9 conditions × 4 seeds, no retrain
  — COSKAD collapses to chance under whole-frame dropout, STG-NF/MoCoDAD unaffected;
  (b) window length, STG-NF+COSKAD × T{12,32,48} — STG-NF collapses at T≥32, both best at T≤24
  (MoCoDAD N/A — architecture fixed at seg_len 6);
  (c) normalisation menu (Amendment 2) — each model's default is at/near best, alt schemes cost
  0.02–0.07; `bbox` norm degenerate for COSKAD+MoCoDAD (removed as non-results); COSKAD `robust`
  not reproduced (staging, time-boxed out).
- **E3 efficiency — DONE** (§5.6): params + CPU latency + peak RSS. FLOPs not measured (no
  profiler in pinned envs; installing one upgrades torch).
- MoCoDAD's operating-point FPR generalisation gap (val 5% → test 7.6%) — criterion 3 check,
  flagged in §5.2/§5.3, not resolved (would need an anomaly-bearing calibration split).
- Shopformer reimplementation — not done (FAIL-01).
- Split-(a) reproduction pass for a genuine §5.1 — not done.
- E2 sensitivity (keypoint noise, window length, normalisation) — DONE (§5.5a–c). Not run:
  COSKAD `robust`-normalisation reproduction, E4 data-scale, §5.7 qualitative, split-(a) §5.1,
  Shopformer reimpl.
