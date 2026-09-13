# PoseLift Skeleton-Based Anomaly Detection — Comparative Study

**Repo:** https://github.com/isha960/poselift_study

A reproducible, pre-registered comparative evaluation of skeleton-based video
anomaly-detection methods on [PoseLift](https://arxiv.org/abs/2501.06591)
(WACV 2025 Workshops) — a privacy-preserving, pose-only retail-shoplifting
benchmark. One shared evaluation harness runs every model under an identical,
leakage-controlled protocol with repeated seeds and proper statistics,
so that reported differences reflect the models, not the evaluation code.

## Headline result

**MoCoDAD (motion-conditioned diffusion, 142K params) outperforms the STG-NF
baseline (~0.6K params, ICCV 2023)** on frame-level ROC-AUC — consistently
across all 4 training seeds *and* all 6 leave-one-camera-out folds — but at
roughly **1000× the inference cost** and with a worse operating-point false-alarm
rate. **COSKAD (Deep-SVDD, 240K params) is statistically tied with STG-NF** in
the main comparison and **collapses under two forms of realistic degradation**:
held-out cameras (2 of 6 folds fall to or below chance) and missing keypoints
(whole-frame dropout → chance-level AUC). The largest single effect in the
whole study is **STG-NF's own run-to-run seed variance** (std 0.078 AUC,
7–26× the other models') — larger than any between-model difference reported
in the source papers. Full numbers, verdicts, and readings: **[`results/results.md`](results/results.md)**.

## What's in this repo

| Path | Contents |
|---|---|
| `harness/` | The shared evaluation harness: data loading (`io_poselift.py`), representation/normalisation (`repr.py`), format conversion (`convert.py`), window→frame aggregation (`aggregate.py`), metrics (`metrics.py`), DeLong test (`delong.py`), statistics (`stats.py`), figure/table generation (`assemble_results.py`), per-model staging/ingest scripts, and the experiment orchestration shell scripts (E1 LOCO, E2 noise/window/norm, E3 efficiency) |
| `configs/` | Every model × seed × fold × experiment-variant YAML config actually used |
| `results/preregistration.md` | Frozen pre-registration (splits, metrics, verdict rule, statistics plan) **written before any test-set score was looked at**, plus Amendment 1 (smoothing σ) and Amendment 2 (normalisation-scheme ablation) |
| `results/results.md` | The full write-up — one section per experiment, every number mapped to its script/figure/table, with a factual reading |
| `results/RUNLOG.md` | Append-only log of every run, failure, deviation, and GPU-hour total |
| `results/raw/metrics.csv` + `results/raw/scores/*.npz` | Every metric of every run in tidy form, and the underlying per-frame score/label arrays — all figures and tables in `results/` are regenerable from these via `harness/assemble_results.py` |
| `results/stats/`, `results/figures/` | Generated statistics tables and vector-PDF figures |
| `results/env/` | Pinned commit hashes for every external model repo, pip freezes, conda environment files |
| `results/splits/` | The exact train/val/test and leave-one-camera-out split files used (with in-code track-leakage verification) |

**Not included** (see `.gitignore`, and the "Reproducing" section below):
cloned third-party model repos (`external/`), converted PoseLift keypoint data
(`data_converted/`, `data_coskad/`, `data_mocodad/` — regenerable, redistribution
terms unclear), and trained model checkpoints / raw training logs
(`results/model_runs/` — not needed to verify any reported number, since the
scored outputs are already in `results/raw/`).

## Models evaluated

| Model | Type | Params | Status |
|---|---|--:|---|
| STG-NF | Normalizing flow (ICCV 2023) | ~616 | reproduced, 4 seeds + LOCO + E2/E3 |
| COSKAD (Euclidean) | Deep-SVDD graph encoder | ~240K | reproduced, 4 seeds + LOCO + E2/E3 |
| MoCoDAD | Motion-conditioned diffusion | ~142K | reproduced, 4 seeds + LOCO + E2/E3 |
| motion-energy | Deterministic joint-speed/accel baseline | — | reproduced (harness-native) |

## Experiments

- **§5.2** Main comparison — frame ROC-AUC/PR-AUC/EER/event-AUC, 4 seeds, pre-registered
  DeLong + cluster-bootstrap + Wilcoxon verdicts vs. STG-NF
- **§5.3** Operating points — recall @ fixed FPR, detection latency, threshold calibrated on an all-normal validation split
- **§5.4 (E1)** Leave-one-camera-out — 6 folds × 4 seeds, threshold-free generalisation test
- **§5.5 (E2)** Sensitivity — (a) robustness to keypoint noise/dropout at eval time, no retraining; (b) window-length ablation; (c) each model's native normalisation-scheme menu
- **§5.6 (E3)** Efficiency — parameter count, CPU single-thread latency, peak memory

## Reproducing

1. Get the PoseLift release data (`Pickle_files/{Train,Test,GT}`) — see the
   [PoseLift paper](https://arxiv.org/abs/2501.06591) for access.
2. Clone the pinned external model repos at the commits in
   `results/env/external_repo_commits.txt`.
3. Build the three environments from `results/env/*_environment.yml`
   (`poselift-harness`, `coskad`, `mocodad`; STG-NF reuses a pre-existing PyTorch 1.10 env).
4. `harness/convert.py` → produces `data_converted/`; `harness/stage_{stgnf,coskad,mocodad}.py`
   → stages each model's native input format from a split JSON in `results/splits/`.
5. Run any `harness/seeds_*.sh` / `e1_*.sh` / `e2_*.sh` orchestration script, or just
   regenerate every table/figure from the committed raw scores:
   `python harness/assemble_results.py`.

Every number in `results/results.md` traces to a script run on real PoseLift
data; no metric was fabricated, interpolated, or estimated — failed or
unreproducible runs are recorded as failures in `RUNLOG.md`, never as a guessed value.
