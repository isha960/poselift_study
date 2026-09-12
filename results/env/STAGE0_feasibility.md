# Stage 0 — Feasibility note & GPU-hour estimate

_2026-08-28. Gate document: study does not proceed to Stage 3 training without user approval._

## Feasibility paragraph

Four of the five planned models are feasible on PoseLift. **STG-NF** imports cleanly in the
existing `stgnf-poselift` env and there is already a working PoseLift runner with a completed
4-seed run to build on — lowest risk. The **motion-energy baseline** is trivial (pure NumPy in
the harness env). **COSKAD** and **MoCoDAD** are self-contained repos with clear entrypoints
(`train_COSKAD.py`, `train_MoCoDAD.py`) and in-repo `models/`+`utils/` packages; their conda
environments were still building at the end of Stage 0 (MoCoDAD solved, COSKAD still solving) and
will be smoke-tested in Stage 2 — moderate risk, mitigations noted in RUNLOG FAIL-02.
**Shopformer is not feasible from official code (RUNLOG FAIL-01):** the repo's code was uploaded
then deleted, and the transformer core, arg system, dataset loader and training loop were never
published on any branch — only the GCAE tokenizer half is recoverable from git history. Shopformer
therefore needs a user decision at this gate: cite-only, reimplement-and-label, or block on authors.
The PoseLift raw data is real, present, and verified (151 videos, 17-joint COCO, 6 cameras, test
base rate 0.317); no download needed. The released `Train/`/`Test/` folders give a usable
"split (a)", but the paper's scenario-windowed 3,721-frame test set is not reconstructible from any
repo, so 5.1 will be release-split comparable with a stated caveat.

## Smoke-test status

| Model | Env | Import / entrypoint | Verdict |
|---|---|---|---|
| STG-NF | `stgnf-poselift` (reuse) | `models.STG_NF.model_pose` imports; runner proven | **ready** |
| motion-energy | `poselift-harness` | n/a (to be written in `harness/`) | **ready** |
| COSKAD | `coskad` (building) | entrypoint + `utils/`,`models/` present in repo | ready pending env |
| MoCoDAD | `mocodad` (building) | entrypoint + `models/mocodad.py` present in repo | ready pending env |
| Shopformer | `environment.yml` names py3.13/torch2.6 | **entrypoint + core modules absent from repo** | **BLOCKED (FAIL-01)** |

## GPU-hour estimate (1× A6000 per run; 4 A6000s usable in parallel)

Per-run training+scoring cost (measured for STG-NF; estimated for the rest from architecture + configs):

| Model | per seed | basis |
|---|--:|---|
| STG-NF | ~0.02 h (~1 min) | measured, existing 4-seed run, 3 epochs, ~1K params |
| motion-energy | ~0 (CPU) | no training |
| COSKAD (Euclidean) | ~0.4 h | 100 epochs, small SAGC GCN, batch 2048, + Deep-SVDD eval |
| MoCoDAD | ~1.25 h | diffusion train (100 ep) + 50-sample generative eval × 5 transforms |
| Shopformer | (~0.15 h *if* reimplemented) | not counted below |

### Budget by stage (10 seeds main; Shopformer excluded)

| Stage | Composition | GPU-h |
|---|---|--:|
| 3 — main comparison, split (b) | (STG-NF+motion+COSKAD+MoCoDAD) × 10 seeds | ~17 |
| 3 — reproduction, split (a) | same × 10 seeds | ~17 |
| E1 — leave-one-camera-out | 6 folds × 4 models × 5 seeds | ~50 |
| E2 — normalisation (retrain) | 4 schemes × 5 seeds × 3 trainable models | ~33 |
| E2 — keypoint noise/dropout (eval-only) | ~10 perturb configs × 5 seeds × 3 models (MoCoDAD generative eval dominates) | ~37 |
| E2 — window-T (retrain) | 4 T-values × 5 seeds × 3 models | ~33 |
| E3 — efficiency | params/FLOPs/CPU-latency, few runs | ~1 |
| **Total (full, 10-seed)** | | **~188 GPU-h** |

### With the pre-registered reductions

| Reduction | saves |
|---|--:|
| Main + repro at **5 seeds** not 10 | ~17 h |
| E2 robustness with reduced MoCoDAD `n_generated_samples` (labelled exploratory) | ~27 h |
| Repro split (a) at 5 seeds | already counted |
| **Realistic total** | **≈ 110–130 GPU-h** |

**Wall-clock:** embarrassingly parallel across seeds/folds/models on 4 free A6000s →
**≈ 2–4 days**. MoCoDAD is ~70–75% of the entire GPU budget; if the compute ceiling is tight,
the first lever is MoCoDAD seed count / generative-sample count, not dropping a stage.
E4 (data-scale curve) is **not** in the total above; it would add ~+20 GPU-h.

## Immediate next actions (post-gate, before any test scoring)

1. Write & commit `results/preregistration.md` (primary metric, ±0.03 margin, seeds, σ grid {0,1,2,4,8},
   window T=24, normalisation = mid-hip + torso, E2/E3 grids, split (b) definition). **Frozen on commit.**
2. Stage 1: build both split files under `results/splits/`, verify **zero train↔{val,test} track-id overlap in code**,
   emit per-camera frame/track/video count tables + base rates.
3. Stage 2: finish `coskad`/`mocodad` env builds; per-model smoke (1 seed, few epochs, tiny subset) through the shared harness.

## Gate questions for the user

1. **Shopformer** — pick: (A) cite-only, (B) reimplement + label as non-official, (C) block on authors.
2. **Seed count** — 10 (≈130 GPU-h) or start at 5 (≈95 GPU-h) with option to extend?
3. **Compute approval** — OK to spend the estimated ~110–130 GPU-h across GPUs 0–3?
