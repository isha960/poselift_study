# RUNLOG — PoseLift skeleton-AD comparative study

Every run, failure, deviation, time-box, and GPU-hour total is recorded here.
Append-only. Times in Europe/London.

---

## Stage 0 — Environment setup & feasibility (2026-08-28)

### Host
- 8 × NVIDIA RTX A6000 (48 GB). GPUs 0–3 free; GPUs 4–7 held by another user (~42 GB each) → **4 GPUs usable**.
- Driver CUDA 12.8. System `nvcc` 10.1 (irrelevant; all envs use pip/conda CUDA wheels).
- Disk: 499 GB free on `/` (1.8 T, 72% used).
- Project root: `~/poselift-study/`  (`external/ harness/ configs/ results/`).

### External repos cloned + pinned (`results/env/external_repo_commits.txt`)
| repo | commit | date | state |
|---|---|---|---|
| orhir/STG-NF | `edb5f32` | 2023-10-13 | full, imports OK |
| aleflabo/COSKAD | `ba55553` | 2025-05-19 | full, self-contained `models/`+`utils/` |
| aleflabo/MoCoDAD | `4eb672d` | 2026-05-17 | full, self-contained `models/`+`utils/` |
| TeCSAR-UNCC/Shopformer | `a089601` | 2025-07-05 | **CODE DELETED — see FAIL-01** |
| TeCSAR-UNCC/PoseLift | `d1633f3` | 2025-02-24 | **no code** — README + sample media only |

### Environments
- **`poselift-harness`** (conda, Python 3.11): numpy, scipy, scikit-learn, pandas, matplotlib,
  pyyaml, tqdm, statsmodels, torch 2.13.0+cpu (for FLOPs/latency utilities only), ptflops, fvcore, pytest.
  Created & verified. Frozen → `results/env/harness_pip_freeze.txt`, `harness_environment.yml`.
- **`stgnf-poselift`** (pre-existing, reused): torch 1.10.1+cu113. STG-NF import smoke test PASSED.
  This env already produced a working 4-seed STG-NF/PoseLift run (`~/PoseLift/STG-NF/`, Aug 23).
- **`coskad`** (conda from `COSKAD/environment.yml`, py3.9 / torch 1.11 / PL 1.6.3):
  build STARTED in background 16:5x, still solving at end of Stage 0 → status pending, see FAIL-02 risk.
  Log: `results/env/build_logs/coskad_env_build.log`.
- **`mocodad`** (conda from `MoCoDAD/environment.yaml`, py3.11 / torch 2.0.1 / PL 2.0.6):
  build STARTED in background, still solving at end of Stage 0 → status pending.
  Log: `results/env/build_logs/mocodad_env_build.log`.

### Data (raw, already on disk — NOT re-downloaded)
`~/PoseLift/Pickle_files/{Train,Test,GT}` — verified real PoseLift release:
- 104 train `.pkl` + 47 test `.pkl` + 47 GT `.npy`  → **151 videos total** (paper says 155; release dropped 4).
- pkl format: `dict{frame:int → dict{person_id:int → [bbox(4), keypoints ndarray(17,3)=(x,y,conf)]}}`.
- **17 joints, COCO layout** confirmed. bbox looks like `xyxy` (to confirm in Stage 1).
- Camera prefixes 1–6 present in BOTH train and test.
  - train per-cam: {1:31, 2:22, 3:21, 4:14, 5:10, 6:6}
  - test  per-cam: {1:11, 2:11, 3:11, 4:7, 5:5, 6:2}
- GT (test only): 41/47 videos contain anomaly frames, 6 all-normal.
  Test frames total **4846**, anomalous **1534**, normal **3312**, **base rate = 0.3165** (video-level, un-windowed).
- The released `Train/`+`Test/` folders constitute a fixed split → candidate "split (a) paper-comparable".

### Existing assets reused
- `~/PoseLift/STG-NF/convert_anno.py` — pkl→JSON converter (`{pid:{frame:{keypoints:[51 floats], scores:None}}}`).
  Basis for the shared harness converter.
- `~/PoseLift/STG-NF/` — working PoseLift STG-NF runner + staged JSON data (104/47) + 4 completed seed runs.

### GPU-hours consumed in Stage 0
0.0 (env solves + smoke imports only; no training).

---

## FAILURES / BLOCKERS

### FAIL-01 — Shopformer official code is not published (BLOCKER for model #4)
`TeCSAR-UNCC/Shopformer` @ `a089601` contains only `README.md`, `LICENSE`, `environment.yml`, `Images/`.
Git history shows the authors uploaded code, then deleted all of it:
`4c94f31 Delete main_to.py`, `4ec037b Delete GCAE directory`, `da356ba Delete models_graph directory`,
`5ce6686 Delete pose_utils.py`, `c27a081 Delete eval2.py`, `3... Delete data_utils.py`.
Recoverable from history (`git show f6a8787:<file>`): `main_to.py`, `GCAE/graph.py`,
`models_graph/gcae/{gcae.py,gcae_training.py}`, `models_graph/graph/{graph,pygeoconv,sagc,st_graph_conv_block}.py`,
`data_utils.py`, `pose_utils.py`, `eval2.py` (~1680 lines).
**But `main_to.py` imports modules that were NEVER in the repo on any branch/tag:**
`args` (arg system), `dataset.get_dataset_and_loader` (the PoseLift loader),
`utils.train_utils_token` (training loop), `models.TransformerPredictor` / `models.ReformerPredictor`
(**the transformer — the core of the method**). Only the GCAE *tokenizer* half was ever released.
Import names mirror STG-NF's scaffold, implying it was built on the STG-NF repo, but that glue is absent.

**Consequence:** Shopformer cannot be *reproduced* from official code. Options for the user (Stage 0 gate decision):
- (A) Demote Shopformer to **cited-only**, like GEPC/TSGAD/GiCiSAD. Study runs with 4 models
  (STG-NF, motion, COSKAD, MoCoDAD). Lowest risk, fully honest.
- (B) **Reimplement** the transformer + loader + training loop from the paper and the partial code,
  label all its numbers "our reimplementation (not official)" everywhere, never compare like-for-like
  against the paper's 69.15. Est. multi-day engineering; GPU cost small (~10 min/seed).
- (C) Email authors (nrashvan@charlotte.edu) for the code; block on reply.
No fabricated Shopformer numbers will be produced under any option.

### FAIL-02 (risk, not yet realised) — COSKAD/MoCoDAD conda solves slow
Both `environment.yml` files carry full 2022–2023 transitive pins. conda 24.5 classic solve was still
running at end of Stage 0. Fallbacks if a solve fails or exceeds a 1-day time-box:
libmamba solver; or minimal hand-built env (python + torch@pinned + pytorch-lightning@pinned +
numpy/scipy/sklearn + geoopt for COSKAD). Will be resolved in Stage 2 smoke and recorded here.

### Not a failure, but a documented discrepancy for 5.1
The paper's Table 2 test set = 3,721 frames (2,221 normal + 1,500 anomalous), scenario-windowed.
The released `Test/` pkls are **full videos** = 4,846 frames (3,312 + 1,534), un-windowed, and only
41 (not 43) contain anomalies. No scenario-windowing script ships in any repo. The "split (a)
paper-comparable" reproduction will therefore be *release-split* comparable, not *paper-table* comparable;
5.1's like-for-like claim will be caveated accordingly (Ground Rule / Stage 1 clause).

---

## Stage 1 — Data & splits (2026-08-28)

- PoseLift raw verified: 151 videos (104 Train / 47 Test), 6 cameras, COCO-17, bbox xyxy.
  4 videos have pkl_frames != gt_frames (1_257, 1_272, 5_299, 6_112) -> scored on GT length,
  logged per-video in data_converted/*/manifest.json.
- `results/splits/split_b_{train,val,test}.json` + `split_b_counts.csv` written.
  - train 66 vids / ~60k normal frames / 331 tracks / all 6 cams
  - val   16 vids / ~13k normal frames /  75 tracks / all 6 cams
  - test  69 vids (41 anomaly + 28 normal) / 22,772 frames / 1,534 anomaly / base rate 0.067 / all 6 cams
  - leakage check IN CODE (`harness/make_split_b.py`): globally-unique track ids
    "<video>::<pid>"; train INTERSECT val = 0, train INTERSECT test = 0, val INTERSECT test = 0. PASS.
- `results/splits/split_a` = the release Train/Test folders (for 5.1 only; caveat in preregistration sec.6).

## Stage 1b — Shared harness built & unit-tested

`harness/`:
- io_poselift.py   - .pkl / GT loader, split iteration
- repr.py          - shared normalisation (mid-hip+torso primary; +3 E2 schemes), within-track interp
- convert.py       - .pkl -> COCO-JSON (STG-NF/COSKAD) + CSV-trajectory 34-col (MoCoDAD) + GT; ran, output in data_converted/
- aggregate.py     - windows->person-frames (mean), persons->frame (max), gap-fill (carry-fwd), Gaussian smooth
- metrics.py       - frame_roc_auc, frame_pr_auc(+base rate), eer, event_roc_auc, recall/fpr/f1@fixed-FPR, detection latency
- delong.py        - DeLong 1988 fast AUC + two-correlated-ROC test
- motion_energy.py - baseline model (mean per-joint speed / acceleration on shared repr)
- run_eval.py      - orchestrator: frame series -> all metrics (with & without smoothing) -> raw/metrics.csv + raw/scores/*.npz
- tests/test_harness.py - **9 tests PASS**: DeLong==sklearn AUC; random scorer AUC~0.5;
  window->frame mean & person->frame max; gap-fill carry-forward; smoothing variance monotone;
  motion-energy articulation jerk >> still; event ROC separable; detection latency.

Note recorded for results.md: mid-hip centring removes whole-body translation, so the
motion-energy baseline on the shared repr measures ARTICULATION energy, not locomotion
(a fast walk-out would not score high). E2 'none' scheme retains locomotion.

GPU-hours consumed to end of Stage 1b: 0.0.

---

## Stage 2 — per-model smoke + runs (2026-08-29)

### motion-energy baseline  [DONE]
- Pure harness (`run_motion.py`), no training. Full split-b run, seeds n=1.
- frame ROC-AUC (sig=0, primary): speed **0.5054**, accel **0.5082**  -> ~chance (expected floor).
- Note: mid-hip centring removes whole-body translation -> baseline measures articulation, not locomotion.

### STG-NF  [DONE]
- env `stgnf-poselift` (torch 1.10.1). Repo pin `edb5f32`, run via existing working copy `~/PoseLift/STG-NF`.
- Data staged split-b in STG-NF format via `harness/stage_stgnf.py`
  ({scene:02d}_{clip:04d}_alphapose_tracked_person.json, unpadded frame keys).
- Train: seed 0, epochs 3, seg_len 24, **seg_stride 6 (native; shared protocol says 12 -> logged deviation)**,
  batch 256, lr 5e-4, adamx. ~1 min. Val pass = re-run w/ --checkpoint on split_b_val (STG-NF's own AUC
  call raises on all-normal val, harmless -- val CSVs written before that).
- Ingest `harness/run_stgnf_ingest.py`: label = 1 - csv_col0, anomaly_score = -csv_col1 (STG-NF score
  higher = more normal). STG-NF native window->frame mapping kept (frame+seg_len//2, gaps='most normal').
- frame ROC-AUC (sig=0, primary): **0.5808**  (STG-NF internal AUC 0.582 -- ingest verified).
  vs its own paper/split-a ~0.64-0.67: split-b is harder (28 extra all-normal test videos, base rate 0.067).

### COSKAD (Euclidean)  [smoke DONE; full train RUNNING]
- **Repo HEAD `ba55553` is BROKEN for the euclidean-encoder path** (see FAIL-03). Re-pinned to initial
  commit **`7d9cecb`** -- the only self-consistent state. Recorded in external_repo_commits.txt.
- env `coskad` (py3.9.11 / torch 1.11 / PL 1.6.3). conda installed a py3.10+ pip into py3.9 -> fixed
  (`pip==23.3.2`) + minimal euclidean deps + matplotlib + geoopt.
- Patches on `7d9cecb` (all logged, behaviour-neutral or leakage-removing):
  1. config-completion: `encoder_type: 'STS_GCN'` (euclidean STC cfg omits it; LitEncoder reads it).
  2. `models/euclidean_encoder_dynamicCenter.py` LR scheduler: monitor `validation_auc`->`loss`,
     `mode='max'`->`'min'`. Shipped drives the plateau scheduler on validation_auc == TEST auc
     (config sets validation:True, validation==test) -> **test leakage**; we set `validation: False`
     and monitor training loss. DEVIATION, prevents leakage.
  3. `eval_COSKAD.py`: (a) `auc=nan` init before try (all-normal split-b clips make roc_auc_score raise),
     (b) guard empty `error_per_person` -> zeros(n_frames) (clips whose tracks are all < seg_len),
     (c) skip transformations that yield no clip scores, (d) `COSKAD_RAW_DUMP` env -> dump raw
     pre-smoothing per-clip scores (mean over 5 transforms) for the shared harness.
- config: dataset_choice HR-STC, normalization markovitz (JSON path, not robust/CSV), seg_len 12
  (COSKAD native; shared T=24 -> logged deviation), vid_res [1920,1080], validation False,
  wandb off, seed 0. `test_path` -> testing/test_frame_mask (init_sub_args sets gt_path=test_path
  when validation:False).
- Data staged split-b via `harness/stage_coskad.py` ({scene}_{clip}_x.json, 6-digit frame keys).
- Smoke (5 train / 5 test, 2 epochs): trains (STSE 239K params), eval dumps 5 raw score arrays OK.
- Full train: seed 0, 100 epochs, GPU 1, ~12 s/epoch (~20 min). PID 3515045.
- Ingest: `harness/ingest_raw_dump.py COSKAD <val_raw> <test_raw>` (no sign flip; score = hypersphere distance).

### MoCoDAD  [smoke DONE; full train RUNNING]
- env `mocodad` (py3.11 / torch 2.0.1 / PL 2.0.6), clean. Repo pin `4eb672d`.
- Patches on `models/mocodad.py` post_processing (logged): init RAW_SCORES; guard empty
  error_per_person -> zeros; `MOCODAD_RAW_DUMP` env -> dump raw pre-smoothing per-clip scores
  (mean over transforms); wrap final roc_auc_score in try/except -> nan. No behaviour change to scoring.
- config: dataset_choice HR-STC, normalization_strategy 'robust' (CSV-trajectory path, paper default),
  seg_len 6 (MoCoDAD native; shared T=24 -> logged deviation), n_generated_samples 50 (paper default),
  vid_res [1920,1080], validation False (-> ModelCheckpoint monitors 'loss_noise', no patch needed),
  wandb off, seed 0. `test_path` -> testing/test_frame_mask.
- Data staged split-b via `harness/stage_mocodad.py`: CSV trajectories in **"{scene}-{clip}"** (DASH)
  folders (MoCoDAD parses folder as scene-clip via '-'); GT npy stay "{scene}_{clip}.npy".
- Smoke (5 train / 5 test, 2 epochs, n_gen 4): trains (STSAE cond 69K + STSAE_Unet 73K = 142K params),
  eval dumps 5 raw score arrays OK.
- Full train: seed 0, 100 epochs, n_gen 50, GPU 3, ~75 s/epoch (~2 h). PID 3518992.

### GPU-hours consumed so far (Stage 2): ~0.05 (STG-NF) + COSKAD full (~0.35, running) + MoCoDAD full (~2.5, running).

## FAIL-03 -- COSKAD repo HEAD is non-functional for its own euclidean-encoder config
`aleflabo/COSKAD` @ `ba55553` (also `b292520`): `models/euclidean_encoder_dynamicCenter.py`'s
`LitEncoder` calls `STSE(c_in=, h_dim=, channels=, encoder_type=)`, but:
  - commit `e875137` refactored `models/common/components.py` (Encoder signature) without updating
    the `models/stse/*` callers;
  - commit `738de05` deleted `models/stse/`, `models/stsae/`, `models/stsve/` and added
    `models/sts/{ae,vae}.py` with an INCOMPATIBLE `STSE.__init__(input_dim, layer_channels,
    hidden_dimension, latent_dim, ..., projector, distance, ...)`, and never updated `LitEncoder`.
Net: HEAD raises `ModuleNotFoundError: No module named 'models.stse'`; `b292520` raises
`TypeError: int + list` in `common/components.py`. Only the **initial commit `7d9cecb`** has a
mutually consistent euclidean path. COSKAD is therefore reproduced from `7d9cecb` with the patches
listed above; this is documented as a repo-integrity finding, not a silent workaround.

### COSKAD  [DONE]
- Full train 100 epochs (GPU 1, ~22 min, final train loss ~2.4e-4). Ckpt selected by min training
  loss = epoch 93. Eval: 5 test-time-augmentation transforms, raw pre-smoothing per-clip scores dumped.
- Extra patch: per-transformation `ROC()` call wrapped in try/except (all-normal val split raises).
- Val pass: 16/16 clips dumped; test pass: 69/69 (5 clips have all tracks < seg_len 12 -> zero score).
- frame ROC-AUC (sig=0, PRIMARY): **0.5971**. (sig=8: 0.690 -- COSKAD internally applies score_process
  win_size=50 smoothing before its own AUC; its per-transform internal AUC was ~0.66-0.67. The shared
  harness at the pre-registered sig=0 shows COSKAD ~= STG-NF, both ~0.58-0.60.)
- metrics.csv deduped (smoke rows overwritten by full-run rows; keep last per metric key).

### Standings so far (split-b, n=1, sig=0 primary, frame ROC-AUC)
  motion-energy(speed) 0.505 | motion-energy(accel) 0.508 | STG-NF 0.581 | COSKAD 0.597 | MoCoDAD pending

### MoCoDAD  [DONE]
- Full train 100 epochs (GPU 3, ~2 h). Ckpt selected by min 'loss_noise' = epoch 96.
- Eval: test pass n_generated_samples=50 (~6 min fwd + generative aggregation), val pass; raw pre-smoothing
  per-clip scores dumped (mean over transforms). test 69/69 (1 empty clip), val 16/16.
- frame ROC-AUC (sig=0, PRIMARY): **0.7205**.  DeLong vs STG-NF: dAUC +0.140, CI [+0.124,+0.157], p~1e-64.
  clip-bootstrap 95% CI [0.609, 0.809] -- lower bound clears STG-NF point est (0.581).
- event ROC-AUC 0.228 (WORSE than STG-NF 0.317) -- frame-strong, event-weak.
- fixed-FPR threshold generalisation: val 5% -> test 7.7% (worst of the 3; STG-NF/COSKAD ~4%).

## Stage 2 COMPLETE -- final standings (split-b, n=1, sig=0 primary, frame ROC-AUC)
  motion-energy(speed) 0.505 | motion-energy(accel) 0.508 | STG-NF 0.581 | COSKAD 0.597 | **MoCoDAD 0.720**
  Verdicts (pre-reg): MoCoDAD promising (2/4 "outperforms" criteria, needs multi-seed+LOCO);
  COSKAD comparable/within-noise; motion baselines significantly below STG-NF.
  Headline: "bigger/newer isn't better" does NOT hold -- the diffusion model (MoCoDAD, 142K params)
  clearly leads; the largest model (COSKAD, 239K) ties the smallest (STG-NF, ~0.6K).
  All far from deployable (best PR-AUC 0.126 vs 0.067 chance; best recall@5%FPR = 16%).

## Artefacts produced this pass
  results/preregistration.md  (+ Amendment 1)
  results/RUNLOG.md
  results/env/{ENV_SUMMARY.txt, external_repo_commits.txt, *_pip_freeze.txt, *_environment.yml, build_logs/}
  results/splits/{split_b_{train,val,test}.json, split_b_counts.csv, _video_inventory.json}
  results/raw/metrics.csv                 (tidy: model,seed,split,fold,variant,smoothed,sigma,metric,value,note)
  results/raw/scores/*.npz                (per-run frame scores+labels+per-video series; 5 runs)
  results/aggregated/summary.csv
  results/stats/{delong.csv, bootstrap.csv}
  results/figures/{roc_pr_curves.pdf, forest_auc_diff.pdf}   (vector)
  results/results.md                      (THE HANDOFF -- one section per Chapter 5 subsection)
  harness/  (io_poselift, repr, convert, aggregate, metrics, delong, motion_energy, run_eval,
             score_run, stats, assemble_results, stage_{stgnf,coskad,mocodad}, run_{motion,coskad_eval,
             mocodad_eval}, run_stgnf_ingest, ingest_raw_dump, make_split_b, tests/) -- 9/9 unit tests pass

GPU-hours consumed this session: STG-NF ~0.03 + COSKAD train ~0.37 + COSKAD eval ~0.1
  + MoCoDAD train ~2.1 + MoCoDAD eval ~0.35 + motion ~0  =  ~2.9 GPU-h  (est. was 1.5-2.5)

---

## 4-SEED PASS (2026-08-29, seeds {0,1,2,3}, split b, main comparison)

### Orchestration bugs caught & fixed mid-run
- MoCoDAD `seeds_mocodad.sh`: `-c ...mocodad_b_s$S_train.yaml` -> bash parsed `$S_train` as an
  (empty) var -> all 3 seed trainings crashed instantly, eval ran on nothing, ingest wrote
  garbage ROC-AUC 0.500. FIX: `${S}_train`. Relaunched; trained fully (~30 min parallel on GPU 1,2,3).
- COSKAD `seeds_coskad.sh`: ingest line ran without the seed arg (edit didn't reach the already-parsed
  loop) -> seeds 1,2,3 ingested as "seed 0", overwriting the seed-0 npz. Training + raw dumps were
  fine. FIX: re-ingested all 4 seeds from intact raw_test_s{1,2,3}/raw_val_s{1,2,3} (+ unsuffixed
  raw_test/raw_val for seed 0) with correct seed labels. Stripped all contaminated COSKAD/MoCoDAD
  rows from metrics.csv first, then re-ran clean.
- motion-energy is deterministic -> seeds 1-3 identical to seed 0 by construction (variance = 0);
  rows written so the multi-seed grid is complete.

### 4-seed frame ROC-AUC (sig=0), split b
| model | per-seed | mean | std | 95% t-CI |
|---|---|---|---|---|
| STG-NF | 0.5808 0.7358 0.6067 0.7180 | 0.6604 | **0.078** | [0.536, 0.784] |
| COSKAD | 0.5971 0.5976 0.5982 0.6030 | 0.5990 | 0.003 | [0.595, 0.603] |
| MoCoDAD | 0.7205 0.7463 0.7358 0.7432 | 0.7364 | 0.011 | [0.718, 0.755] |
| motion(speed) | 0.5054 x4 | 0.5054 | 0.000 | - |
| motion(accel) | 0.5082 x4 | 0.5082 | 0.000 | - |

### Verdicts (pre-registration sec.2)  -> stats/verdicts.csv
- **MoCoDAD: OUTPERFORMS STG-NF** on the seed-level criteria: mean dAUC +0.076 (>=0.03),
  all 4 seeds positive (+0.140,+0.010,+0.129,+0.025), DeLong seed-0 CI [+0.124,+0.157] excludes 0,
  clip-bootstrap LB (0.609) > STG-NF seed-0 point (0.581), Cliff's delta +0.75.
  CAVEATS: (c4) LOCO not run; (Wilcoxon) p=0.125 = n=4 floor, Holm 0.50 -> seed-test underpowered;
  (c3) MoCoDAD realises 7.6% test FPR at the nominal 5% val threshold vs ~3.9% for STG-NF/COSKAD.
- **COSKAD: comparable / within noise** -- mean dAUC -0.061 (STG-NF mean pulled up by its high
  seeds), direction inconsistent, DeLong CI includes 0.
- **motion(speed/accel): BELOW STG-NF** -- all seeds negative, Cliff's delta -1.0.

### Headline
- STG-NF seed instability (std 0.078, 7x MoCoDAD, 26x COSKAD) is the largest effect in the study.
  The seed-0 (Pass-1) ranking was misleading: COSKAD's seed-0 edge over STG-NF vanishes; MoCoDAD's
  +0.14 lead shrinks to +0.076 but becomes consistent across seeds.
- "bigger/newer isn't better" REJECTED for MoCoDAD (diffusion, 142K) vs STG-NF (~0.6K);
  HOLDS for COSKAD (239K SVDD) which ties STG-NF.
- Nothing deployable: best (MoCoDAD) PR-AUC 0.137, EER 0.34, recall 18% @ ~5-8% FPR,
  worst event-level ROC-AUC (0.235).

### GPU-hours this pass
STG-NF x3 ~0.1 + COSKAD x3 ~1.1 (parallel wall ~25min) + MoCoDAD x3 ~1.5 (parallel wall ~35min)
+ evals ~0.6  =  ~3.3 GPU-h.   Session total so far ~6.2 GPU-h.

### Still open
E1 LOCO (confirms MoCoDAD c4) | split-(a) reproduction for genuine 5.1 | Shopformer reimpl |
E2/E3 | MoCoDAD operating-point FPR gap (c3).

---

## E3 — Efficiency (2026-08-29)   [DONE]

Profiler: `harness/eff_profile.py` + per-model `eff_{stgnf,coskad,mocodad}.py`, each run IN THAT
MODEL'S pinned env. CPU, `torch.set_num_threads(1)`, batch 1, median of 200 forwards (20 warmup);
MoCoDAD measured via its real `test_step` generative path (median of 4-8). Peak RSS = ru_maxrss.
FLOPs not measured -- no profiler in the pinned envs and installing one upgraded torch (see note);
params + CPU latency are the deployability signal.  -> results/stats/efficiency.csv

| model | params (trainable) | CPU ms / window | windows/s | peak RSS |
|---|---|---|---|---|
| STG-NF (normalizing flow) | 616 | 4.88 | 205 | 327 MB |
| COSKAD (STSE encoder) | 239,716 | 2.15 | 465 | 290 MB |
| MoCoDAD (diffusion, n_gen=50) | 142,294 | **2595.8** | **0.39** | 722 MB |
| MoCoDAD (n_gen=1) | | 55.3 | 18 | |

Findings:
- **MoCoDAD's paper-default generative inference is ~2.6 s / window on CPU (0.39 win/s)** --
  ~1200x slower than COSKAD, ~530x slower than STG-NF. Non-real-time on CPU; even n_gen=1 (55 ms)
  is 18 win/s. Its frame-ROC-AUC lead comes at ~1000x the inference cost.
- **STG-NF is the smallest model (616 params) but NOT the fastest** -- its 8 sequential flow steps
  make it 2x slower per window than COSKAD's single encoder forward.
- **COSKAD is fastest (465 win/s) despite being the largest** (240K params, one forward).
- Pareto (figures/efficiency_pareto.pdf): MoCoDAD = accuracy at ~1000x cost; STG-NF & COSKAD =
  real-time-capable, tied at ~0.60-0.66 ROC-AUC.

NOTE / incident: `pip install ptflops` into the `coskad` env pulled torch 1.11 -> 2.8. Caught
BEFORE E1 COSKAD started; env restored to `torch==1.11.0+cu113` (ptflops left installed but unused/
incompatible). The 4-seed COSKAD pass ran earlier on torch 1.11 -- unaffected. COSKAD efficiency
row re-measured on the restored 1.11 env.

## E1 — Leave-one-camera-out (2026-08-29 → complete 2026-08-31 06:10)   [DONE]
6 folds (hold out camera k: train normal on the other 5, test = all of camera k), 4 seeds, no val
(threshold-free metrics: ROC-AUC / PR-AUC / EER / event-ROC-AUC). Splits: `harness/make_loco_splits.py`
-> `results/splits/loco_f{1..6}_{train,test}.json`, track-leak asserted 0 per fold.
Fold test sizes: f1 42 vids (10 anom) ... f6 8 vids (1 anom -- widest CI, flagged).
Driver: `/tmp/e1_driver.sh` (STG-NF -> COSKAD -> MoCoDAD). `fold` threaded through score_run/run_eval;
rows tagged `fold=f{k}` in metrics.csv. Ingest: `harness/loco_ingest.py`.
Ingest: `harness/loco_ingest.py` (no val → `score_run.score_all(model, {}, pv, fold=f'f{k}')`).
Coverage: 3 models × 6 folds × 4 seeds = 72 `frame_roc_auc` rows in metrics.csv, all present.
`assemble_results.loco()` → `figures/loco_heatmap.pdf` + `stats/loco_matrix.csv`.
`stats/verdicts.csv` updated: cols `c4_all_LOCO_folds_positive`, `LOCO_mean_dAUC`.

### Frame ROC-AUC, 4-seed mean per held-out camera (σ=0)  → stats/loco_matrix.csv
| model | C1 | C2 | C3 | C4 | C5 | C6 | LOCO mean | folds<0.5 |
|---|--|--|--|--|--|--|--|--|
| STG-NF  | 0.624 | 0.594 | 0.680 | 0.663 | 0.571 | 0.679 | 0.635 | 0 |
| COSKAD  | 0.574 | 0.497 | 0.705 | 0.498 | 0.618 | 0.833 | 0.621 | 2 |
| MoCoDAD | 0.699 | 0.786 | 0.717 | 0.672 | 0.852 | 0.872 | 0.767 | 0 |

### Pre-registered criterion 4 (direction consistent across all 4 seeds AND all 6 LOCO folds)
- **MoCoDAD − STG-NF per fold**: +0.075, +0.192, +0.037, +0.009, +0.282, +0.193
  → all 6 folds positive **TRUE** (mean +0.131). Combined with the 4-seed check (§5.2, all 4
  seeds positive) → **criterion 4 MET**. MoCoDAD verdict stays "OUTPERFORMS STG-NF (criteria
  1,2,4 met; criterion 3 FPR-generalisation flagged)".
- **COSKAD − STG-NF per fold**: −0.050, −0.097, +0.025, −0.165, +0.047, +0.153 → not consistent.
  COSKAD collapses to ≤ chance on held-out C2 (0.497) and C4 (0.498) → **not camera-robust**.
  Verdict "comparable / within noise (LOCO: fails on held-out C2,C4 → below chance)".
- **STG-NF**: most fold-consistent (range 0.571–0.680) but per-seed spread within a fold still
  huge (C6 seeds 0.47–0.94; C5 0.41–0.80) — same instability as §5.2, now across cameras too.
- All three models' LOCO means sit below their in-domain split-(b) numbers → every model loses
  accuracy under camera shift; MoCoDAD loses the least.

### GPU-hours (E1)
STG-NF 6 folds × 4 seeds ≈ 0.3 GPU-h · COSKAD 6×4 (4 seeds parallel on GPUs 0–3, wall ~4×25 min)
≈ 4.5 GPU-h · MoCoDAD 6×4 (parallel) ≈ 6.0 GPU-h · LOCO evals ≈ 1.2  =  **~12 GPU-h**.
Session running total (Stage 1 + 1b + 4-seed + E3 + E1) ≈ **18.5 GPU-h**.

### Chapter 5
§5.4 filled in results.md (LOCO table, per-fold Δ, criterion-4 result, COSKAD C2/C4 finding).
§5.6 filled from E3. §5.5 (E2 error attribution) and §5.7 (qualitative) NOT run — not requested.

--- (was, mid-run:)
STG-NF folds done so far (4-seed mean): f1 0.624, f2 0.594, f3 0.680 (per-seed spread still ~0.3).

---

## E2-noise — eval-time keypoint-degradation robustness (2026-09-01)   [DONE]

Scope: the cheap, no-retrain arm of E2 (preregistration §7 "E2 keypoint noise").
Degrade the split-(b) TEST poses only; re-score the existing 4-seed checkpoints; train + val
stay clean. Threshold-free frame ROC-AUC primary (same protocol as E1). **Exploratory** (§8).

Harness (all new, `harness/`):
- `perturb.py` — 9 pre-registered conditions, deterministic RNG `default_rng([20260901, cond_idx, seed])`,
  self-test passes. gaussN = +N(0,N px) on (x,y); jdropP = per-(frame,joint) hard drop; fdropP =
  per-(track,frame) whole-person drop. Noise applied in RAW PIXEL space (literal "{1,2,4,8}px" spec);
  each repo's native normalisation runs downstream unchanged.
- `io_poselift._find_pkl` patched: `POSELIFT_PKL_OVERRIDE` env → perturbed {video}.pkl for TEST
  videos only (train/val fall through to real pickles → unaffected even with env set).
- `e2_noise_build.py` (perturbed pkls + perturbed converted tree per cond,seed),
  `noise_ingest.py` (variant=<cond>, no val, like loco_ingest),
  `e2_noise_{stgnf,coskad,mocodad}.sh` + `stage_mocodad_test.py` (checkpoint-only eval vs the
  4-seed checkpoints; per-seed data roots so 4 seeds run parallel on GPUs 0-3),
  `e2_noise_driver.sh` (9 conds × 4 seeds × 3 models).
- `assemble_results.e2noise()` → `figures/e2noise_curves.pdf` + `stats/e2noise_matrix.csv`.

Run: 2026-09-01 15:12 → 16:27, ~8 min/condition, **0 failed runs, 27/27 (model×cond) cells x4 seeds**.
metrics.csv deduped 18270 → 18000 (backup `raw/metrics.csv.bak_pre_e2dedupe`).

### Frame ROC-AUC, 4-seed mean (σ=0), Δ vs clean   → stats/e2noise_matrix.csv
| condition | STG-NF | COSKAD | MoCoDAD |
|---|--|--|--|
| clean          | 0.660 | 0.599 | 0.736 |
| Gaussian 1–8px | 0.659–0.665 (flat) | 0.597–0.599 (flat) | 0.725–0.741 (flat) |
| joint-drop 5%  | 0.663 | 0.558 (−0.041) | 0.738 |
| joint-drop 10% | 0.672 | 0.552 (−0.047) | 0.736 |
| joint-drop 20% | 0.651 | 0.556 (−0.043) | 0.737 |
| frame-drop 5%  | 0.666 | 0.525 (−0.074) | 0.731 |
| frame-drop 10% | 0.695 | **0.503 (−0.096, = chance)** | 0.726 |

### Findings
- **Gaussian jitter (≤8px): no effect on any model** (|Δ| ≤ 0.012). Windowed skeleton-AD absorbs
  sub-bbox coordinate error.
- **COSKAD is the only model that degrades — and only under MISSING keypoints.** joint-drop:
  stable −0.04/−0.05. frame-drop 10%: ROC-AUC 0.503, EER 0.497, all 4 seeds 0.495–0.513. Its graph
  encoder + `markovitz` norm needs a full skeleton and a present person every frame; harness
  carry-forward gap-fill does not rescue it.
- **MoCoDAD most robust on every condition** (max |Δ| 0.012) — most stable model under seed (§5.2),
  camera (§5.4) AND input degradation.
- **STG-NF frame-drop "+0.034" is seed noise** — per-seed spread 0.594–0.802 on fdrop10, same
  instability as §5.2. Its whole row = flat within its own ±0.07 seed band.
- PR-AUC / EER agree: COSKAD PR-AUC 0.097 → 0.069 by fdrop10; STG-NF/MoCoDAD hold.

### GPU-hours
STG-NF 36 evals ≈ 0.25 · COSKAD 36 evals (4|| ) ≈ 0.6 · MoCoDAD 36 evals @ n_gen=50 (4|| ) ≈ 3.6
= **~4.5 GPU-h**. No training. Session running total ≈ **23 GPU-h**.

### Chapter 5
§5.5 filled in results.md (table, 4 findings, deployment implication, `figures/e2noise_curves.pdf`).
Still open: E2-norm (normalisation-scheme ablation, retrain) · E2-window (seg-len ablation, retrain)
· split-(a) §5.1 · Shopformer reimpl · §5.7 qualitative.

---

## E2-window + E2-norm — retraining arms (2026-09-02)   [DONE, partial]

Harness (all new, `harness/`): `ingest_raw_variant.py`, `ingest_stgnf_variant.py` (variant-tagged
ingest, with val pass for fixed-FPR calibration), `e2_window_{stgnf,coskad}.sh`,
`e2_norm_{stgnf,coskad,mocodad}.sh`, `stage_mocodad_json.py` (COCO-JSON staging for MoCoDAD's
non-robust normalisation paths), `e2_retrain_driver.sh`, `e2_norm_recover.sh`.
Patch: `MoCoDAD/utils/dataset_utils.py` `np.int` → `int` (removed in NumPy ≥1.24; only the
non-`robust` dataset path hits it) — `results/env/patches/MoCoDAD_npint_dataset_utils.patch`.

### E2-window (STG-NF + COSKAD; MoCoDAD N/A)
Retrain at T ∈ {12,32,48}, stride T/2, 4 seeds. MoCoDAD excluded: native seg_len 6 is
architecturally load-bearing (diffusion conditions on first half, predicts second;
`conditioning_indices:[0,1,2]`) — a length sweep is not a clean ablation (pre-reg §7
"for models that allow it").  → `stats/e2window_matrix.csv`, `figures/e2window_curves.pdf`

| T (stride T/2) | STG-NF | COSKAD |
|---|--|--|
| 12 | 0.673 | 0.599 |
| 24 (main run) | 0.660 | 0.599 |
| 32 | 0.518 | 0.572 |
| 48 | 0.436 | 0.535 |

- **STG-NF collapses for T ≥ 32** (0.518 → 0.436 = below chance). 3-epoch flow training can't fit
  the longer sequences; NLL degrades, score inverts. T=12 ≈ T=24 within its seed band.
- **COSKAD degrades monotonically** (−0.003/−0.027/−0.064), per-seed spread ±0.002 → real trend.
- No model benefits from a longer window → the T≈12–24 windows used in §5.2 are near-optimal.

### E2-norm — Amendment 2 (each model's own normalisation menu, config-only)
→ `stats/e2norm_matrix.csv`

| model | default | alternative(s) | Δ |
|---|--|--|--|
| STG-NF  | normalised 0.660 | raw/unnormalised 0.634 | −0.027 |
| COSKAD  | markovitz 0.599  | stan 0.584            | −0.015 |
| MoCoDAD | robust 0.736     | markovitz 0.670 · stan 0.713 | −0.066 · −0.023 |

- Every model's shipped default is at/near its best; alternatives cost 0.015–0.066 → normalisation
  is a second-order factor vs window length or seed. STG-NF ~agnostic; MoCoDAD most sensitive.

### FAILURES (rows stripped from metrics.csv — never kept as numbers)
- **First pass**: 1800 rows @ constant 0.5 — COSKAD {robust,bbox}, MoCoDAD {markovitz,stan,bbox}.
  Causes: MoCoDAD non-robust reads COCO-JSON not CSV trajectories (FileNotFoundError) + `np.int`;
  COSKAD robust reads trajectory CSVs (FileNotFoundError); COSKAD bbox degenerate.
  Backup `raw/metrics.csv.bak_pre_e2strip`.
- **Recovery** (`e2_norm_recover.sh`, after JSON staging + np.int patch + trajectory staging):
  - MoCoDAD markovitz 0.670 ✓, stan 0.713 ✓  — recovered.
  - **MoCoDAD bbox 0.9340 on ALL 4 seeds, zero variance** → NOT a real result. `normalize_pose_bbox`
    divides by per-segment kp width/height ≈ 0 for near-stationary retail poses → inputs collapse;
    the stable high AUC is a clip-length/coverage artifact, not detection. Row stripped
    (720 rows), backup `raw/metrics.csv.bak_pre_e2strip` (2nd strip in place).
  - **COSKAD robust — NOT reproduced.** 2nd attempt: `ValueError: not enough values to unpack`
    in `aggregate_rnn_autoencoder_data` — needs MoCoDAD-style dash-named (`s-c`) trajectory
    folders + a fitted scaler. Time-boxed out (pre-reg §9). COSKAD comparison = markovitz vs stan.
  - **COSKAD bbox — NOT retried.** Same division-by-≈0 as MoCoDAD bbox; reported as a degenerate
    normalisation, not a number.

### GPU-hours
E2-window ≈ 4 (STG-NF trivial + COSKAD 12 trainings) · E2-norm 1st pass ≈ 3 (partly wasted on
failed runs) · recovery ≈ 4 (MoCoDAD 3×4 generative eval dominates) = **~11 GPU-h**.
Session running total ≈ **34 GPU-h**.

### Chapter 5
§5.5 restructured into 5.5a (noise) / 5.5b (window) / 5.5c (norm, Amendment 2). Header + Open/failed
updated. Amendment 2 added to preregistration.md. E4 data-scale, split-(a) §5.1, §5.7 qualitative,
Shopformer reimpl, COSKAD-robust-norm reproduction remain not run.

---

## Fix (2026-09-03) — aggregated/summary.csv `main` rows were fold-contaminated

`assemble_results.summary()` did not filter by `fold`, so the `main`-variant aggregate mixed in
the 24 E1/LOCO fold rows → e.g. STG-NF `main` σ0 showed 0.640 (base_rate 0.027) instead of the
correct 0.660 (0.067). Fix: skip `fold != 'none'` rows in `summary()`. Regenerated
`aggregated/summary.csv`. **No authoritative number changed** — `stats/perseed.csv`,
`stats/verdicts.csv`, `results.md` §5.2 read the npz score arrays directly with the correct
filter and were always right; only the summary.csv convenience table was affected.
Post-fix: STG-NF 0.6604±0.078 / COSKAD 0.5990±0.003 / MoCoDAD 0.7364±0.012 (= perseed.csv).
