# Pre-registration — PoseLift skeleton-AD comparative study

**Committed:** 2026-08-28, before any test-set scoring.
**Status of this document:** frozen on commit. Any change after the first test score
is logged as a dated amendment at the bottom, never an edit above this line.

---

## 0. Scope of the CURRENT pass (explicitly limited)

This pass = **one training run per model on split (b), no experiments.**
- Seeds: **seed 0 only** for every model. Every metric it produces is labelled
  **"preliminary (n=1)"** in `results.md` and carries no confidence interval or
  significance test that depends on seed repetition.
- Experiments **E1 (LOCO), E2 (representation/robustness), E3 (efficiency), E4**
  are **deferred** to a later, separately-approved pass.
- Models this pass: STG-NF, COSKAD (Euclidean), MoCoDAD, motion-energy baseline.
  **Shopformer** = to be **re-implemented from the paper + partial git-history code**;
  every Shopformer number is labelled **"reimplementation — NOT official code"**
  and is never compared like-for-like against the paper's 69.15. It may land after
  the other four.

The rest of this document pre-registers the FULL study so later passes inherit
frozen choices.

---

## 1. Primary metric & comparison

- **Primary metric:** frame-level ROC-AUC (micro-averaged over all test frames,
  scores concatenated across videos).
- **Secondary reported:** frame PR-AUC (always quoted with the test base rate),
  EER, event-level ROC-AUC, recall @ fixed FPR ∈ {0.01, 0.05, 0.10}, median
  detection latency at each fixed FPR.
- **Accuracy is never reported.**
- **Primary comparison:** each contemporary method {COSKAD, MoCoDAD, Shopformer-reimpl,
  motion-energy} **vs STG-NF** (the baseline).
- **"Meaningful" margin:** ΔROC-AUC ≥ **0.03**.

## 2. "Outperforms STG-NF" verdict (full study only)

A method is said to outperform STG-NF only if **ALL** hold:
1. mean ΔROC-AUC ≥ 0.03 (over seeds);
2. Holm–Bonferroni-corrected **DeLong** test AND cluster-**bootstrap** 95% CI of
   the AUC difference both exclude 0;
3. the gain is **not** driven by a large FPR rise at the fixed operating point;
4. direction consistent across **all 4 seeds** AND across **all 6 LOCO folds**.
Anything less → **"comparable / within noise"**. Verdict recorded per model in
`stats/verdicts.csv`.

## 3. Seeds

- Full study: seeds **{0, 1, 2, 3}** (4). (Reduced from the methodology's 10/5
  by user decision on compute; recorded as a deviation.)
- Current pass: **{0}** only.
- All seeded: Python `random`, NumPy, torch (+ `torch.cuda`), and
  `pytorch_lightning.seed_everything` where applicable. cuDNN deterministic where
  the model allows it; non-determinism that remains is noted per model.

## 4. Shared representation & sequence protocol (frozen)

- **Keypoints:** HRNet 2D COCO-17 as delivered in the PoseLift `.pkl`
  (`x, y, conf`). Models needing 18 joints get the neck = mid-shoulder insertion
  via the repo's own converter; logged in Table 3.1.
- **Normalisation (primary scheme):** translate so **mid-hip** (joints 11,12) is
  the origin; if either hip has `conf < tau`, fall back to **mid-shoulder**
  (joints 5,6); scale by **torso length** = ‖mid-shoulder − mid-hip‖ computed
  per window; if that is 0 / undefined, fall back to **bbox diagonal**. Keep
  `conf` as a 3rd channel.
- **Low-confidence handling:** joints with `conf < tau`, **tau = 0.0** for the
  primary runs (PoseLift confs are already sparse; masking at >0 removes too much
  — recorded choice). Missing/zero joints are linearly interpolated **within a
  single person track only**, never across tracks or across the split boundary.
- **Window:** **T = 24 frames** (~1.6 s @ 15 fps), **stride 12**, per person.
  Models with a hard-coded window (MoCoDAD native 6+? ; COSKAD native 12) use
  their **native** window; stride = T/2, normalisation, and aggregation stay
  identical; the deviation is logged in Table 3.1 and `results.md`.
- **Score orientation:** every model's native anomaly score, oriented
  **higher = more anomalous**, sign verified on the validation split before any
  test scoring:
  - STG-NF → negative log-likelihood
  - COSKAD (Euclidean) → distance to learned hypersphere centre
  - MoCoDAD → denoising/reconstruction error, aggregated over generated samples
    by **min** (its `aggregation_strategy='best'`)
  - Shopformer-reimpl → reconstruction error (MSE)
  - motion-energy → mean per-joint speed (primary); mean per-joint acceleration
    (secondary variant)
- **Window → frame score:** **mean** over all windows covering the frame.
- **Multi-person → frame score:** **max** over persons present in the frame.
- **Frames with no detected person:** carry forward the previous frame's score;
  count and report how many frames affected per video.
- **Temporal smoothing:** 1-D Gaussian over the per-frame score series,
  **sigma chosen ONCE on the validation split from {0, 1, 2, 4, 8} frames**,
  then frozen. **Every metric is reported BOTH with and without smoothing.**
- **Thresholding:** threshold-free metrics need none. For F1 / fixed-FPR:
  pick the threshold on **val** (max-F1 variant AND fixed-FPR variant), apply
  unchanged to test. EER reported separately, computed on test directly.
- **Event score / latency:** per anomaly video, event flagged if the max smoothed
  frame score inside the labelled anomaly window exceeds the operating-point
  threshold; detection latency = frames from labelled onset to first exceedance,
  NaN if never.

## 5. Hyper-parameter search budget (frozen, equal per model)

Each model gets an equal budget of **N = 8 validation trials** to tune only:
temporal smoothing sigma (5 values, above), and — if the model exposes it —
learning rate ∈ {default, default/3} and epochs ∈ {default, default×1.5}.
Everything else stays at the model's published defaults. The exact grid searched
per model is written to `configs/<model>_val_search.yaml` and echoed in `results.md`.
**The test set is not touched during search.**

## 6. Splits (frozen definitions; built in Stage 1)

- **Split (a) — "paper-comparable":** the PoseLift release's own `Train/` (104
  normal videos) vs `Test/` (47 videos: 41 with anomaly frames + 6 all-normal).
  Used **only** for the §5.1 reproduction. Caveat recorded: the release test set
  is full-video / un-windowed (4,846 frames, base rate 0.317), whereas the
  papers' Table-2 test set is scenario-windowed to 3,721 frames — the windowing
  script is in no public repo, so §5.1 is *release-split* comparable, not
  *paper-table* comparable, and the STG-NF 67.46 / 84.06 / 0.39 and Shopformer
  69.15 targets are cited with that caveat and with "(their n = 1)".
- **Split (b) — "pre-registered", used for ALL main results:**
  - The **41 videos containing any anomaly frame → TEST ONLY** (their normal
    frames included; no normal frames are ever mined from an anomaly video into
    train or val).
  - The **110 all-normal videos** (104 from `Train/` + 6 from `Test/`) are split
    **60 / 15 / 25 → train / val / test**, split **by video**, **stratified by
    camera** so every camera C1–C6 appears in train, val, and test.
  - **No person track may span a split boundary.** After assignment, verified in
    code: **zero track-id intersection** between train and {val, test}. Any
    residual overlap → the offending video is moved wholesale to satisfy the
    constraint, re-verified, logged.
  - Split file: `results/splits/split_b_{train,val,test}.json` — lists of video
    ids + per-video track-id lists. Treated as a published artefact.
  - Seed for the split RNG: **20260828**. Recorded so the split is regenerable.

## 7. Deferred-experiment grids (frozen now so a later pass cannot p-hack them)

- **E1 LOCO:** 6 folds, hold out one camera; train normal on the other 5, test on
  held-out camera's normal + shoplifting. Seeds {0..3}. Primary output: 6 × models
  frame-ROC-AUC matrix + mean; heatmap.
- **E2 normalisation:** schemes ∈ {mid-hip+torso (primary), mid-shoulder+bbox-diag,
  bbox min-max, none}. Requires retrain per scheme. Seeds {0..3}.
- **E2 keypoint noise (eval-time only, no retrain):** Gaussian on (x,y) at
  normalised-σ equivalents of {1, 2, 4, 8}px; random joint dropout p ∈
  {0.05, 0.1, 0.2}; whole-frame drop p ∈ {0.05, 0.1}. Seeds {0..3}.
- **E2 window T ∈ {12, 24, 32, 48}** for models that allow it; stride T/2; retrain.
  Seeds {0..3}.
- **E3 efficiency:** parameter count; FLOPs per window (fvcore/ptflops); CPU
  single-thread batch-1 latency = median of 200 timed runs after 20 warm-up →
  FPS; peak RAM via `resource`/`tracemalloc`. Reported once (no seeds).
- **E4 (optional):** normal-track subsample ∈ {25, 50, 75, 100}%, seeds {0..2}.

## 8. Statistics (full study)

Per (model, variant, metric): mean + 95% t-interval over seeds.
Confirmatory, split (b), smoothed, pre-registered — each contemporary method vs
STG-NF: DeLong (p + 95% CI of AUC diff) on pooled test scores; cluster bootstrap
(1000 resamples over **test tracks/clips**, not frames) → CI of AUC and of the
diff; Wilcoxon signed-rank on the 4 matched per-seed AUCs; effect size Cliff's
delta (+ rank-biserial). **Holm–Bonferroni** across the 4 comparisons.
LOCO / E2 / E3 results reported with CIs but labelled **exploratory**.

## 9. What counts as a failure

A model that will not train or score on PoseLift within a **1-day time-box**
(stated per model in RUNLOG) is reported as **"not reproduced"** with the exact
breakage — never replaced by a guessed number.

---

### Amendments (append-only, dated)

_(none yet)_

## Amendment 1 (2026-08-29) — smoothing sigma cannot be tuned on an anomaly-free val split

Sec.6 fixes split (b) `val` as **normal-only** (correctly: no anomaly frames may leak
into val). Sec.4 said the Gaussian smoothing sigma is chosen on val from {0,1,2,4,8}.
These conflict: with zero anomalies in val there is no anomaly-detection signal to select
sigma against, and any "minimise normal-score variance" criterion trivially picks the
largest sigma.

**Resolution (frozen before any test scoring):**
- **Primary metric is reported at sigma = 0 (no smoothing).** It needs no tuning.
- sigma in {1,2,4,8} is reported **alongside, labelled exploratory** smoothing-sensitivity,
  under the sec.4 rule "every metric reported WITH and WITHOUT smoothing".
- **Fixed-FPR thresholds are still calibrated on val** — FPR is defined on negatives only,
  so the normal-only val set is valid for that. Recall / latency at those thresholds are
  measured on test. (unchanged from sec.4)
- No "best sigma" is ever selected using test labels.

## Amendment 2 (2026-09-02) — E2-norm run over each model's native normalisation menu

Sec.7 pre-registered E2-norm as: retrain every model under the four shared-harness
normalisation schemes {mid-hip+torso, mid-shoulder+bbox-diag, bbox-minmax, none}.
On contact with the three external repos this is not tractable without high risk:
each repo (STG-NF, COSKAD, MoCoDAD) normalises **inside its own dataset pipeline**, so
forcing a shared scheme means pre-normalising the keypoints AND disabling the repo's
internal normalisation — which (a) has no config path and needs per-repo code surgery,
and (b) puts Deep-SVDD / diffusion / normalizing-flow training on out-of-dynamic-range
inputs (esp. scheme "none" = raw pixels ~0–1920), with a real chance of non-convergence
that would produce uninterpretable numbers.

**Resolution (frozen before E2-norm scoring):**
- E2-norm is run as: **each model retrained under every normalisation strategy it already
  ships**, config-only, no code surgery.
  - COSKAD menu: {markovitz (default), robust, stan, bbox}
  - MoCoDAD menu: {robust (default), markovitz, stan, bbox}
  - STG-NF: no strategy menu — only an on/off switch (`--global_pose_segs` = unnormalised);
    reported as default-vs-raw.
- This answers the same underlying question (sensitivity of each model to its normalisation
  choice) without the convergence risk.
- All E2-norm rows remain **exploratory** (sec.8), never compared against the frozen §5.2
  verdicts.
- Runs that failed to produce a converged model are reported as **failures** in RUNLOG and
  their rows removed from `metrics.csv` — never kept as a number:
  - `bbox` normalisation: degenerate on PoseLift for COSKAD (constant 0.5) and MoCoDAD
    (ROC-AUC 0.934 identical across all 4 seeds → clip-length artifact). Cause: divides by
    per-segment keypoint width/height ≈ 0 for near-stationary retail poses.
  - COSKAD `robust`: not reproduced within the sec.9 time-box (needs dash-named
    trajectory-CSV staging + fitted scaler; staging parser failed twice).
