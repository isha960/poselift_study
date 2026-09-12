#!/bin/bash
# E2-norm recovery: reruns the arms that failed on first pass, after fixes:
#  - COSKAD 'robust' : needed trajectory-CSV staging (now in data_coskad/b/*/trajectories)
#  - MoCoDAD non-'robust' : needed COCO-JSON staging (data_mocodad/b/pose/) + np.int patch
# COSKAD 'bbox' is NOT retried: normalize_pose_bbox divides by per-segment w/h -> ~0 for
# low-motion retail poses -> degenerate constant score (documented, not a result).
PS=~/poselift-study; H=$PS/harness
LOG=$PS/results/model_runs/e2_norm_recover.log
echo "=== E2-NORM RECOVER START $(date) ===" | tee -a $LOG

echo "--- COSKAD norm_robust $(date +%H:%M) ---" | tee -a $LOG
bash $H/e2_norm_coskad.sh robust >>$LOG 2>&1 || echo "!! coskad robust FAILED" | tee -a $LOG

for STRAT in markovitz stan bbox; do
  echo "--- MoCoDAD norm_$STRAT $(date +%H:%M) ---" | tee -a $LOG
  bash $H/e2_norm_mocodad.sh $STRAT >>$LOG 2>&1 || echo "!! mocodad $STRAT FAILED" | tee -a $LOG
done
echo "=== E2-NORM RECOVER COMPLETE $(date) ===" | tee -a $LOG
