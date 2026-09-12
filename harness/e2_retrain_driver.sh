#!/bin/bash
# E2 retraining arms: E2-window (STG-NF + COSKAD, T in {12,32,48} @ stride T/2) and
# E2-norm (STG-NF norm_raw; COSKAD {robust,stan,bbox}; MoCoDAD {markovitz,stan,bbox}).
# MoCoDAD E2-window is NOT run: native seg_len 6 is architecturally load-bearing (diffusion
# conditions on the first half, predicts the second) so a length sweep is not a clean ablation.
# Sequential (all three repos share hardcoded / single staging roots). ~12-16 GPU-h.
PS=~/poselift-study; H=$PS/harness
LOG=$PS/results/model_runs/e2_retrain_driver.log
echo "=== E2-RETRAIN START $(date) ===" | tee -a $LOG

echo "##### E2-WINDOW #####" | tee -a $LOG
for TSPEC in "12 6" "32 16" "48 24"; do
  set -- $TSPEC; T=$1; ST=$2
  echo "--- E2-window T=$T stride=$ST  $(date +%H:%M) ---" | tee -a $LOG
  bash $H/e2_window_stgnf.sh  $T $ST 0 >>$LOG 2>&1 || echo "!! e2_window_stgnf T$T FAILED" | tee -a $LOG
  bash $H/e2_window_coskad.sh $T $ST     >>$LOG 2>&1 || echo "!! e2_window_coskad T$T FAILED" | tee -a $LOG
done

echo "##### E2-NORM #####" | tee -a $LOG
echo "--- E2-norm STG-NF norm_raw  $(date +%H:%M) ---" | tee -a $LOG
bash $H/e2_norm_stgnf.sh 0 >>$LOG 2>&1 || echo "!! e2_norm_stgnf FAILED" | tee -a $LOG
for STRAT in robust stan bbox; do
  echo "--- E2-norm COSKAD $STRAT  $(date +%H:%M) ---" | tee -a $LOG
  bash $H/e2_norm_coskad.sh $STRAT >>$LOG 2>&1 || echo "!! e2_norm_coskad $STRAT FAILED" | tee -a $LOG
done
for STRAT in markovitz stan bbox; do
  echo "--- E2-norm MoCoDAD $STRAT  $(date +%H:%M) ---" | tee -a $LOG
  bash $H/e2_norm_mocodad.sh $STRAT >>$LOG 2>&1 || echo "!! e2_norm_mocodad $STRAT FAILED" | tee -a $LOG
done

echo "=== E2-RETRAIN COMPLETE $(date) ===" | tee -a $LOG
