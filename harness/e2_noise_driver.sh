#!/bin/bash
# E2-noise full sweep: 9 pre-registered conditions x 4 seeds x {STG-NF, COSKAD, MoCoDAD}.
# Checkpoint-only eval on perturbed split-(b) TEST sets (no retrain). Rows -> raw/metrics.csv
# with variant=<cond>. Usage: e2_noise_driver.sh [cond ...]   (default: all 9)
PS=~/poselift-study; H=$PS/harness
DATA=$PS/data_e2noise
LOG=$PS/results/model_runs/e2_noise_driver.log
CONDS="${*:-gauss1 gauss2 gauss4 gauss8 jdrop05 jdrop10 jdrop20 fdrop05 fdrop10}"
mkdir -p $PS/results/model_runs/stgnf/e2noise $PS/results/model_runs/coskad/e2noise $PS/results/model_runs/mocodad/e2noise
echo "=== E2-NOISE START $(date) :: $CONDS ===" | tee -a $LOG

for COND in $CONDS; do
  echo "===== $COND  build $(date +%H:%M) =====" | tee -a $LOG
  for S in 0 1 2 3; do
    [ -d $DATA/${COND}_s${S}/pkls ] || \
      ~/.conda/envs/poselift-harness/bin/python $H/e2_noise_build.py $COND $S $DATA/${COND}_s${S} >>$LOG 2>&1
  done

  echo "  STG-NF (serial, gpu0) $(date +%H:%M)" | tee -a $LOG
  for S in 0 1 2 3; do
    bash $H/e2_noise_stgnf.sh $COND $S 0 $DATA/${COND}_s${S}/pkls >>$LOG 2>&1 \
      || echo "  !! STG-NF $COND s$S FAILED" | tee -a $LOG
  done

  echo "  COSKAD (4 seeds || gpu0-3) $(date +%H:%M)" | tee -a $LOG
  for S in 0 1 2 3; do
    nohup bash $H/e2_noise_coskad.sh $COND $S $S $DATA/${COND}_s${S}/pkls $PS/data_coskad/b_e2n_s${S} >>$LOG 2>&1 &
  done
  wait

  echo "  MoCoDAD (4 seeds || gpu0-3) $(date +%H:%M)" | tee -a $LOG
  for S in 0 1 2 3; do
    nohup bash $H/e2_noise_mocodad.sh $COND $S $S $DATA/${COND}_s${S}/conv $PS/data_mocodad/b_e2n_s${S} >>$LOG 2>&1 &
  done
  wait

  rm -rf $DATA/${COND}_s0 $DATA/${COND}_s1 $DATA/${COND}_s2 $DATA/${COND}_s3
  echo "  === $COND DONE $(date +%H:%M) ===" | tee -a $LOG
done
echo "=== E2-NOISE COMPLETE $(date) ===" | tee -a $LOG
