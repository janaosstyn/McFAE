#! /bin/bash
set -o errexit

NAME=nocdr3dup_epgrouped5cv_paperparams_smallpad
DATAFOLDER=TITAN/data/imrex_data/nocdr3dup_default_epgrouped5cv
TCRFILE=TITAN/data/imrex_data/tcrs.csv
EPFILE=TITAN/data/imrex_data/epitopes.csv
SAVEFOLDER=TITAN/models/$NAME
PARAMS=TITAN/params/params_training_AACDR3_paper.json

mkdir -p $SAVEFOLDER

for CV in {0..4}; do
  mkdir -p $SAVEFOLDER/cv$CV
  python TITAN/scripts/flexible_training.py $DATAFOLDER/train$CV.csv $DATAFOLDER/test$CV.csv \
    $TCRFILE $EPFILE $SAVEFOLDER/cv$CV/ $PARAMS "${NAME}_${CV}" bimodal_mca
done