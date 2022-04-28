NAME=titanData_strictsplit_imrexprep_23l_10cv
DATAFOLDER=TITAN/data/imrex_data/titanData_strictsplit_imrexprep_23l
TCRFILE=TITAN/data/tcr.csv
EPFILE=TITAN/data/epitopes.csv
SAVEFOLDER=TITAN/models/$NAME
PARAMS=TITAN/params/params_training_AACDR3_paper.json

mkdir -p $SAVEFOLDER

for CV in {0..9}; do
  mkdir -p $SAVEFOLDER/cv$CV
  python TITAN/scripts/flexible_training.py $DATAFOLDER/train$CV.csv $DATAFOLDER/test$CV.csv $TCRFILE $EPFILE \
    $SAVEFOLDER/cv$CV/ $PARAMS "${NAME}_${CV}" bimodal_mca 2>&1 | tee -a $SAVEFOLDER/cv$CV.log
done
