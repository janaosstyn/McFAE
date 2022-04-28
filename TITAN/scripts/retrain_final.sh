NAME=nocdr3dup_epgroupedfinal_paperparams_smallpad
SAVEFOLDER=models/$NAME
PARAMS=params_training_AACDR3_paper.json

mkdir $SAVEFOLDER

python scripts/flexible_training.py data/imrex_data/final_train.csv data/imrex_data/test0.csv data/imrex_data/tcrs.csv \
 data/imrex_data/epitopes.csv $SAVEFOLDER/ params/$PARAMS $NAME bimodal_mca

