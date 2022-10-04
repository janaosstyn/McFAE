# McFAE (Molecular Complex Feature Attribution Extraction)

## Setup environment

All required packages can be installed in a conda environment. `conda` and `git` need to be installed on your machine.
Installation can be done with

    conda env create -f environment.yml

The McFAE environment can be activated with

    conda activate McFAE

Run following script to unzip the data. This requires `unzip` to be installed, the zips at `data/pdb/pdb.zip`
and `data/tcr3d_images/imrex_input_images.zip` can also be extracted manually.

    bash src/scripts/unzip_data.sh

## ImRex model

A pretrained ImRex model is available in `ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv`
but it can be retrained by running following command **from the root directory**.

    bash src/scripts/retrain_ImRex/retrain_attributions_model.sh

Following command was used to create the ImRex data (without the samples also present in the molecular complex data)

    python src/scripts/retrain_ImRex/remove_pdb_data.py

## TITAN models

The 3 pretrained TITAN models are available in `TITAN/models`. `titanData_strictsplit_nocdr3` is the default
model, `nocdr3dup_epgrouped5cv_paperparams_smallpad` is the model trained on ImRex data
and `titanData_strictsplit_scrambledtcrs` is the model trained on scrambled tcr data.

They can be retrained by running following command **from the root directory**. Note that this will overwrite the
pretrained models.

    bash src/scripts/retrain_TITAN/retrain_strictsplit_model.sh
    bash src/scripts/retrain_TITAN/retrain_titan_on_imrex_data_model.sh
    bash src/scripts/retrain_TITAN/retrain_scrambled_tcrs_model.sh

The data for the 3 models was created with following commands:

    python src/scripts/retrain_TITAN/remove_pdb_data.py
    python src/scripts/retrain_TITAN/imrex_to_titan_data.py
    python src/scripts/retrain_TITAN/create_scrambled_tcrs.py

## Feature attribution extraction

The data for this step is already present but can be recreated with:

    python src/scripts/TCR3D_to_model_data.py

All feature attributions, distance matrices, correlation and intermediary results are saved in  `/data`
folder where a subfolder is made for each model configuration. All results can be reproduced by running:

    python src/imrex_attributions.py
    python src/titan_attributions.py

This will produce all results that are not yet present in the `/data/{configuration_name}` folder and save them in the
appropriate place. Always run `imrex_attributions.py` first, this also calculates shared results like molecular complex
distance and random correlation.

Recreating the PyMol scripts to show feature attributions on the molecular complex can be done with

    python src/scripts/3d_highlighter.py

These are not included because they require the absolute path to the PDB files.

All plots were made with `src/scripts/plot.py`, the table with statistics about the TITAN data was made
with `src/scripts/retrain_TITAN/inspect_titan_data.py`.
