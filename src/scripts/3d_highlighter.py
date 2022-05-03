import os

import matplotlib as mpl
import pandas as pd
from Bio.PDB import PDBParser
from colour import Color
from matplotlib import pyplot as plt

from src.imrex_attributions import ImrexAttributionsHandler
from src.titan_attributions import TITANAttributionsHandler
from src.util import pdb2fasta_mapper, get_cmap

PDB_FOLDER = 'data/pdb'


def write_script(pdb_id, cdr3_highlight, ep_highlight, out_path):
    """
    Write a single PyMol script to highlight feature attributions on the PDB complex

    Parameters
    ----------
    pdb_id              PDB ID
    cdr3_highlight      Feature attributions from the CDR3 to highlight
    ep_highlight        Feature attributions from the epitope to highlight
    out_path            Path to save the script
    """

    # Get the CDR3 and epitope sequences
    complex_data = pd.read_csv(f'data/complex_data_original.csv', index_col=0).loc[pdb_id]
    ep_chain = complex_data['epitope_chain']
    tcrb_chain = complex_data['tcrb_chain']

    # Use the PDB file to get a list of all chains and hide those that are not TCRB or epitope
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, f'{PDB_FOLDER}/{pdb_id.lower()}.pdb')[0]
    all_chains = [c.id for c in structure.get_chains()]
    hide_chains = [c for c in all_chains if c != ep_chain and c != tcrb_chain]

    # Color scale for highlighting
    color_scale = list(
        Color('lime').range_to(Color('blue'), 101))  # Color scale from Green to Blue, index from 0 to 100

    # Convert attributions to PyMol colors
    cdr3_colors = {k: f'0x{color_scale[round(v * 100)].get_hex_l()[1:]}' for k, v in cdr3_highlight.items()}
    ep_colors = {k: f'0x{color_scale[round(v * 100)].get_hex_l()[1:]}' for k, v in ep_highlight.items()}

    pdb_mapper = {v: k for k, v in
                  pdb2fasta_mapper(f'{PDB_FOLDER}/{pdb_id.lower()}.pdb', pdb_id, 0, tcrb_chain,
                                   complex_data['tcrb_seq']).items()}
    fasta_offset = complex_data['CDR3_start']

    # Start writing the script
    script_file = open(f'{out_path}/{pdb_id}.pml', 'w')
    # Make sure PyMol loads the PDB file
    script_file.write(f'load {os.getcwd()}/{PDB_FOLDER}/{pdb_id.lower()}.pdb\n')
    # Set the background color white
    script_file.write('bg_color white\n')
    # Hide all obsolete chains
    for hc in hide_chains:
        script_file.write(f'hide everything, chain {hc}\n')

    # Color the full TCRB and epitope chain
    script_file.write(f'color purple, chain {ep_chain}\n')
    script_file.write(f'color grey80, chain {tcrb_chain}\n')

    # Update colors of CDR3 region
    for i, color in cdr3_colors.items():
        script_file.write(
            f'color {color}, chain {tcrb_chain} and resi {pdb_mapper[fasta_offset + i]}\n')

    # Update colors of epitope
    for i, color in ep_colors.items():
        script_file.write(f'color {color}, chain {ep_chain} and resi {i + 1}\n')

    # Zoom and orient to CDR3 and epitope region
    selection = f'chain {ep_chain} or (chain {tcrb_chain} and ' \
                f'resi {pdb_mapper[complex_data["CDR3_start"]]}-{pdb_mapper[complex_data["CDR3_end"]]})'
    script_file.write(f'zoom {selection}\n')
    script_file.write(f'orient {selection}\n')
    script_file.close()


def write_all_attribution_highlight_scripts(modelhandler, method='SHAP BGdist'):
    """
    Write a PyMol script to show feature attributions on the PDB complex for all PDB samples

    Parameters
    ----------
    modelhandler    Model to use for feature attribution extraction
    method          Feature attribution extraction method
    """

    # Plot the colorbar
    fig = plt.figure(figsize=(6.4 / 1.3, 4.8 / 1.3))
    ax = fig.add_axes([0.05, 0.05, 0.07, 0.9])

    cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical', cmap=get_cmap())
    cb.set_label(f"{method} feature attribution")
    plt.savefig('output/plots/colorbar.png', bbox_inches='tight', dpi=300)

    # Get the AA attributions from the modelhandler
    aa_norm_attributions = modelhandler.get_aa_norm_attributions()
    sequences = modelhandler.get_sequences()
    for pdb_id, methods in aa_norm_attributions.items():
        attributions = methods[method]
        ep_seq, cdr3_seq = sequences[pdb_id]
        # Split attributions in epitope and CDR3
        ep_highlight = dict(enumerate(attributions[:len(ep_seq)]))
        cdr3_highlight = dict(enumerate(attributions[len(ep_seq):]))
        # Create the save folder
        save_folder = f'{modelhandler.save_folder}/{modelhandler.name}/pymol_scripts'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        write_script(pdb_id, cdr3_highlight, ep_highlight, save_folder)


def main():
    imrex_attribution_handler = ImrexAttributionsHandler(
        name="imrex_nocdr3dup",
        display_name="ImRex",
        model_path="ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/iteration_2/"
                   "2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv-epoch20.h5",
        image_path="data/tcr3d_images/",
        save_folder="data"
    )

    titan_strictsplit_handler = TITANAttributionsHandler(
        name="titan_strictsplit",
        display_name="TITAN",
        model_path="TITAN/models/titanData_strictsplit_nocdr3/cv5/titanData_strictsplit_nocdr3_5/",
        tcrs_path='data/tcrs.csv',
        epitopes_path='data/epitopes.csv',
        data_path='data/tcr3d_titan_input.csv',
        save_folder='data'
    )

    titan_on_imrex_data_handler = TITANAttributionsHandler(
        name="titan_nocdr3dup",
        display_name="TITAN on ImRex data",
        model_path="TITAN/models/nocdr3dup_epgrouped5cv_paperparams_smallpad/cv2/"
                   "nocdr3dup_epgrouped5cv_paperparams_smallpad_2/",
        tcrs_path='data/tcrs.csv',
        epitopes_path='data/epitopes.csv',
        data_path='data/tcr3d_titan_input.csv',
        save_folder='data'
    )
    write_all_attribution_highlight_scripts(imrex_attribution_handler)
    write_all_attribution_highlight_scripts(titan_strictsplit_handler)
    write_all_attribution_highlight_scripts(titan_on_imrex_data_handler)


if __name__ == "__main__":
    main()
