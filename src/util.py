import datetime
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from Bio import Align
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser, Selection
from colour import Color
from matplotlib.colors import ListedColormap


def get_cmap():
    """
    Create the ListedColormap from lime to blue used to highlight feature attributions on the PDB complex

    Returns
    -------
    colormap
    """
    cmap = [c.rgb for c in list(Color('lime').range_to(Color('blue'), 256))]
    cmap = ListedColormap(cmap)
    cmap.set_bad("white")
    return cmap


def split_line(s, max_len):
    """
    Insert newlines in string so the width will be <= max_len

    Parameters
    ----------
    s           input string
    max_len     max width (in number of characters)

    Returns
    -------
    split string
    """
    if len(s) > max_len:
        words = s.split(' ')
        new_s = ""
        curr_len = 0
        for word in words:
            if curr_len + len(word) < max_len:
                if curr_len == 0:
                    new_s += word
                    curr_len += len(word)
                else:
                    new_s += ' ' + word
                    curr_len += len(word) + 1
            else:
                new_s += '\n' + word
                curr_len = len(word)
        return new_s
    return s


def list_feature_list_to_list_imgs(z):
    """
    Convert a list of feature lists to a list of matrices (also see: img_to_feature_list(img))

    Parameters
    ----------
    z       list of feature lists

    Returns
    -------

    """
    return tf.convert_to_tensor(np.reshape(z, (-1, 20, 11, 4)))


def get_mean_feature_values(all_imgs):
    """
    Get the average matrix from a list of matrices

    Parameters
    ----------
    all_imgs    list of matrices

    Returns
    -------
    average matrix
    """
    return np.mean(all_imgs, axis=0)


def img_to_feature_list(img):
    """
    Convert matrix to list of features by row-wise concatenation

    Parameters
    ----------
    img     matrix

    Returns
    -------
    list of features
    """
    if isinstance(img, np.ndarray):
        return img.flatten()
    else:
        return img.numpy().flatten()


def imgs_to_list_of_feature_lists(imgs):
    """
    Convert list of matrices to list of feature lists  (also see: img_to_feature_list(img))

    Parameters
    ----------
    imgs        list of matrices

    Returns
    -------
    list of feature lists
    """
    return imgs.reshape(imgs.shape[0], -1)


def duplicate_input_pair_lists(l, r):
    """
    Duplicate both input sequences

    Parameters
    ----------
    l       input 1
    r       input 2

    Returns
    -------
    Duplicated l and duplicated r
    """
    return l.repeat(2, 1), r.repeat(2, 1)


def concatted_inputs_to_input_pair_lists(concatted_inputs):
    """
    List of concattenated inputs to 2 lists of separate inputs

    Parameters
    ----------
    concatted_inputs

    Returns
    -------
    2 lists of separated inputs
    """
    return torch.stack([torch.tensor(i[:25]) for i in concatted_inputs]), torch.stack(
        [torch.tensor(i[25:]) for i in concatted_inputs])


def imrex_remove_padding(m, width, height):
    """
    Remove padding added by ImRex

    Parameters
    ----------
    m           Matrix from which to remove padding
    width       Final width to reach
    height      Final height to reach

    Returns
    -------
    Matrix without padding
    """
    ver_padding = m.shape[0] - width
    hor_padding = m.shape[1] - height

    ver_before = ver_padding // 2
    ver_after = ver_padding - ver_before

    hor_before = hor_padding // 2
    hor_after = hor_padding - hor_before

    m = m[ver_before:m.shape[0] - ver_after]
    m = m[:, hor_before:m.shape[1] - hor_after]

    return m


def imrex_remove_padding_from_3d_matrix(m_list, width, height):
    """
    Remove padding added by ImRex

    Parameters
    ----------
    m_list      3D matrix from which to remove padding on the last 2 dimensions
    width       Final width to reach
    height      Final height to reach

    Returns
    -------
    List of 2D matrices from which padding is removed
    """
    result_list = [imrex_remove_padding(m=m, width=width, height=height) for m in m_list]

    return result_list


def aa_remove_padding(att, in_data):
    """
    Remove padding added by TITAN

    Parameters
    ----------
    att         Array to remove padding from
    in_data     Original input data, used to know the dimensions without padding

    Returns
    -------
    Array without padding
    """
    start_indices = np.where(in_data == 30)[0]
    end_indices = np.where(in_data == 31)[0]
    return np.concatenate((att[start_indices[0] + 1:end_indices[0]], att[start_indices[1] + 1:end_indices[1]]))


def aa_add_padding(aa, l, fill=np.nan):
    """
    Add padding to AA data

    Parameters
    ----------
    aa      array to add padding to
    l       final lenght to reach
    fill    number to use as padding (np.nan by default)

    Returns
    -------
    Padded array
    """
    if len(aa) > l:
        print(f'Input {aa} already longer than padding length {l}')
    if len(aa) == l:
        return aa
    diff = l - len(aa)
    pad_before = diff // 2
    pad_after = diff - pad_before
    return np.concatenate(([fill] * pad_before, aa, [fill] * pad_after))


def remove_padding(aa, l):
    """
    TODO
    Parameters
    ----------
    aa
    l

    Returns
    -------

    """
    if len(aa) == l:
        return aa
    diff = len(aa) - l
    pad_before = diff // 2
    pad_after = diff - pad_before
    return aa[pad_before:len(aa) - pad_after]


def add_padding_2d(m, width, height, fill=np.nan):
    """
    Add padding to a 2D matrix

    Parameters
    ----------
    m       array to add padding to (2D numpy array)
    l       final length to reach
    fill    number to use as padding (np.nan by default)

    Returns
    -------
    Padded array
    """
    if m.shape[0] > width or m.shape[1] > height:
        print(f'Input matrix exceeding maximum dimensions')
    if m.shape[0] == width and m.shape[1] == height:
        return m

    hor_padding = - (m.shape[0] - width)
    ver_padding = - (m.shape[1] - height)

    ver_before = ver_padding // 2
    ver_after = ver_padding - ver_before

    hor_before = hor_padding // 2
    hor_after = hor_padding - hor_before

    pad_hor_before = np.full((hor_before, m.shape[1]), fill)
    pad_hor_after = np.full((hor_after, m.shape[1]), fill)
    m = np.concatenate((pad_hor_before, m, pad_hor_after), axis=0)

    pad_ver_before = np.full((m.shape[0], ver_before), fill)
    pad_ver_after = np.full((m.shape[0], ver_after), fill)
    m = np.concatenate((pad_ver_before, m, pad_ver_after), axis=1)

    return m


def normalize_2d(m):
    """
    Normalize matrix by dividing each value by the max value of the matrix. The largest value will become 1, the
    smallest not necessary zero

    Parameters
    ----------
    m       matrix to normalize

    Returns
    -------
    normalized matrix
    """
    max_val = np.max(m)
    if max_val == 0.0:
        return m
    m = m / max_val
    return m


def error_setup(dm, att):
    """
    Setup for error calculation: normalize inverse of distance matrix and normalize attribution matrix

    Parameters
    ----------
    dm      distance matrix
    att     attribution matrix

    Returns
    -------
    prepared distance matrix and attribution matrix
    """
    dm = normalize_2d(1 / dm)
    att = normalize_2d(att)
    return dm, att


def rmse(dm, att):
    """
    Calculate the root-mean-square error (RMSE) between the distance matrix and attribution matrix.
    The distance matrix is first inverted and normalized, the attribution matrix is first normalized.

    Parameters
    ----------
    dm      distance matrix
    att     attribution matrix

    Returns
    -------
    RMSE
    """
    dm, att = error_setup(dm, att)
    return np.sqrt(np.mean(np.square(dm - att)))


def rmse_for_list(dm_list, att_list):
    """
    Calculate the root-mean-square error (RMSE) between the distance matrix and attribution matrix for a list of
    distance matrices and a list of attribution matrices
    The distance matrix is first inverted and normalized, the attribution matrix is first normalized.

    Parameters
    ----------
    dm_list     list of distance matrices
    att_list    list of attribution matrices

    Returns
    -------
    RMSE
    """
    result_list = []
    for i, att in enumerate(att_list):
        dm_temp, att = error_setup(dm_list[i], att)
        result_list.append(np.sqrt(np.mean(np.square(dm_temp - att))))
    return result_list


def matrix_to_aa(m, method):
    """
    Merge a (k x l) matrix to an array using the 'min' or 'max' method. This takes the min/max for each row and column and
    concatenates those results into an array of length k + l.
    Parameters
    ----------
    m           matrix to merge
    method      Method to use. 'min' for taking the minimum, used on distance matrices. 'max' for taking the maximum,
                used on attribution matrices.

    Returns
    -------
    Array of merged matrix
    """
    if method == 'min':
        ep_dist = np.min(m, axis=0)
        cdr3_dist = np.min(m, axis=1)
    elif method == 'max':
        ep_dist = np.max(m, axis=0)
        cdr3_dist = np.max(m, axis=1)
    else:
        logging.getLogger(__name__).error(f'Method {method} not recognized in matrix_to_aa')
        return
    return np.concatenate((ep_dist, cdr3_dist))


def generate_path_inputs(baseline, input, alphas):
    """
    Create path inputs for IG

    Parameters
    ----------
    baseline    baseline image
    input       input image
    alphas      interpolation steps

    Returns
    -------
    interpolated inputs
    """
    # Expand dimensions for vectorized computation of interpolations.
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(input, axis=0)
    delta = input_x - baseline_x
    path_inputs = baseline_x + alphas_x * delta

    return path_inputs


def integral_approximation(gradients, method='riemann_trapezoidal'):
    """
    Solves the problem of discontinuous gradient feature importances by taking small steps in the feature space to
    compute local gradients between predictions and inputs across the feature space and then averages these gradients
    together to produce feature attributions.

    Parameters
    ----------
    gradients   input gradients
    method      method for integral approximation

    Returns
    -------
    IG
    """
    # different ways to compute the numeric approximation with different tradeoffs
    # riemann trapezoidal usually the most accurate
    if method == 'riemann_trapezoidal':
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    elif method == 'riemann_left':
        grads = gradients
    elif method == 'riemann_midpoint':
        grads = gradients
    elif method == 'riemann_right':
        grads = gradients
    else:
        raise AssertionError("Provided Riemann approximation method is not valid.")

    # average integration approximation
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients


def pdb2fasta_mapper(pdb_filepath, pdb_id, model, chain, fasta_seq):
    """
    The amino acids in the PDB file use other indices than those in the FASTA sequence format. This function uses
    pairwise alignment to match the amino acids and create a mapper to convert the index from the PDB file to the
    corresponding index in the FASTA sequence.

    This dict can be reversed (value -> key) to create a FASTA to PDB mapper.

    Parameters
    ----------
    pdb_filepath    Path to the PDB file
    pdb_id          PDB ID
    model           Model ID in the PDB file, usually 0
    chain           ID of the TCRB chain in the PDB file
    fasta_seq       FASTA sequence to align to

    Returns
    -------
    A 'mapper' (dict) with key PDB_ID and value FASTA_ID
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_filepath)
    pdb_tcrb_chain = structure[model][chain]
    pdb_tcrb_sel = Selection.unfold_entities(pdb_tcrb_chain, "R")

    pdb_tcrb_seq = ''
    pdb_tcrb_id_seq = []
    for r in pdb_tcrb_sel:
        if r.get_id()[0] == ' ':
            pdb_tcrb_seq += protein_letters_3to1[r.get_resname().title()]
            pdb_tcrb_id_seq.append(r.get_id()[1])

    aligner = Align.PairwiseAligner()
    alignment = aligner.align(pdb_tcrb_seq, fasta_seq)[0]
    pdb_aligned, fasta_aligned = alignment.aligned

    pdb2fasta_mapper = {}
    for pdb_chunck, fasta_chunk in zip(pdb_aligned, fasta_aligned):
        for pdb_chunck_id, fasta_chunk_id in zip(range(pdb_chunck[0], pdb_chunck[1]),
                                                 range(fasta_chunk[0], fasta_chunk[1])):
            pdb_numbering_id = pdb_tcrb_id_seq[pdb_chunck_id]
            pdb2fasta_mapper[pdb_numbering_id] = fasta_chunk_id

    return pdb2fasta_mapper


def seq2i_mapper(df, data_col):
    """
    Creates a mapper from TCR/epitope sequence to ID from the TCR/epitope file

    Parameters
    ----------
    df          Pandas dataframe of the TCR/epitope file
    data_col    Column name of the TCR/epitope sequences

    Returns
    -------
    A 'mapper' dict with key: TCR/epitope sequence and value: ID in the TCR/epitope file
    """
    return pd.Series(df['i'].values, index=df[data_col]).to_dict()


def i2seq_mapper(df, data_col):
    """
    Creates a mapper from the TCR/epitope file ID to the TCR/epitope sequence

    Parameters
    ----------
    df          Pandas dataframe of the TCR/epitope file
    data_col    Column name of the TCR/epitope sequences

    Returns
    -------
    A 'mapper' dict with key: ID in the TCR/epitope file and value: the TCR/epitope sequence
    """
    return pd.Series(df[data_col].values, index=df['i']).to_dict()


def residue_distance_min(res1, res2):
    """
    Calculate the (minimal) distance between 2 amino acids by returning the distance of the 2 closest atoms.

    Parameters
    ----------
    res1    Amino acid 1
    res2    Amino acid 2

    Returns
    -------
    Distance
    """
    dist = np.inf
    for a1 in res1:
        for a2 in res2:
            d = a1 - a2
            if d < dist:
                dist = d
    return dist


def get_distance_matrices():
    """
    Calculate the distance matrix between the epitope and CDR3 sequence for each PDB complex.

    Returns
    -------
    A distance matrix for each PDB complex
    """
    tcr3df = pd.read_csv("data/tcr3d_imrex_output.csv")
    complex_data_df = pd.read_csv('data/complex_data_original.csv', index_col=0)
    distance_matrices = {}
    for pdb_id in tcr3df['PDB_ID']:
        complex_data = complex_data_df.loc[pdb_id]
        ep_chain_id = complex_data['epitope_chain']
        tcrb_chain_id = complex_data['tcrb_chain']

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, f'data/pdb/{pdb_id.lower()}.pdb')[0]
        ep_chain = structure[ep_chain_id]
        tcrb_chain = structure[tcrb_chain_id]

        mapper = pdb2fasta_mapper(f'data/pdb/{pdb_id.lower()}.pdb', pdb_id, 0, tcrb_chain_id, complex_data['tcrb_seq'])
        pdb_mapper = {v: k for k, v in mapper.items()}

        dist_matrix = np.zeros((len(complex_data['cdr3']), len(complex_data['antigen.epitope'])))
        for i, mi in zip(range(int(complex_data['CDR3_start']), int(complex_data['CDR3_end']) + 1),
                         range(dist_matrix.shape[0])):
            if i in pdb_mapper:
                for ep, mj in zip(ep_chain, range(dist_matrix.shape[1])):
                    dist_matrix[mi][mj] = residue_distance_min(tcrb_chain[pdb_mapper[i]], ep)
        distance_matrices[pdb_id] = dist_matrix
    return distance_matrices


def setup_logger(name):
    log_file = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')}{name}.log"
    level = logging.INFO
    # create file logger
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_file, level=level, format=log_fmt)
    # apply settings to root logger, so that loggers in modules can inherit both the file and console logger
    logger = logging.getLogger()
    # add console logger
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(log_fmt))
    logger.addHandler(console)

    # suppress tf logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ERROR
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("shap").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def correlation_nan(f, x, y, replace=0., with_p=False):
    """
    Calculate the correlation and p-value between input 1 (x) and input 2 (y), if the correlation is NaN, replace with
    the replace value

    Parameters
    ----------
    f           correlation function
    x           input 1
    y           input 2
    replace     value to replace a NaN correlation with
    with_p      Also return the p-value of the correlation if True

    Returns
    -------
    Correlation between x and y or replace when NaN, p-value if with_p is True
    """
    c, p = f(x, y)
    if with_p:
        return replace if np.isnan(c) else c, p
    else:
        return replace if np.isnan(c) else c


def p_value_stats(model_name, method_p):
    print(f"{model_name} aa correlation p-values")
    for method, ps in method_p.items():
        if method in ['SHAP BGdist', 'Vanilla', 'SmoothGrad', 'VanillaIG']:
            num_significant1 = sum(1 for p in ps if p < 0.05)
            num_significant2 = sum(1 for p in ps if p < 0.01)
            num_significant3 = sum(1 for p in ps if p < 0.001)
            print(
                f"{method}\t{num_significant1 * 100 / len(ps):.2f}%\t{num_significant2 * 100 / len(ps):.2f}%\t"
                f"{num_significant3 * 100 / len(ps):.2f}%")
