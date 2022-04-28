"""
Used to convert the molecular complex data to input for ImRex and TITAN for feature attribution extraction
"""

import pandas as pd


def seq2i_mapper(df, data_col):
    return pd.Series(df['i'].values, index=df[data_col]).to_dict()


def tcr3d_to_imrex():
    tcr3df = pd.read_csv("data/complex_data_original.csv")
    imrex_data = pd.DataFrame(
        {'PDB_ID': tcr3df['PDB_ID'], 'cdr3': tcr3df['cdr3'], 'antigen.epitope': tcr3df['antigen.epitope']})
    imrex_data.to_csv('data/tcr3d_imrex_input.csv', index=None, sep=';')


def create_tcr3d_all_files():
    tcr3df = pd.read_csv("data/complex_data_original.csv")
    all_epitopes = pd.DataFrame({'ep': tcr3df['antigen.epitope'].unique()})
    all_epitopes['i'] = all_epitopes.index

    all_tcrs = pd.DataFrame({'cdr3': tcr3df['cdr3'].unique()})
    all_tcrs['i'] = all_tcrs.index

    all_epitopes.to_csv('data/epitopes.csv', sep='\t', header=None, index=None)
    all_tcrs.to_csv('data/tcrs.csv', sep='\t', header=None, index=None)


def tcr3d_to_titan():
    tcr3df = pd.read_csv("data/tcr3d_imrex_output.csv")
    all_epitopes = pd.read_csv(f'data/epitopes.csv', sep='\t', header=None, names=['ep', 'i'])
    all_tcrs = pd.read_csv(f'data/tcrs.csv', sep='\t', header=None, names=['cdr3', 'i'])
    tcr3df = tcr3df[['antigen.epitope', 'cdr3']]

    ep_mapper = seq2i_mapper(all_epitopes, 'ep')
    cdr3_mapper = seq2i_mapper(all_tcrs, 'cdr3')

    tcr3df['antigen.epitope'] = tcr3df['antigen.epitope'].map(ep_mapper)
    tcr3df['cdr3'] = tcr3df['cdr3'].map(cdr3_mapper)
    tcr3df = tcr3df.rename(columns={'antigen.epitope': 'ligand_name', 'cdr3': 'sequence_id'})
    tcr3df['label'] = 1
    tcr3df.to_csv(f'data/tcr3d_titan_input.csv', sep=',')


if __name__ == "__main__":
    tcr3d_to_titan()
    create_tcr3d_all_files()
    tcr3d_to_imrex()
