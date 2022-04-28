import os
import pandas as pd


def seq2i_mapper(df, data_col):
    return pd.Series(df['i'].values, index=df[data_col]).to_dict()


def i2seq_mapper(df, data_col):
    return pd.Series(df[data_col].values, index=df['i']).to_dict()


def transform_data_cv(data_folder, save_folder, all_epitopes, all_tcrs):
    ep_mapper = seq2i_mapper(all_epitopes, 'ep')
    cdr3_mapper = seq2i_mapper(all_tcrs, 'cdr3')

    for cv in range(5):
        train = pd.read_csv(f'{data_folder}iteration_{cv}/train_fold_{cv}.csv', sep=';')
        test = pd.read_csv(f'{data_folder}iteration_{cv}/test_fold_{cv}.csv', sep=';')

        for data, d_name in zip([train, test], ['train', 'test']):
            data['antigen.epitope'] = data['antigen.epitope'].map(ep_mapper)
            data['cdr3'] = data['cdr3'].map(cdr3_mapper)
            data = data.rename(columns={'antigen.epitope': 'ligand_name', 'cdr3': 'sequence_id', 'y': 'label'})
            data.to_csv(f'{save_folder}{d_name}{cv}.csv', sep=',')


def main():
    imrex_epitopes = pd.read_csv(f'TITAN/data/imrex_data/epitopes.csv', sep='\t', header=None, names=['ep', 'i'])
    imrex_tcrs = pd.read_csv(f'TITAN/data/imrex_data/tcrs.csv', sep='\t', header=None, names=['cdr3', 'i'])

    data_folder = 'TITAN/data/imrex_data/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/'
    save_folder = 'TITAN/data/imrex_data/nocdr3dup_default_epgrouped5cv/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    transform_data_cv(data_folder, save_folder, imrex_epitopes, imrex_tcrs)


if __name__ == "__main__":
    main()
