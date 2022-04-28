import os
import pandas as pd


def seq2i_mapper(df, data_col):
    return pd.Series(df['i'].values, index=df[data_col]).to_dict()


def i2seq_mapper(df, data_col):
    return pd.Series(df[data_col].values, index=df['i']).to_dict()


def remove_3d_cdr3s_from_titan(data_path, save_folder, all_epitopes, all_tcrs):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    i2ep_mapper = i2seq_mapper(all_epitopes, 'ep')
    i2cdr3_mapper = i2seq_mapper(all_tcrs, 'cdr3')
    ep2i_mapper = seq2i_mapper(all_epitopes, 'ep')
    cdr32i_mapper = seq2i_mapper(all_tcrs, 'cdr3')

    pdb_data = pd.read_csv('ImRex/data/interim/TCR3D_valid.csv')

    for cv in range(10):
        for t in ['train', 'test']:
            print(f"CV {cv}, {t}")
            td = pd.read_csv(f'{data_path}/fold{cv}/{t}+covid.csv', index_col=0)
            td['ligand_name'] = td['ligand_name'].map(i2ep_mapper)
            td['sequence_id'] = td['sequence_id'].map(i2cdr3_mapper)

            num_data = len(td)
            new_data = td[~td['sequence_id'].isin(pdb_data['cdr3'])]
            new_data = new_data.reset_index(drop=True)

            new_data['ligand_name'] = new_data['ligand_name'].map(ep2i_mapper)
            new_data['sequence_id'] = new_data['sequence_id'].map(cdr32i_mapper)
            print(f"Removed {num_data - len(new_data)}/{num_data} samples based on cdr3 duplicates")
            print(f"{len(new_data)} positive samples remaining")

            if not os.path.exists(f"{save_folder}fold{cv}/"):
                os.makedirs(f"{save_folder}fold{cv}/")

            new_data.to_csv(f"{save_folder}fold{cv}/{t}+covid.csv")


def main():
    titan_epitopes = pd.read_csv(f'TITAN/data/epitopes.csv', sep='\t', header=None, names=['ep', 'i'])
    titan_tcrs = pd.read_csv(f'TITAN/data/tcr.csv', sep='\t', header=None, names=['cdr3', 'i'])

    remove_3d_cdr3s_from_titan('TITAN/data/strict_split', 'TITAN/data/strict_split_nocdr3/', titan_epitopes, titan_tcrs)


if __name__ == "__main__":
    main()
