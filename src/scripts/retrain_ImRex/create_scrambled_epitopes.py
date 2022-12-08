import random

import pandas as pd


def scramble_epitopes(full_dataset_path, dataset_path):
    full_df = pd.read_csv(full_dataset_path, delimiter=';')
    df = pd.read_csv(dataset_path, delimiter=';')
    epitopes = full_df['antigen.epitope'].unique()

    aa_list = list(char for seq in epitopes for char in seq)

    new_epitopes = {}
    for epitope in epitopes:
        new_epitopes[epitope] = ''.join(random.choices(aa_list, k=len(epitope)))

    for epitope in df['antigen.epitope'].unique():
        if epitope not in epitopes:
            print(epitope, 'not in full df')

    full_df = full_df.replace({'antigen.epitope': new_epitopes})
    df = df.replace({'antigen.epitope': new_epitopes})

    save_path = '/'.join(full_dataset_path.split('/')[:-1])
    full_name = '.'.join(full_dataset_path.split('/')[-1].split('.')[:-1])
    dataset_name = '.'.join(dataset_path.split('/')[-1].split('.')[:-1])

    full_df.to_csv(f'{save_path}/{full_name}_scrambled_eps.csv', sep=';', index=None)
    df.to_csv(f'{save_path}/{dataset_name}_scrambled_eps.csv', sep=';', index=None)


def scramble_tcrs(full_dataset_path, dataset_path, uniform_length=None):
    full_df = pd.read_csv(full_dataset_path, delimiter=';')
    df = pd.read_csv(dataset_path, delimiter=';')
    tcrs = full_df['cdr3'].unique()

    aa_list = list(char for seq in tcrs for char in seq)

    new_tcrs = {}
    for tcr in tcrs:
        if uniform_length is not None:
            new_tcrs[tcr] = ''.join(random.choices(aa_list, k=uniform_length))
        else:
            new_tcrs[tcr] = ''.join(random.choices(aa_list, k=len(tcr)))

    for tcr in df['cdr3'].unique():
        if tcr not in tcrs:
            print(tcr, 'not in full df')

    full_df = full_df.replace({'cdr3': new_tcrs})
    df = df.replace({'cdr3': new_tcrs})

    save_path = '/'.join(full_dataset_path.split('/')[:-1])
    full_name = '.'.join(full_dataset_path.split('/')[-1].split('.')[:-1])
    dataset_name = '.'.join(dataset_path.split('/')[-1].split('.')[:-1])

    if uniform_length is not None:
        full_df.to_csv(f'{save_path}/{full_name}_scrambled_tcrs_{uniform_length}.csv', sep=';', index=None)
        df.to_csv(f'{save_path}/{dataset_name}_scrambled_tcrs_{uniform_length}.csv', sep=';', index=None)
    else:
        full_df.to_csv(f'{save_path}/{full_name}_scrambled_tcrs.csv', sep=';', index=None)
        df.to_csv(f'{save_path}/{dataset_name}_scrambled_tcrs.csv', sep=';', index=None)


if __name__ == "__main__":
    scramble_tcrs('ImRex/data/interim/vdjdb-2019-08-08/vdjdb-human.csv',
                  'ImRex/data/interim/vdjdb-2019-08-08/vdjdb-human-trb-mhci-no10x-size-down400_pdb_no_cdr3_dup.csv', 15)
