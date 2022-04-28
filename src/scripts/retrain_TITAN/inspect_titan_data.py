import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def i2seq_mapper(df, data_col):
    return pd.Series(df[data_col].values, index=df['i']).to_dict()


def precision_group(x):
    return precision_score(x['label'].to_numpy(dtype=float), np.round(x['pred'].to_numpy()))


def recall_group(x):
    return recall_score(x['label'].to_numpy(dtype=float), np.round(x['pred'].to_numpy()))


def positive_rate_group(x):
    preds = np.round(x['pred'].to_numpy())
    return np.count_nonzero(preds) / len(preds)


def inspect_titan_dataset(datafolder, model, cv):
    titan_p = f"TITAN/data/{datafolder}/fold{cv}/test+covid.csv"
    titan_train_p = f"TITAN/data/{datafolder}/fold{cv}/train+covid.csv"
    titan_predictions = np.load(f'TITAN/models/{model}/cv{cv}/{model}_{cv}/results/ROC-AUC_preds.npy')

    all_epitopes = pd.read_csv(f'TITAN/data/epitopes.csv', sep='\t', header=None, names=['ep', 'i'])
    all_tcrs = pd.read_csv(f'TITAN/data/tcr.csv', sep='\t', header=None, names=['cdr3', 'i'])
    ep_mapper = i2seq_mapper(all_epitopes, 'ep')
    cdr3_mapper = i2seq_mapper(all_tcrs, 'cdr3')

    # read both datasets and map with their real AA sequence
    titan_d = pd.read_csv(titan_p, index_col=0)
    titan_train_d = pd.read_csv(titan_train_p, index_col=0)
    titan_d['ligand_name'] = titan_d['ligand_name'].map(ep_mapper)
    titan_d['sequence_id'] = titan_d['sequence_id'].map(cdr3_mapper)
    titan_train_d['ligand_name'] = titan_train_d['ligand_name'].map(ep_mapper)
    titan_train_d['sequence_id'] = titan_train_d['sequence_id'].map(cdr3_mapper)

    # split in negative and positive datasets
    titan_pos = titan_d[titan_d['label'] == 1][['ligand_name', 'sequence_id']].reset_index(drop=True)
    titan_train_pos = titan_train_d[titan_train_d['label'] == 1][['ligand_name', 'sequence_id']].reset_index(drop=True)
    titan_neg = titan_d[titan_d['label'] == 0][['ligand_name', 'sequence_id']].reset_index(drop=True)
    titan_train_neg = titan_train_d[titan_train_d['label'] == 0][['ligand_name', 'sequence_id']].reset_index(drop=True)

    # remove duplicates and sequence of bad length
    titan_pos_dup = titan_pos.duplicated()
    titan_train_pos_dup = titan_train_pos.duplicated()

    titan_pos = titan_pos[~titan_pos_dup]
    titan_train_pos = titan_train_pos[~titan_train_pos_dup]
    titan_pos = titan_pos[titan_pos['sequence_id'].str.len() >= 10]
    titan_pos = titan_pos[titan_pos['sequence_id'].str.len() <= 23]
    titan_train_pos = titan_train_pos[titan_train_pos['sequence_id'].str.len() >= 10]
    titan_train_pos = titan_train_pos[titan_train_pos['sequence_id'].str.len() <= 23]

    titan_neg = titan_neg[~titan_pos_dup]
    titan_train_neg = titan_train_neg[~titan_train_pos_dup]
    titan_neg = titan_neg[titan_neg['sequence_id'].str.len() >= 10]
    titan_neg = titan_neg[titan_neg['sequence_id'].str.len() <= 23]
    titan_train_neg = titan_train_neg[titan_train_neg['sequence_id'].str.len() >= 10]
    titan_train_neg = titan_train_neg[titan_train_neg['sequence_id'].str.len() <= 23]

    # create overview of # samples per epitope
    epitopes = titan_d['ligand_name'].unique()
    ep_stats = pd.DataFrame(index=epitopes)
    ep_stats['# pos'] = titan_pos['ligand_name'].value_counts()
    ep_stats['TITAN # neg'] = titan_neg['ligand_name'].value_counts()

    ep_stats = ep_stats.sort_values('# pos', ascending=False)
    predictions = titan_d.assign(pred=titan_predictions[0])

    ep_stats['TITAN precision'] = predictions.groupby('ligand_name').apply(precision_group)
    ep_stats['TITAN recall'] = predictions.groupby('ligand_name').apply(recall_group)
    ep_stats['TITAN PPR'] = predictions.groupby('ligand_name').apply(positive_rate_group)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(ep_stats)
    ep_stats.to_csv(f'output/tables/titan_inspection/{datafolder}_{model}_cv{cv}.csv')


def main():
    inspect_titan_dataset('strict_split_nocdr3', 'titanData_strictsplit_nocdr3', cv=5)


if __name__ == "__main__":
    main()
