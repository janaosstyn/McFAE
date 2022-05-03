import random

import pandas as pd


def scramble_tcrs():
    """
    Create a new TITAN TCR file with scrambled TCR sequences. The amino acid usage distribution is kept.
    """
    all_tcrs = pd.read_csv(f'TITAN/data/tcr.csv', sep='\t', header=None, names=['cdr3', 'i'])
    aa_list = list({char for seq in all_tcrs['cdr3'].values for char in seq})
    new_tcrs = []
    for tcr in all_tcrs['cdr3']:
        new_tcrs.append(''.join(random.choices(aa_list, k=len(tcr))))
    all_tcrs['cdr3'] = new_tcrs
    all_tcrs.to_csv(f'TITAN/data/scrambled_tcr.csv', sep='\t', header=None, index=None)


def main():
    scramble_tcrs()


if __name__ == "__main__":
    main()
