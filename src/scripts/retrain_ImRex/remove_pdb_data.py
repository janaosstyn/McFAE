import pandas


def main(data_path, save_nocdr3_path):
    """
    Remove samples already present in the PDB dataset from the ImRex dataset

    Parameters
    ----------
    data_path           ImRex data path
    save_nocdr3_path    New data path
    """
    pdb_data = pandas.read_csv('ImRex/data/interim/TCR3D_valid.csv')
    imrex_data = pandas.read_csv(data_path, sep=";")

    num_data = len(imrex_data)
    new_data = imrex_data[~imrex_data['cdr3'].isin(pdb_data['cdr3'])]
    print(f"Removed {num_data - len(new_data)}/{num_data} samples based on cdr3 duplicates")
    print(f"{len(new_data)} positive samples remaining")
    new_data.to_csv(save_nocdr3_path, sep=";", index=False)


if __name__ == "__main__":
    main('ImRex/data/interim/vdjdb-2019-08-08/vdjdb-human-trb-mhci-no10x-size-down400.csv',
         'ImRex/data/interim/vdjdb-2019-08-08/vdjdb-human-trb-mhci-no10x-size-down400_pdb_no_cdr3_dup.csv')
