import pandas


def imrex_cv(model):
    """
    Create an overview of the validation ROC AUC and PR AUC per cross-validation split

    Parameters
    ----------
    model       ImRex model name
    """
    result_list = []
    for cv in range(5):
        results = pandas.read_csv(f'ImRex/models/models/{model}/iteration_{cv}/metrics.csv')

        val_roc_auc = results['val_roc_auc'].iloc[-1]
        val_pr_auc = results['val_pr_auc'].iloc[-1]

        result_list.append(
            pandas.DataFrame([[cv, val_roc_auc, val_pr_auc]], columns=['cv', 'val_roc_auc', 'val_pr_auc']))
    results = pandas.concat(result_list, ignore_index=True)
    results.to_csv(f'ImRex/models/models/{model}/full_metrics.csv', index=None)


if __name__ == "__main__":
    imrex_cv('2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv')
