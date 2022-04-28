import pandas


def create_full_metrics(model, cvs):
    result_list = []
    for cv in range(cvs):
        results = pandas.read_csv(f'TITAN/models/{model}/cv{cv}/results_overview.csv', header=None,
                                  names=['model_name', 'roc_auc', 'ap', 'loss', 'epoch'])
        val_roc_auc = results['roc_auc'].iloc[0]
        val_pr_auc = results['ap'].iloc[0]
        result_list.append(
            pandas.DataFrame([[cv, val_roc_auc, val_pr_auc]], columns=['cv', 'val_roc_auc', 'val_pr_auc']))

    results = pandas.concat(result_list, ignore_index=True)
    results.to_csv(f'TITAN/models/{model}/full_metrics.csv', index=None)


def main():
    create_full_metrics("titanData_strictsplit_nocdr3", 10)
    create_full_metrics("nocdr3dup_epgrouped5cv_paperparams_smallpad", 5)
    create_full_metrics("titanData_strictsplit_scrambledtcrs", 10)


if __name__ == "__main__":
    main()
