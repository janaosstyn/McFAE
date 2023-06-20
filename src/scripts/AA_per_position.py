import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_array(dataframe):
    cdr3_sequences = dataframe['cdr3'].tolist()
    max_len = len(max(cdr3_sequences, key=len))
    dataframe = pd.DataFrame()
    for sequence in cdr3_sequences:
        diff = max_len - len(sequence)
        pad_before = diff // 2
        pad_after = diff - pad_before

        dataframe[sequence] = [c for c in ('0' * pad_before + sequence + '0' * pad_after)]

    all_counts = []
    all_labels = []
    unique_labels = set()
    df = pd.DataFrame({-1: [0, 0, 0, 0, 0, 0, 0, 0, 0]})
    df = df.set_index(pd.Index(['Variance', 'Standard deviation', 'Mean', 'Median', 'Q1', 'Q2', 'Q3', 'IQR', '% padding']))
    for row in range(dataframe.shape[0]):
        row_list = dataframe.loc[row].tolist()
        labels, counts = np.unique(row_list, return_counts=True)
        print(labels)
        print(counts)
        all_labels.append(labels.tolist())
        all_counts.append(counts.tolist())
        unique_labels.update(labels.tolist())
        df[row] = [
            np.var(counts),
            np.std(counts),
            np.mean(counts),
            np.median(counts),
            np.quantile(counts, 0.25),
            np.quantile(counts, 0.5),
            np.quantile(counts, 0.75),
            np.quantile(counts, 0.75) - np.quantile(counts, 0.25),
            0 if labels[0] != '0' else counts[0] / np.sum(counts)
        ]

        print()

    df = df.drop(columns=[-1])
    df = df.T
    df['Normalized standard deviation'] = df['Standard deviation'] / df['Standard deviation'].max()
    columns = df.columns
    for column in columns[:-1]:
        df[[column]].plot.line()
        ax = df['% padding'].plot(secondary_y=True, color='r')
        ax.set_ylabel('% padding')
        plt.xticks(list(range(df.shape[0])))
        plt.xlabel('Position')
        plt.ylabel(column)
        plt.title(f'{column} per position')
        plt.show()
        plt.clf()
    print(df)

    labels = {label: [] for label in sorted(list(unique_labels))}
    for i in range(len(all_labels)):
        for label in sorted(list(unique_labels)):
            if label in all_labels[i]:
                labels[all_labels[i][all_labels[i].index(label)]].append(all_counts[i][all_labels[i].index(label)])
            else:
                labels[label].append(0)

    new_dataframe = pd.DataFrame(labels)
    new_dataframe = (new_dataframe / dataframe.shape[1])
    new_dataframe = new_dataframe.reset_index()
    new_dataframe.plot(
        x="index",
        y=sorted(list(unique_labels)),
        stacked=True,
        kind="bar",
        figsize=(11, 9),
        cmap=plt.cm.get_cmap('tab20b_r', len(unique_labels))
    )
    plt.title('# occurrences per AA per position')
    plt.xlabel('Position')
    plt.ylabel('# occurrences')
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", prop={'size': 12})
    plt.set_cmap('plasma')
    plt.show()

    new_dataframe.drop(columns="index", inplace=True)
    new_dataframe.to_csv('data/per_position_variability.csv', index=False)

    df.to_csv('data/variability_metrics.csv', index=False)
    print(df)
    return cdr3_sequences


data = pd.read_csv('../ImRex/data/interim/vdjdb-2019-08-08/vdjdb-human-trb-mhci-no10x-size-down400_pdb_no_cdr3_dup.csv', ';')
seq = to_array(data)
print()
