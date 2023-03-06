import math
import os
import pickle
from typing import List, Dict

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from src.imrex_attributions import ImrexAttributionsHandler
from src.titan_attributions import TITANAttributionsHandler
from src.util import aa_add_padding, split_line, imrex_remove_padding


def extract_method(dictionary, method_name):
    result = dict()
    for pdb_id in dictionary:
        result[pdb_id] = dictionary[pdb_id][method_name]
    return result


def extract_channel(dictionary, channel):
    result = dict()
    for pdb_id in dictionary:
        if pdb_id.endswith(f'_{channel}'):
            result[pdb_id] = dictionary[pdb_id]
    return result


def split_channels(attributions: [List, Dict]) -> List[List[Dict]]:
    """
    Given dictionary --> return list with as content a single list of dictionaries, 1 dictionary / channel.
    Given list       --> return list of lists where each list within the list consists of dictionaries (1 / channel)

    Parameters
    ----------
    attributions: either a list of dictionaries or a dictionary

    Returns
    -------
    The resulting list with lists of dictionaries
    """
    if isinstance(attributions, list):
        channels = []
        for i in range(len(attributions)):
            channels.extend(split_channels(attributions[i]))
        return channels
    else:
        channels = [{}, {}, {}, {}]
        for key, value in attributions.items():
            if isinstance(list(value.keys())[0], int):
                for channel, attributions in value.items():
                    channels[channel][key] = attributions
            elif isinstance(key, int):
                return [[attributions[i] for i in range(len(attributions))]]
            else:
                return [[attributions]]
        return [channels]


def to_dataframes(
        attributions,
        model,
        keep_columns=None,
        remove_columns=None,
        rename_columns=None,
        expand=None
):
    attributions = split_channels(attributions)
    result_list = []
    for i in range(len(attributions)):
        result = None
        for j in range(len(attributions[i])):
            df = pd.DataFrame(attributions[i][j]).T
            if keep_columns is not None:
                remove_columns = [column for column in df.columns.values.tolist() if column not in keep_columns]
            if remove_columns is not None:
                df = df.drop(columns=remove_columns)
            if rename_columns is not None:
                df = df.rename(columns=rename_columns)

            if expand is not None:
                for column in df.columns.values.tolist():
                    new_columns = [f'{column}_{expand_name}' for expand_name in expand]
                    df[new_columns] = df[column].apply(lambda x: pd.Series(x))
                    df = df.drop(columns=[column])
            df = df * (-1)
            df = df.rename(columns={column: f'{model}_{column}_{j}' for column in df.columns.values.tolist()})
            if result is None:
                result = df
            else:
                result[df.columns.values.tolist()] = df
        result_list.append(result)
    return result_list


def get_comparison_plot_filename(
        model_handlers,
        keywords=None,
        method=None,
        correlation_method=None,
        channels=False,
        combined=False
):
    import inspect
    dir_name = f"output/plots/comparison_plots/{inspect.currentframe().f_back.f_code.co_name.replace('plot_', '')}_plots"

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    return f"{dir_name}/" \
           f"{'_'.join([str(handler) for handler in model_handlers])}" \
           f"{f'_{method}' if method is not None else ''}" \
           f"{f'_{correlation_method}' if correlation_method is not None else ''}" \
           f"{'' if keywords is None else '_' + '_'.join(keywords)}" \
           f"{f'_ch' if channels else ''}" \
           f"{f'_comb' if channels and combined else ''}" \
           f".png"


def get_sample_2d_details_filename(
        model_handler,
        pdb_id,
        method,
        channels=False,
        combined=False
):
    dir_name = f'output/plots/sample_2d_details/{str(model_handler)}'

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    return f"{dir_name}/" \
           f"{pdb_id}" \
           f"_{method}" \
           f"{f'_ch' if channels else ''}" \
           f"{f'_comb' if channels and combined else ''}" \


def plot_model_performance_comparison(metrics, names, save_name=""):
    if len(names) > 3:
        plt.figure(figsize=(6.4 / 1.3, 4.8 / 1.3))
    else:
        plt.figure(figsize=(3.2 / 1.3, 4.8 / 1.3))

    val_roc_aucs = [m['val_roc_auc'].to_numpy() for m in metrics]
    val_pr_aucs = [m['val_pr_auc'].to_numpy() for m in metrics]
    for metric, m_name, short_name in zip([val_roc_aucs, val_pr_aucs], ['ROC AUC', 'PR AUC'], ['roc', 'pr']):
        # for name, results in zip(names, metric):
        #     print(name, m_name, round(np.mean(results), 3), '+-', round(np.std(results), 3))
        sns.boxplot(data=metric, showfliers=False)
        sns.stripplot(data=metric, color='0.25', s=4)
        plt.ylabel(m_name)
        plt.xticks(range(len(names)), [split_line(n, 13) for n in names])
        plt.grid(axis='y')
        plt.tight_layout()
        # plt.savefig(
        #     fname=get_comparison_plot_filename(
        #
        #
        #         variables=[short_name, save_name])
        #     , dpi=300
        # )
        plt.savefig(  # TODO
            f'output/plots/model_performance_comparison_{short_name}{"" if save_name is None else "_" + save_name}.png',
            dpi=300)
        plt.clf()
    # print()


def plot_model_performance():
    imrex = pd.read_csv(
        'ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/full_metrics.csv')
    imrex_scrambled_eps = pd.read_csv('ImRex/models/models/2022-11-30_13-55-56_scrambled_eps/full_metrics.csv')
    imrex_scrambled_tcrs = pd.read_csv('ImRex/models/models/2022-12-01_09-52-59_scrambled_tcrs/full_metrics.csv')
    titan_on_imrex_data = pd.read_csv('TITAN/models/nocdr3dup_epgrouped5cv_paperparams_smallpad/full_metrics.csv')
    titan_scrambled_tcs = pd.read_csv('TITAN/models/titanData_strictsplit_scrambledtcrs/full_metrics.csv')
    titan = pd.read_csv('TITAN/models/titanData_strictsplit_nocdr3/full_metrics.csv')

    sns.set_palette("deep")
    plot_model_performance_comparison(
        [imrex, titan],
        ['ImRex', 'TITAN'], "subset")
    plot_model_performance_comparison(
        [imrex, titan, titan_on_imrex_data, titan_scrambled_tcs],
        ['ImRex', 'TITAN', 'TITAN on ImRex data', 'TITAN scrambled TCRs'])
    plot_model_performance_comparison(
        [imrex, imrex_scrambled_eps, titan, titan_on_imrex_data, titan_scrambled_tcs],
        ['ImRex', 'ImRex scrambled epitopes', 'TITAN', 'TITAN on ImRex data', 'TITAN scrambled TCRs'],
        'scrambled epitopes')

    plot_model_performance_comparison(
        [imrex, imrex_scrambled_tcrs, titan, titan_on_imrex_data, titan_scrambled_tcs],
        ['ImRex', 'ImRex scrambled TCRs', 'TITAN', 'TITAN on ImRex data', 'TITAN scrambled TCRs'],
        'scrambled tcrs')


def plot_method_correlation_comparison(
        attribution_handler: [ImrexAttributionsHandler, TITANAttributionsHandler],
        methods_subset: List[str] = None,
        display_combined=False,
        correlation_method='pearson'
):
    """
    Create a plot that compares correlation between the different feature attribution extraction methods.
    If the correlations are calculated over the 4 physicochemical properties (channels), a single figure with 4
    subplots is created (one for each property).

    Parameters
    ----------
    attribution_handler: either ImrexAttributionsHandler or TITANAttributionsHandler
    methods_subset: a list of methods to plot (optionally, if not provided all methods are plotted)
    display_combined
    correlation_method
    """

    # differentiate based on the type of the attributions handler
    if isinstance(attribution_handler, ImrexAttributionsHandler):
        attribution_types = ['aa', 'pair-wise']
        correlations = [
            attribution_handler.get_aa_correlation(correlation_method),
            attribution_handler.get_correlation(correlation_method)
        ]
        random_correlations = [
            attribution_handler.get_aa_random_correlation(correlation_method),
            attribution_handler.get_random_correlation(correlation_method)
        ]
    elif isinstance(attribution_handler, TITANAttributionsHandler):
        attribution_types = ['aa']
        correlations = [attribution_handler.get_aa_correlation(correlation_method)]
        random_correlations = [attribution_handler.get_aa_random_correlation(correlation_method)]
    else:
        raise TypeError(
            f"attribution_handler of wrong type, got {type(attribution_handler)} but expected "
            f"{ImrexAttributionsHandler} or {TITANAttributionsHandler}"
        )

    # create plot for each attribute type
    for attribution_type, correlation_results, random_results in zip(attribution_types, correlations,
                                                                     random_correlations):
        # get the method subset
        if methods_subset is None:
            methods = list(list(correlation_results.values())[0].keys())
            methods = set(methods) - {'IG', 'SHAP mean'}

            if set(methods) == {
                'SHAP BGdist', 'Vanilla', 'SmoothGrad', 'VanillaIG', 'SmoothGradIG',
                'GuidedIG', 'XRAI', 'BlurIG', 'SmoothGradBlurIG'
            }:
                methods = [  # order methods
                    'Vanilla', 'VanillaIG', 'SmoothGrad', 'SmoothGradIG', 'GuidedIG',
                    'BlurIG', 'SmoothGradBlurIG', 'XRAI', 'SHAP BGdist'
                ]
        else:
            methods = methods_subset

        channel_split = False
        # get the correlations per method (1 dictionary/method in case of channel split, else 1 dictionary)
        method_correlation = {}
        for pdb_id, pdb_method_corr in correlation_results.items():
            for method, method_corr in pdb_method_corr.items():
                if method not in methods:
                    continue  # skip method
                elif method not in method_correlation:
                    method_correlation[method] = []

                key = None if '_' not in pdb_id else \
                    (f'Ch {pdb_id.split("_")[1]}' if "combined" not in pdb_id else "Combi")

                if key == "Combi" and not display_combined:
                    continue

                if key is not None and key not in method_correlation[method]:
                    channel_split = True
                    if isinstance(method_correlation[method], list):
                        method_correlation[method] = {key: [-method_corr]}
                    else:
                        method_correlation[method][key] = [-method_corr]
                elif key is not None:
                    method_correlation[method][key].append(-method_corr)
                else:
                    method_correlation[method].append(-method_corr)

        # SHAP BGdist should be displayed as SHAP, VanillaIG as IG
        correlation = {method: method_correlation[method] for method in methods}
        correlation = {"SHAP" if k == 'SHAP BGdist' else "IG" if k == "VanillaIG" else k: v for k, v in
                       correlation.items()}

        f, axs = plt.subplots(
            nrows=3 if channel_split else 1,
            ncols=3 if channel_split else 1,
            figsize=(12 if channel_split else 11, 12 if channel_split else 3),
            sharex=True,
            sharey=True
        )

        dataframes = []
        if channel_split:
            for method in correlation:
                corr_df = pd.DataFrame(correlation[method])
                dataframes.append(corr_df)
            name_axs = [axs[j][k] for j in range(3) for k in range(3)]
            x_labels = list(correlation.keys())
            axs[1][0].set_ylabel("Pearson correlation")
        else:
            dataframes = [pd.DataFrame(correlation)]
            name_axs = [axs]
            x_labels = "Feature attribution extraction method"
            axs.set_ylabel("Pearson correlation")

        for dataframe, name_ax, x_label in zip(dataframes, name_axs, x_labels):
            sns.boxplot(data=dataframe, ax=name_ax, showfliers=False)
            sns.stripplot(data=dataframe, color="0.25", s=3, ax=name_ax)
            x = [-0.5, dataframe.shape[1] - 0.5]
            y = [random_results[0]] * 2
            y_error_min = [random_results[0] - random_results[1]] * 2
            y_error_max = [random_results[0] + random_results[1]] * 2
            name_ax.set_xlabel(x_label)
            name_ax.plot(x, y, '--',
                         label=f"Random correlation\n({random_results[0]:.3f} +- {random_results[1]:.3f})")
            name_ax.fill_between(x, y_error_min, y_error_max, alpha=0.3)
            name_ax.grid(axis='y')
            name_ax.legend()

        if channel_split:
            axs[2][1].set_xlabel(axs[2][1].get_xlabel() + "\nFeature attribution extraction method")

        plt.tight_layout()
        plt.savefig(
            fname=get_comparison_plot_filename(
                model_handlers=[attribution_handler],
                keywords=[attribution_type] + (['subset'] if methods_subset is not None else []),
                correlation_method=correlation_method,
                channels=len(method_correlation) > 1,
                combined=display_combined
            ), dpi=300
        )
        plt.clf()


def plot_method_correlation_comparison_all_models_subset(model_handlers, display_combined=False):
    """
    Creates a plot as follows:
        * If no channel information is included in the correlation information gathered from a model handler:
            - # subplots = # model names
            - Plot organization: 1 x {# subplots}
            - Content:
                + 1 subplot / model name = 1 column / model name
                + Each subplot contains one boxplot for each method in 'methods_subset'
        * If channel information included:
            - # sub plots = # model names x # methods
            - Plot organization: {# subplots} x {# methods}
            - Content:
                + 1 row / model name
                + Each row contains 1 plot / method in 'methods_subset'
                + Each subplot contains one boxplot for each channel (there are 4 channels)
    Parameters
    ----------
    model_handlers
    display_combined

    Returns
    -------

    """
    methods_subset = ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist']
    correlations = [None, None, None]
    random_correlations = [None, None, None]
    names = ["ImRex pairwise", "ImRex AA", "TITAN AA"]
    for attribution_handler in model_handlers:
        if isinstance(attribution_handler, ImrexAttributionsHandler):
            correlations[0] = attribution_handler.get_correlation()
            correlations[1] = attribution_handler.get_aa_correlation()
            random_correlations[0] = attribution_handler.get_random_correlation()
            random_correlations[1] = attribution_handler.get_aa_random_correlation()
        elif isinstance(attribution_handler, TITANAttributionsHandler):
            correlations[2] = attribution_handler.get_aa_correlation()
            random_correlations[2] = attribution_handler.get_aa_random_correlation()
        else:
            raise TypeError(
                f"attribution_handler of wrong type, got {type(attribution_handler)} but expected "
                f"{ImrexAttributionsHandler} or {TITANAttributionsHandler}"
            )

    # In absence of a model handler: one of the entries is None --> remove None
    zipped_correlations = [x for x in list(zip(correlations, random_correlations, names)) if None not in x]
    correlations = [x[0] for x in zipped_correlations]
    random_correlations = [x[1] for x in zipped_correlations]
    names = [x[2] for x in zipped_correlations]

    # 2 possibilities:
    #   1) no channels  --> 1 image containing 1 plot with 4 methods for each model
    #   2) channels     --> 1 image containing 4 plots (one for each method) with 4 channels for each model

    if '_' not in list(correlations[0].keys())[0]:
        # case 1: 1 row with as many plots as there are items in "names"
        f, axs = plt.subplots(nrows=1, ncols=len(names), figsize=((10 / 3) * len(names), 3), sharey=True)
        has_channels = False
    else:
        # case 2
        f, axs = plt.subplots(
            nrows=len(names),  # 1 plot / model handler
            ncols=len(methods_subset),
            figsize=((10 / 3) * len(methods_subset), (10 / 3) * len(names)),
            sharex=True,
            sharey=True
        )
        has_channels = True

    for i, (ax, correlation_result, random_result, name) in enumerate(zip(axs, correlations, random_correlations, names)):
        # first create a dictionary
        #       * case 1 (no channels): maps method name to correlation values
        #       * case 2 (channels): maps method name to a map with channels that map to correlation values
        method_correlation = {}
        for pdb_id, pdb_corr in correlation_result.items():
            for method, method_corr in pdb_corr.items():
                if method not in methods_subset:
                    continue

                if has_channels:  # case 2
                    channel = "Combi" if "combined" in pdb_id.split("_")[1] else f'Ch {pdb_id.split("_")[1]}'
                    if not display_combined and channel == "Combi":
                        continue
                    if method not in method_correlation:
                        method_correlation[method] = {}

                    if channel in method_correlation[method]:
                        method_correlation[method][channel].append(-method_corr)
                    else:
                        method_correlation[method][channel] = [-method_corr]
                else:  # case 1
                    if method in method_correlation:
                        method_correlation[method].append(-method_corr)
                    else:
                        method_correlation[method] = [-method_corr]
        method_correlation = {
            "SHAP" if k == 'SHAP BGdist' else "IG" if k == "VanillaIG" else k: v
            for k, v in method_correlation.items()
        }

        # then create one or more Pandas dataframes and create plots from them
        if not has_channels:
            # case 1: one frame
            dataframes = [pd.DataFrame(method_correlation)]
            name_axs = [ax]

            axs[0].set_ylabel('Pearson correlation')
            axs[1].set_xlabel('Feature attribution extraction method')
        else:
            # case 2: one frame per method
            dataframes = [pd.DataFrame(method_correlation[method]) for method in method_correlation]
            name_axs = [ax[0], ax[1], ax[2], ax[3]]

            axs[i][0].set_ylabel(f'Pearson correlation\n{name}')
            if i == 1:
                for j, method in enumerate(methods_subset):
                    axs[i][j].set_xlabel(
                        "SHAP" if method == 'SHAP BGdist' else "IG" if method == "VanillaIG" else method
                    )

        # finally, add collected data to the plot
        for dataframe, name_ax in zip(dataframes, name_axs):
            sns.boxplot(data=dataframe, ax=name_ax, showfliers=False)
            sns.stripplot(data=dataframe, color="0.25", s=3, ax=name_ax)
            x = [-0.5, dataframe.shape[1] - 0.5]
            y = [random_result[0]] * 2
            y_error_min = [random_result[0] - random_result[1]] * 2
            y_error_max = [random_result[0] + random_result[1]] * 2
            name_ax.plot(x, y, '--', label=f"Random correlation\n({random_result[0]:.3f} +- {random_result[1]:.3f})")
            name_ax.fill_between(x, y_error_min, y_error_max, alpha=0.3)
            name_ax.grid(axis='y')
            name_ax.legend()

    plt.tight_layout()
    name = '_'.join([model_handler.name for model_handler in model_handlers])
    plt.savefig(
        fname=get_comparison_plot_filename(
            model_handlers=model_handlers,
            correlation_method='pearson',
            channels=len(correlations) > 1,
            combined=display_combined
        )
    )
    plt.clf()


def plot_aa_pearson_correlation_model_comparison(
        model_handlers,
        random_index,
        method,
        save_post,
        display_combined=False
):
    """
    Compare epitope with CDR3 for the listed models (model_handlers)

    Parameters
    ----------
    model_handlers: list of model handlers
    random_index: random index for the random correlation calculation
    method: the method for which to create the plot
    save_post:
    display_combined

    Returns
    -------

    """
    model_correlation_ps = []
    model_correlation = []
    names_ps = []
    names = []
    channel_split = False
    for model_handler in model_handlers:
        # per sequence
        correlation = model_handler.get_aa_correlation_ps()
        method_correlation_ep = []
        method_correlation_cdr3 = []
        display_names = []
        pdb_ids = []
        for pdb_id, methods in correlation.items():
            method_correlation_ep.append(-methods[method][0])
            method_correlation_cdr3.append(-methods[method][1])
            pdb_ids.append(pdb_id)

        if '_' in pdb_ids[0]:
            channel_split = True
            zipped_correlations = list(zip(method_correlation_ep, method_correlation_cdr3, pdb_ids))
            method_correlation_ep = []
            method_correlation_cdr3 = []
            for i in range(4):
                method_correlation_ep.append(
                    [element[0] for element in zipped_correlations
                     if "combined" not in element[2] and int(element[2].split('_')[1]) == i]
                )
                method_correlation_cdr3.append(
                    [element[1] for element in zipped_correlations
                     if "combined" not in element[2] and int(element[2].split('_')[1]) == i]
                )
                display_names.append(f'Ch {i}')
            if display_combined:
                method_correlation_ep.append(
                    [element[0] for element in zipped_correlations
                     if "combined" in element[2]]
                )
                method_correlation_cdr3.append(
                    [element[1] for element in zipped_correlations
                     if "combined" in element[2]]
                )
                display_names.append(f'Combi')
        else:
            method_correlation_ep = [method_correlation_ep]
            method_correlation_cdr3 = [method_correlation_cdr3]
            display_names.append(model_handler.display_name)

        for i in range(len(display_names)):
            model_correlation_ps.append(method_correlation_ep[i])
            model_correlation_ps.append(method_correlation_cdr3[i])
            if save_post == "all":
                names_ps.append(display_names[i] + ' epitope')
                names_ps.append(display_names[i] + ' CDR3')
            else:
                names_ps.append(display_names[i] + '\nepitope')
                names_ps.append(display_names[i] + '\nCDR3')

        # full
        correlation = model_handler.get_aa_correlation()
        method_correlation = []
        if '_' in pdb_ids[0]:
            for i in range(4):
                for pdb_id, methods in correlation.items():
                    if "combined" not in pdb_id and int(pdb_id.split('_')[1]) == i:
                        method_correlation.append(-methods[method])
                model_correlation.append(method_correlation)
                names.append(f'Ch {i}')
                method_correlation = []
            if display_combined:
                for pdb_id, methods in correlation.items():
                    if "combined" in pdb_id:
                        method_correlation.append(-methods[method])
                model_correlation.append(method_correlation)
                names.append(f'Combi')
        else:
            for pdb_id, methods in correlation.items():
                method_correlation.append(-methods[method])
            model_correlation.append(method_correlation)
            names.append(model_handler.display_name)

    fig_size = (14, 6) if channel_split else (6.4 / 1.05, 4.8 / 1.05)
    per_seq_palette = [c for c in sns.color_palette("deep", 5) for _ in range(2)]
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=fig_size,
        gridspec_kw={'width_ratios': [1, 2]},
        sharey=True
    )
    sns.boxplot(data=model_correlation, ax=axs[0], showfliers=False)
    sns.stripplot(data=model_correlation, ax=axs[0], color='0.25', s=3)
    sns.boxplot(data=model_correlation_ps, ax=axs[1], palette=per_seq_palette, showfliers=False)
    sns.stripplot(data=model_correlation_ps, ax=axs[1], color='0.25', s=3)

    axs[0].set_ylabel('Pearson correlation')
    axs[0].set_xticklabels([split_line(n, 12 if save_post == 'all' else 15) for n in names])
    axs[1].set_xticklabels([split_line(n, 12 if save_post == 'all' else 15) for n in names_ps])

    re_ps = model_handlers[random_index].get_aa_random_correlation_ps()
    re = model_handlers[random_index].get_aa_random_correlation()

    re_ps_ep_range = (re_ps[0][0] + re_ps[0][1], re_ps[0][0] - re_ps[0][1])
    re_ps_cdr3_range = (re_ps[1][0] + re_ps[1][1], re_ps[1][0] - re_ps[1][1])
    re_range = (re[0] + re[1], re[0] - re[1])
    x_ps = [-0.5, len(names_ps) - 0.5]
    x = [-0.5, len(names) - 0.5]
    # print(f"Random correlation epitope ({re_ps[0][0]} +- {re_ps[0][1]})")
    # print()

    sep = ' ' if save_post == 'all' else '\n'
    axs[1].plot(x_ps, [re_ps[0][0]] * 2, '--',
                label=f"Random correlation epitope{sep}({round(re_ps[0][0], 3):.3f} +- {round(re_ps[0][1], 3)})")
    axs[1].plot(x_ps, [re_ps[1][0]] * 2, '--',
                label=f"Random correlation CDR3{sep}({round(re_ps[1][0], 3):.3f} +- {round(re_ps[1][1], 3)})")

    axs[1].fill_between(x_ps, re_ps_ep_range[0], re_ps_ep_range[1], alpha=0.3)
    axs[1].fill_between(x_ps, re_ps_cdr3_range[0], re_ps_cdr3_range[1], alpha=0.3)

    axs[0].plot(x, [re[0]] * 2, '--', label=f"Random correlation{sep}({round(re[0], 3):.3f} +- {round(re[1], 3)})")

    axs[0].fill_between(x, re_range[0], re_range[1], alpha=0.3)
    if save_post == "all":
        axs[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.3), loc="lower right", borderaxespad=0)
        axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.3), loc="lower right", borderaxespad=0)
    else:
        axs[1].legend(fontsize=9, bbox_to_anchor=(0, 1.02, 1, 0.3), loc="lower right", borderaxespad=0)
        axs[0].legend(fontsize=8, bbox_to_anchor=(0, 1.02, 1, 0.3), loc="lower right", borderaxespad=0)
    axs[0].set_xlabel('Model')
    axs[1].set_xlabel('Model + sequence')

    fig.tight_layout()
    axs[0].grid(axis='y')
    axs[1].grid(axis='y')

    plt.savefig(
        fname=get_comparison_plot_filename(
            model_handlers=model_handlers,
            method=method,
            combined=display_combined,
            channels=channel_split
        ),
        dpi=300
    )
    plt.close()


def plot_sample_details(model_handlers, method, save_post, dist_i=0, display_combined=False):
    """
    Creates a feature attribution heat map for each pdb complex.

    Parameters
    ----------
    model_handlers
    method
    save_post
    dist_i
    display_combined

    Returns
    -------

    """
    model_attributions = [model.get_aa_norm_attributions() for model in model_handlers]
    model_names = [model.display_name for model in model_handlers]

    if '_' in list(model_attributions[0].keys())[0]:
        for i in range(len(model_attributions)):
            extracted_model_attributions = [dict(), dict(), dict(), dict()]
            if not display_combined:
                model_names = model_names[:i] + [model_names[i] + f' Ch {j}' for j in range(4)] + model_names[i + 1:]
            else:
                model_names = model_names[:i] + [model_names[i] + f' Ch {j}' for j in range(4)] + ['Combined'] + \
                              model_names[i + 1:]
                extracted_model_attributions.append(dict())

            for key, value in model_attributions[i].items():
                if 'combined' in key:
                    if display_combined:
                        extracted_model_attributions[-1][key.split('_')[0]] = value
                else:
                    extracted_model_attributions[int(key.split('_')[1])][key.split('_')[0]] = value
            model_attributions = model_attributions[:i] + extracted_model_attributions + model_attributions[i + 1:]

    sequences = model_handlers[dist_i].get_sequences()
    distances = model_handlers[dist_i].get_aa_norm_distances()
    # plt.figure(figsize=(10 / 1.3, 3 / 1.3))
    for pdb_id, dist in distances.items():
        if pdb_id not in model_attributions[0].keys():
            continue
        ep_len = len(sequences[pdb_id][0])
        heatmap = []
        names = []
        for model_attribution, name in zip(model_attributions, model_names):
            attribution = model_attribution[pdb_id][method]
            heatmap.append(np.concatenate((attribution[:ep_len], [-1], attribution[ep_len:])))
            names.append(name)
        heatmap.append(np.concatenate((dist[:ep_len], [-1], dist[ep_len:])))
        heatmap = np.array(heatmap)
        heatmap = np.ma.masked_where(heatmap == -1, heatmap)
        names.append('Residue proximity')
        plt.gcf().set_size_inches(10 / 1.3, 3 / 1.3)
        grid = plt.imshow(heatmap, cmap='Greys')
        plt.xticks(list(range(ep_len)) + list(range(ep_len + 1, ep_len + 1 + len(sequences[pdb_id][1]))),
                   sequences[pdb_id][0] + sequences[pdb_id][1])
        plt.yticks(list(range(len(heatmap))), names)
        plt.colorbar(grid, orientation='horizontal', pad=0.2, label=f"{method} feature attribution")
        plt.tight_layout()
        name = '_'.join([model_handler.name for model_handler in model_handlers])
        plt.savefig(f'output/plots/sample_details/{name}_{method}_{pdb_id}_{save_post}{"_combined" if display_combined else ""}'
                    f'.png', dpi=300, bbox_inches='tight')
        plt.clf()
        print(f'output/plots/sample_details/{name}_{method}_{pdb_id}_{save_post}{"_combined" if display_combined else ""}'
              f'.png')


def plot_TITAN_methods_sample_details(titan_handler, imrexhandler, subset):
    attributions = titan_handler.get_aa_norm_attributions()
    distances = imrexhandler.get_aa_norm_distances()
    sequences = imrexhandler.get_sequences()
    # plt.figure(figsize=(10 / 1.3, 3 / 1.3))
    for pdb_id, pdb_attributions in attributions.items():
        dist = distances[pdb_id]
        ep_len = len(sequences[pdb_id][0])
        heatmap = []

        for method in subset:
            heatmap.append(np.concatenate((pdb_attributions[method][:ep_len], [-1], pdb_attributions[method][ep_len:])))

        names = ["SHAP" if k == 'SHAP BGdist' else "IG" if k == "VanillaIG" else k for k in subset]

        heatmap.append(np.concatenate((dist[:ep_len], [-1], dist[ep_len:])))
        heatmap = np.array(heatmap)
        heatmap = np.ma.masked_where(heatmap == -1, heatmap)
        names.append('Residue proximity')
        plt.gcf().set_size_inches(10 / 1.3, 5 / 1.3)
        grid = plt.imshow(heatmap, cmap='Greys')
        plt.xticks(list(range(ep_len)) + list(range(ep_len + 1, ep_len + 1 + len(sequences[pdb_id][1]))),
                   sequences[pdb_id][0] + sequences[pdb_id][1])
        plt.yticks(list(range(len(heatmap))), names)
        plt.colorbar(grid, orientation='horizontal', pad=0.2, label=f"{titan_handler.display_name} feature attribution")
        plt.tight_layout()  # TODO: filename
        plt.savefig(f'output/plots/sample_details/{titan_handler.display_name}_{pdb_id}.png', dpi=300,
                    bbox_inches='tight')
        plt.clf()


def plot_2d_sample_attributions(model_handler, methods, save_name, display_combined=False):
    attributions = model_handler.get_norm_attributions()
    sequences = model_handler.get_sequences()
    distances = model_handler.get_norm_distances()
    channel_split = '_' in list(attributions.keys())[0]

    # gradient gray color bar
    fig = plt.figure(figsize=(6.4 / 2, 4.8 / 2))
    ax = fig.add_axes([0.05, 0.05, 0.07, 0.9])
    cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical', cmap=plt.get_cmap('Greys'))
    cb.set_label(f"Feature attribution / residue proximity")
    plt.savefig('output/plots/colorbar_grey.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    for pdb_id, attribution in attributions.items():
        if channel_split and ('combined' in pdb_id or int(pdb_id.split('_')[1]) > 0):
            continue
        ep, cdr3 = sequences[pdb_id.split('_')[0]]
        # fig, axs = plt.subplots(1, len(methods) + 1, sharey=True, figsize=(12 / 1.3, 4.8 / 1.3))
        rows = 4 if channel_split and len(methods) > 1 else 1
        columns = 5 if channel_split and len(methods) == 1 else len(methods) + 1
        if display_combined and len(methods) == 1:
            columns += 1
        fig, axs = plt.subplots(
            nrows=rows,
            ncols=columns,
            sharex=True,
            sharey=True,
            figsize=(2 * columns, 3.75 * rows)
        )

        if channel_split and len(methods) > 1:
            for row in range(4):
                channel_attribution = attributions[pdb_id.replace('_0', f'_{row}')]
                for column, method in enumerate(methods):
                    att = channel_attribution[method]
                    grid = axs[row][column].imshow(att, cmap='Greys', vmin=0, vmax=1)
                    axs[row][column].set_xticks(list(range(len(ep))))
                    axs[row][column].set_xticklabels(ep)
                    if row == 0:
                        axs[row][column].set_title(
                            "SHAP" if method == 'SHAP BGdist' else "IG" if method == "VanillaIG" else method
                        )
                    if column == 0:
                        axs[row][column].set_ylabel(f'CDR3 Ch {row}')

                dist = distances[pdb_id.split('_')[0]]
                axs[row][-1].imshow(dist, cmap='Greys', vmin=0, vmax=1)
                axs[row][-1].set_xticks(list(range(len(ep))))
                axs[row][-1].set_xticklabels(ep)
                if row == 0:
                    axs[row][-1].set_title('Pairwise\nresidue proximity')

            for i in range(columns):
                axs[0][i].set_xlabel('epitope')

        elif channel_split:
            method = methods[0]
            for row in range(4):
                channel_attribution = attributions[pdb_id.replace('_0', f'_{row}')]
                att = channel_attribution[method]
                grid = axs[row].imshow(att, cmap='Greys', vmin=0, vmax=1)
                axs[row].set_xticks(list(range(len(ep))))
                axs[row].set_xticklabels(ep)
                axs[row].set_title(
                    f'{"SHAP" if method == "SHAP BGdist" else "IG" if method == "VanillaIG" else method} Ch {row}'
                )

            if display_combined:
                channel_attribution = attributions[pdb_id.replace('_0', f'_combined')]
                att = channel_attribution[method]
                grid = axs[-2].imshow(att, cmap='Greys', vmin=0, vmax=1)
                axs[-2].set_xticks(list(range(len(ep))))
                axs[-2].set_xticklabels(ep)
                axs[-2].set_title(
                    f'{"SHAP" if method == "SHAP BGdist" else "IG" if method == "VanillaIG" else method} Combined'
                )

            dist = distances[pdb_id.split('_')[0]]
            axs[-1].imshow(dist, cmap='Greys', vmin=0, vmax=1)
            axs[-1].set_xticks(list(range(len(ep))))
            axs[-1].set_xticklabels(ep)
            axs[-1].set_title('Pairwise\nresidue proximity')

            for i in range(columns):
                axs[i].set_xlabel('epitope')

            axs[0].set_ylabel('CDR3')

        else:
            for column, method in enumerate(methods):
                att = attribution[method]
                grid = axs[column].imshow(att, cmap='Greys', vmin=0, vmax=1)
                axs[column].set_xticks(list(range(len(ep))))
                axs[column].set_xticklabels(ep)
                axs[column].set_title("SHAP" if method == 'SHAP BGdist' else "IG" if method == "VanillaIG" else method)
                axs[column].set_xlabel('epitope')

            dist = distances[pdb_id]
            axs[-1].imshow(dist, cmap='Greys', vmin=0, vmax=1)
            axs[-1].set_xticks(list(range(len(ep))))
            axs[-1].set_xticklabels(ep)
            axs[-1].set_title('Pairwise\nresidue proximity')
            axs[-1].set_xlabel('epitope')

            axs[0].set_ylabel('CDR3')

        plt.yticks(list(range(len(cdr3))), cdr3)
        plt.tight_layout(pad=0.1)
        plt.savefig(
            fname=get_sample_2d_details_filename(
                model_handler=model_handler,
                pdb_id=pdb_id.split("_")[0],
                channels=channel_split,
                method=save_name,
                combined=len(methods) == 1 and display_combined
            ),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)


def plot_2d_ImRex_input(image_path, model_handler):
    input_imgs = {}
    for f in sorted(os.listdir(image_path)):
        if not f.endswith('.pkl'):
            continue
        input_imgs[f[:-4]] = tf.cast(tf.convert_to_tensor(pickle.load(open(image_path + f, 'rb'))), tf.float32)

    sequences = model_handler.get_sequences()
    for pdb_id, input_img in input_imgs.items():
        ep, cdr3 = sequences[pdb_id]
        fig, axs = plt.subplots(1, 5, sharey=True, figsize=(9, 3))
        cmyk_array = (input_img.numpy() * 255).astype(np.uint8)
        cmyk_array = imrex_remove_padding(cmyk_array, len(cdr3), len(ep))
        for i, feat in enumerate(["Hydrophobicity", "Isoelectric point", "Mass", "Hydrophilicity"]):
            channel = cmyk_array[:, :, i]
            channels = []
            for j in range(i):
                channels.append(np.zeros(channel.shape, dtype=np.uint8))
            channels.append(channel)
            for j in range(i + 1, 4):
                channels.append(np.zeros(channel.shape, dtype=np.uint8))
            axs[i].imshow(Image.fromarray(np.stack(channels, axis=2), mode='CMYK'))
            axs[i].set_xticks(list(range(len(ep))))
            axs[i].set_xticklabels(ep)
            axs[i].set_title(feat)

        axs[4].imshow(Image.fromarray(cmyk_array, mode='CMYK'))
        axs[4].set_xticks(list(range(len(ep))))
        axs[4].set_xticklabels(ep)
        axs[4].set_title("Combined")

        plt.yticks(list(range(len(cdr3))), cdr3)
        axs[0].set_ylabel('CDR3')
        axs[2].set_xlabel('epitope')
        plt.tight_layout()
        plt.savefig(f'output/plots/{model_handler.name}_input_features/{pdb_id}.png', dpi=300)
        plt.close()


def plot_positional_average(model_handlers, method, save_post, dist_i=0, display_combined=False):
    """
    Creates a feature attribution heat map for each pdb complex.

    Parameters
    ----------
    model_handlers
    method
    save_post
    dist_i
    display_combined

    Returns
    -------

    """
    model_attributions = {model.display_name: model.get_aa_norm_attributions() for model in model_handlers}
    sequences = model_handlers[dist_i].get_sequences()
    distances = model_handlers[dist_i].get_aa_norm_distances()

    channel_split = False
    if '_' in list(model_attributions[list(model_attributions.keys())[0]].keys())[0]:
        channel_split = True
        extracted_model_attributions_dict = dict()
        for model_name, attributions in model_attributions.items():
            extracted_model_attributions_list = [dict(), dict(), dict(), dict()]
            if display_combined:
                extracted_model_attributions_list.append(dict())
            for key, value in attributions.items():
                if "combined" in key:
                    if display_combined:
                        extracted_model_attributions_list[-1][key.split('_')[0]] = value
                else:
                    extracted_model_attributions_list[int(key.split('_')[1])][key.split('_')[0]] = value
            for i in range(min(4, len(extracted_model_attributions_list))):
                extracted_model_attributions_dict[f'{model_name} Ch {i}'] = extracted_model_attributions_list[i]
            if display_combined:
                extracted_model_attributions_dict['Combined'] = extracted_model_attributions_list[-1]
        model_attributions = extracted_model_attributions_dict

    pdb_subset = list(model_attributions[list(model_attributions.keys())[0]].keys())

    eps = [v[0] for k, v in sequences.items() if k in pdb_subset]
    cdr3s = [v[1] for k, v in sequences.items() if k in pdb_subset]

    max_ep = len(max(eps, key=len))
    max_cdr3 = len(max(cdr3s, key=len))

    pos_distances_ep = []
    pos_distances_cdr3 = []
    pos_model_attributions = {k: [] for k, v in model_attributions.items()}
    for pdb_id, dist in distances.items():
        if pdb_id not in model_attributions[list(model_attributions.keys())[0]].keys():
            continue

        ep = sequences[pdb_id][0]
        pos_distances_ep.append(aa_add_padding(dist[:len(ep)], max_ep))
        pos_distances_cdr3.append(aa_add_padding(dist[len(ep):], max_cdr3))

        for model_name, model_attribution in model_attributions.items():
            attribution = model_attribution[pdb_id][method]
            pos_model_attributions[model_name].append(
                (aa_add_padding(attribution[:len(ep)], max_ep), aa_add_padding(attribution[len(ep):], max_cdr3)))
    heatmap = []
    names = []
    for name, pos_attributions in pos_model_attributions.items():
        pos_attributions_ep = [a[0] for a in pos_attributions]
        pos_attributions_cdr3 = [a[1] for a in pos_attributions]
        heatmap.append(np.concatenate((np.nanmean(pos_attributions_ep, 0), [-1], np.nanmean(pos_attributions_cdr3, 0))))
        names.append(name)
    heatmap.append(np.concatenate((np.nanmean(pos_distances_ep, 0), [-1], np.nanmean(pos_distances_cdr3, 0))))
    names.append('Residue proximity')
    plt.gcf().set_size_inches(10 / 1.3, (4 if display_combined else 3) / 1.3)
    heatmap = np.array(heatmap)
    heatmap = np.ma.masked_where(heatmap == -1, heatmap)
    grid = plt.imshow(heatmap, cmap='Greys')
    plt.xticks(list(range(max_ep)) + list(range(max_ep + 1, max_ep + 1 + max_cdr3)),
               ['$\mathregular{e_{' + str(i + 1) + '}}$' for i in range(max_ep)] +
               ['$\mathregular{c_{' + str(i + 1) + '}}$' for i in range(max_cdr3)])
    plt.yticks(list(range(len(heatmap))), names)
    plt.colorbar(grid, orientation='horizontal', pad=0.2, label=f"{method} feature attribution")
    plt.clim(0, 1)
    plt.tight_layout()
    plt.savefig(
        fname=get_comparison_plot_filename(
            model_handlers=model_handlers,
            method=method,
            channels=channel_split,
            combined=display_combined
        ),
        dpi=300,
        bbox_inches='tight'
    )
    plt.clf()


def plot_random_positional_average_diff(
        model_handlers,
        method,
        save_post,
        display_combined=False
):
    """
    Creates a feature attribution heat map for each pdb complex.

    Parameters
    ----------
    model_handlers
    method
    save_post
    display_combined

    Returns
    -------

    """

    # get the model attributions for both handlers
    model_attributions = {
        model.display_name: model.get_aa_norm_attributions()
        for model in model_handlers
    }
    # sequences and distances are the same for both handlers
    sequences = model_handlers[0].get_sequences()
    distances = model_handlers[0].get_aa_norm_distances()

    # if channels were split, tune the model_attributions dictionary
    channel_split = False
    if '_' in list(model_attributions[list(model_attributions.keys())[0]].keys())[0]:
        channel_split = True
        model_attributions = {
            model_name: {
                **{
                    f'Ch {i}': {
                        key.replace(f'_{i}', ''): value for key, value in model_attributions[model_name].items()
                        if key.endswith(f'_{i}')
                    }
                    for i in range(4)
                },
                **{
                    f'Combined': {
                        key.replace(f'_combined', ''): value for key, value in model_attributions[model_name].items()
                        if key.endswith('_combined')
                    }
                }
            }
            for model_name in [handler.display_name for handler in model_handlers]
        }

    # get the pdbs that occur in model attributions
    pdb_list = list(model_attributions[model_handlers[0].display_name]['Ch 0'].keys())

    # obtain the maximum lengths of eps and cdr3s
    max_ep = len(max([v[0] for k, v in sequences.items() if k in pdb_list], key=len))
    max_cdr3 = len(max([v[1] for k, v in sequences.items() if k in pdb_list], key=len))

    # extract distances (overall) and attributions (per model handler)
    pos_distances_ep = []
    pos_distances_cdr3 = []
    pos_model_attributions = {
        k: {
            ch: {'ep': [], 'cdr3': []} for ch in model_attributions[model_handlers[0].display_name].keys()
        }
        for k in model_attributions.keys()
    }
    for pdb_id, dist in distances.items():
        if pdb_id not in pdb_list:
            continue
        ep = sequences[pdb_id][0]

        # extract distances
        pos_distances_ep.append(aa_add_padding(dist[:len(ep)], max_ep))
        pos_distances_cdr3.append(aa_add_padding(dist[len(ep):], max_cdr3))

        # extract attributions for each model handler
        for model_name, model_attribution in model_attributions.items():
            for channel, channel_attribution in model_attribution.items():
                attribution = channel_attribution[pdb_id][method]
                pos_model_attributions[model_name][channel]['ep'].append(aa_add_padding(attribution[:len(ep)], max_ep))
                pos_model_attributions[model_name][channel]['cdr3'].append(aa_add_padding(attribution[len(ep):], max_cdr3))

    # create heat maps
    heatmaps = {
        model.display_name: []
        for model in model_handlers
    }
    names = []
    for model_name, pos_attributions in pos_model_attributions.items():
        for channel, channel_attributions in pos_attributions.items():
            heatmaps[model_name].append(
                np.concatenate(
                    (np.nanmean(channel_attributions['ep'], 0), [-1], np.nanmean(channel_attributions['cdr3'], 0))
                )
            )
            names.append(channel)

    # calculate differences
    scramb_ep_handler = [handler for handler in model_handlers if 'scrambled_eps' in handler.name][0]
    scramb_tcr_handler = [handler for handler in model_handlers if 'scrambled_tcrs' in handler.name][0]
    imrex_attributions_handler = [handler for handler in model_handlers if 'nocdr3dup' in handler.name][0]

    heatmaps['mean'] = []
    for i in range(5):
        arr1 = heatmaps[scramb_ep_handler.display_name][i]
        arr2 = heatmaps[scramb_tcr_handler.display_name][i]
        heatmaps['mean'].append(
            (arr1 + arr2) / 2
        )

    heatmaps['diff_1'] = []  # scrambled eps - scrambled tcrs
    heatmaps['diff_2'] = []  # scrambled eps - nocdr3dup
    heatmaps['diff_3'] = []  # scrambled tcrs - nocdr3dup
    for i in range(5):
        arr1 = heatmaps[scramb_ep_handler.display_name][i]
        arr2 = heatmaps[scramb_tcr_handler.display_name][i]
        arr3 = heatmaps[imrex_attributions_handler.display_name][i]
        heatmaps['diff_1'].append(np.abs(arr1 - arr2))
        heatmaps['diff_2'].append(np.abs(arr1 - arr3))
        heatmaps['diff_3'].append(np.abs(arr2 - arr3))

    heatmaps['sub'] = []
    heatmaps['add'] = []

    for i in range(5):
        mean_arr = heatmaps['mean'][i]
        imrex_arr = heatmaps[imrex_attributions_handler.display_name][i]
        heatmaps['add'].append(
            imrex_arr + 1 - mean_arr
        )
        heatmaps['sub'].append(
            imrex_arr - mean_arr
        )
        # min_val = min(min_val, np.min(heatmaps['add'][-1]))
        # max_val = max(max_val, np.max(heatmaps['add'][-1]))
        heatmaps['add'][-1][max_ep] = 0
    # for i in range(5):
    #     heatmaps['diff'][i] = heatmaps['diff'][i] * (1 / max_val)

    for name, heatmap in heatmaps.items():
        heatmap.append(np.concatenate((np.nanmean(pos_distances_ep, 0), [-1], np.nanmean(pos_distances_cdr3, 0))))

    names = sorted(list(set(names)))
    names.append('Residue proximity')
    for name, heatmap in heatmaps.items():
        if 'ImRex' in name:
            continue

        # scale
        if 'diff' not in name:
            min_val, max_val = math.inf, -math.inf
            for i in range(5):
                min_val = min(min_val, np.min(heatmap[i]))
                max_val = max(max_val, np.max(heatmap[i]))
            print(max_val, min_val)
            max_val += -min_val
            print(max_val, min_val)
            for i in range(5):
                heatmap[i] = -min_val + heatmap[i] * (1 / max_val)
            min_val, max_val = math.inf, -math.inf
            for i in range(5):
                min_val = min(min_val, np.min(heatmap[i]))
                max_val = max(max_val, np.max(heatmap[i]))
                heatmaps['add'][-1][max_ep] = 0
            print(max_val, min_val)

        plt.gcf().set_size_inches(10 / 1.3, (4 if display_combined else 3) / 1.3)
        heatmap = np.array(heatmap)
        heatmap = np.ma.masked_where(heatmap == -1, heatmap)
        grid = plt.imshow(heatmap, cmap='Greys')
        plt.xticks(list(range(max_ep)) + list(range(max_ep + 1, max_ep + 1 + max_cdr3)),
                   ['$\mathregular{e_{' + str(i + 1) + '}}$' for i in range(max_ep)] +
                   ['$\mathregular{c_{' + str(i + 1) + '}}$' for i in range(max_cdr3)])
        plt.yticks(list(range(len(heatmap))), names)
        plt.colorbar(grid, orientation='horizontal', pad=0.2, label=f"{method} feature attribution")
        plt.clim(0, 1)
        plt.tight_layout()
        if name == 'diff_1':
            display_handlers = [scramb_ep_handler, scramb_tcr_handler]
            name = 'diff'
        elif name == 'diff_2':
            display_handlers = [scramb_ep_handler, imrex_attributions_handler]
            name = 'diff'
        elif name == 'diff_3':
            display_handlers = [scramb_tcr_handler, imrex_attributions_handler]
            name = 'diff'
        else:
            display_handlers = model_handlers
        print(name)
        plt.savefig(
            fname=get_comparison_plot_filename(
                model_handlers=display_handlers,
                method=method,
                channels=channel_split,
                combined=display_combined,
                keywords=[name]
            ),
            dpi=300,
            bbox_inches='tight'
        )
        plt.clf()


def random_function_2():
    import inspect
    print(inspect.currentframe().f_back.f_code.co_name)

def random_function(val):
    # import inspect
    # print(inspect.currentframe().f_code.co_name)
    random_function_2()
    exit()


def main():
    # random_function(val=0)
    imrex_attributions_handler = ImrexAttributionsHandler(
        name="imrex_nocdr3dup",
        display_name="ImRex",
        model_path="ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/iteration_2/"
                   "2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv-epoch20.h5",
        image_path="data/tcr3d_images/",
        save_folder="data",
    )

    imrex_scrambled_eps = ImrexAttributionsHandler(
        name="imrex_scrambled_eps",
        display_name="ImRex scrambled eps",
        model_path="ImRex/models/models/2022-11-30_13-55-56_scrambled_eps/iteration_0/"
                   "2022-11-30_13-55-56_scrambled_eps-epoch20.h5",
        image_path="data/tcr3d_images/",
        save_folder="data"
    )

    imrex_scrambled_tcrs = ImrexAttributionsHandler(
        name="imrex_scrambled_tcrs",
        display_name="ImRex scrambled TCRs",
        model_path="ImRex/models/models/2022-12-01_09-52-59_scrambled_tcrs/iteration_0/"
                   "2022-12-01_09-52-59_scrambled_tcrs-epoch20.h5",
        image_path="data/tcr3d_images/",
        save_folder="data"
    )

    titan_on_imrex_data_handler = TITANAttributionsHandler(
        name="titan_nocdr3dup",
        display_name="TITAN on ImRex data",
        model_path="TITAN/models/nocdr3dup_epgrouped5cv_paperparams_smallpad/cv2/"
                   "nocdr3dup_epgrouped5cv_paperparams_smallpad_2/",
        tcrs_path='data/tcrs.csv',
        epitopes_path='data/epitopes.csv',
        data_path='data/tcr3d_titan_input.csv',
        save_folder='data'
    )

    # titan_strictsplit_handler = TITANAttributionsHandler(
    #     name="titan_strictsplit",
    #     display_name="TITAN",
    #     model_path="TITAN/models/titanData_strictsplit_nocdr3/cv5/titanData_strictsplit_nocdr3_5/",
    #     tcrs_path='data/tcrs.csv',
    #     epitopes_path='data/epitopes.csv',
    #     data_path='data/tcr3d_titan_input.csv',
    #     save_folder='data'
    # )
    sns.set_palette("deep")

    plot_random_positional_average_diff(
        model_handlers=[imrex_scrambled_eps, imrex_scrambled_tcrs, imrex_attributions_handler],
        method='SmoothGrad',
        save_post='imrex_channels',
        display_combined=True
    )

    save_post = 'imrex_channels'
    for model_handler in [imrex_attributions_handler, imrex_scrambled_eps, imrex_scrambled_tcrs]:
        model_handlers = [model_handler]
        print(model_handler.display_name)
        # model_handlers = [imrex_attributions_handler, titan_strictsplit_handler]

        # Fig 2: Correlation comparison of 4 extraction methods for ImRex pairwise, ImRex AA (and maybe TITAN)
        print('plot_method_correlation_comparison_all_models_subset')
        plot_method_correlation_comparison_all_models_subset(model_handlers=model_handlers)
        plot_method_correlation_comparison_all_models_subset(model_handlers=model_handlers, display_combined=True)
        # # TODO: this one should be finished now
        #
        # # Fig 3: Feature attributions for ImRex AA and TITAN and Residue proximity
        # print('plot_sample_details')
        # plot_sample_details(model_handlers=model_handlers, method='SmoothGrad', save_post=save_post)
        # plot_sample_details(model_handlers=model_handlers, method='SmoothGrad', save_post=save_post, display_combined=True)
        # # TODO: this one should be finished now

        # Fig 4: Average feature attributions for ImRex AA and TITAN and Residue proximity
        print('plot_positional_average')
        plot_positional_average(model_handlers=model_handlers, method='SmoothGrad', save_post=save_post)
        plot_positional_average(model_handlers=model_handlers, method='SmoothGrad', save_post=save_post,
                                display_combined=True)
        # TODO: this one should be finished now

        # Fig 5: Compare correlation between ImRex AA and TITAN and epitope/CDR3
        print('plot_aa_model_comparison')
        plot_aa_pearson_correlation_model_comparison(model_handlers=model_handlers, random_index=0, method='SmoothGrad', save_post=save_post)
        plot_aa_pearson_correlation_model_comparison(model_handlers=model_handlers, random_index=0, method='SmoothGrad', save_post=save_post,
                                                     display_combined=True)
        # TODO: this one should be finished now

        # Fig 6: Model performance
        # plot_model_performance()
        # TODO: nothing changes?

        # Fig S1: ImRex 2D feature encoding input
        # plot_2d_ImRex_input('data/tcr3d_images/', imrex_attributions_handler)
        # TODO: nothing changes?

        # Fig S2: Same as Fig 2 but with all extraction methods
        print('plot_method_correlation_comparison')
        plot_method_correlation_comparison(model_handler)
        plot_method_correlation_comparison(model_handler, display_combined=True)
        # plot_method_correlation_comparison(titan_strictsplit_handler)
        # TODO: this one should be finished now (TITAN may no work)

        # Fig S3 and S4: Detailed feature attributions for ImRex and TITAN
        print('plot_2d_sample_attributions')
        plot_2d_sample_attributions(
            model_handler=model_handler,
            methods=['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist'],
            save_name="4_methods"
        )
        for method in ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist']:
            print('plot_2d_sample_attributions', method)
            plot_2d_sample_attributions(
                model_handler=model_handler,
                methods=[method],
                save_name=method
            )
            plot_2d_sample_attributions(
                model_handler=model_handler,
                methods=[method],
                save_name=method,
                display_combined=True
            )
        # TODO: this one should be finished now

        # plot_TITAN_methods_sample_details(titan_strictsplit_handler, imrex_attributions_handler,
        #                                   ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist'])
        # TODO: TITAN, skipped this one

        # # Fig S5: Average feature attribution for the 3 models
        # plot_positional_average([imrex_attributions_handler, titan_strictsplit_handler, titan_on_imrex_data_handler],
        #                         'SmoothGrad', 'all')
        # TODO: same implementation as figure 4
        # # Fig S6: Like Fig 5 but on all models
        # plot_aa_model_comparison([imrex_attributions_handler, titan_strictsplit_handler, titan_on_imrex_data_handler], 0,
        #                          'SmoothGrad', 'all', (12.8 / 1.15, 4.8 / 1.15))
        # TODO: same implementation as figure 5

        # Fig S7: Spearman correlation for all extraction methods
        # plot_method_correlation_comparison(imrex_attributions_handler, correlation_method='spearman')
        # plot_method_correlation_comparison(titan_strictsplit_handler, correlation_method='spearman')

        print()
        print()


if __name__ == "__main__":
    main()
