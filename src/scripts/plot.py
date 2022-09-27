import os
import pickle

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


def plot_model_comparison(metrics, names, savename=None):
    if len(names) > 3:
        plt.figure(figsize=(6.4 / 1.3, 4.8 / 1.3))
    else:
        plt.figure(figsize=(3.2 / 1.3, 4.8 / 1.3))

    val_roc_aucs = [m['val_roc_auc'].to_numpy() for m in metrics]
    val_pr_aucs = [m['val_pr_auc'].to_numpy() for m in metrics]
    for metric, m_name, short_name in zip([val_roc_aucs, val_pr_aucs], ['ROC AUC', 'PR AUC'], ['roc', 'pr']):
        for name, results in zip(names, metric):
            print(name, m_name, round(np.mean(results), 3), '+-', round(np.std(results), 3))
        sns.boxplot(data=metric, showfliers=False)
        sns.stripplot(data=metric, color='0.25', s=4)
        plt.ylabel(m_name)
        plt.xticks(range(len(names)), [split_line(n, 13) for n in names])
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(
            f'output/plots/model_performance_comparison_{short_name}{"" if savename is None else "_" + savename}.png',
            dpi=300)
        plt.clf()
    print()


def plot_model_performance():
    imrex = pd.read_csv(
        'ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/full_metrics.csv')
    titan_on_imrex_data = pd.read_csv('TITAN/models/nocdr3dup_epgrouped5cv_paperparams_smallpad/full_metrics.csv')
    titan_scrambled_tcs = pd.read_csv('TITAN/models/titanData_strictsplit_scrambledtcrs/full_metrics.csv')
    titan = pd.read_csv('TITAN/models/titanData_strictsplit_nocdr3/full_metrics.csv')

    sns.set_palette("deep")
    plot_model_comparison(
        [imrex, titan],
        ['ImRex', 'TITAN'], "subset")
    plot_model_comparison(
        [imrex, titan, titan_on_imrex_data, titan_scrambled_tcs],
        ['ImRex', 'TITAN', 'TITAN on ImRex data', 'TITAN scrambled TCRs'])


def plot_method_correlation_comparison(attribution_handler: [ImrexAttributionsHandler, TITANAttributionsHandler],
                                       methods_subset=None):
    if isinstance(attribution_handler, ImrexAttributionsHandler):
        attribution_types = ['aa', 'pair-wise']
        correlations = [attribution_handler.get_aa_correlation(), attribution_handler.get_correlation()]
        random_correlations = [attribution_handler.get_aa_random_correlation(),
                               attribution_handler.get_random_correlation()]
    elif isinstance(attribution_handler, TITANAttributionsHandler):
        attribution_types = ['aa']
        correlations = [attribution_handler.get_aa_correlation()]
        random_correlations = [attribution_handler.get_aa_random_correlation()]
    else:
        raise TypeError(f"attribution_handler of wrong type, got {type(attribution_handler)} but expected "
                        f"{ImrexAttributionsHandler} or {TITANAttributionsHandler}")
    for attribution_type, correlation_results, random_results in zip(attribution_types, correlations,
                                                                     random_correlations):
        if methods_subset is None:
            methods = list(list(correlation_results.values())[0].keys())
            if 'IG' in methods: methods.remove('IG')
            if 'SHAP mean' in methods: methods.remove('SHAP mean')
            if set(methods) == {'SHAP BGdist', 'Vanilla', 'SmoothGrad', 'VanillaIG', 'SmoothGradIG', 'GuidedIG',
                                'XRAI', 'BlurIG', 'SmoothGradBlurIG'}:
                methods = ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SmoothGradIG', 'GuidedIG', 'BlurIG',
                           'SmoothGradBlurIG', 'XRAI', 'SHAP BGdist']
            plt.figure(figsize=(11, 3))
        else:
            methods = methods_subset
            plt.figure(figsize=(6.4 / 1.3, 4.8 / 1.3))
        method_correlation = {}
        for pdb_id, pdb_corr in correlation_results.items():
            for method, method_corr in pdb_corr.items():
                # We just do -1 * correlation, this gives the correlation between feature attribution and residue
                # proximity
                if method in method_correlation:
                    method_correlation[method].append(-method_corr)
                else:
                    method_correlation[method] = [-method_corr]

        correlation = {method: method_correlation[method] for method in methods}
        correlation = {"SHAP" if k == 'SHAP BGdist' else "IG" if k == "VanillaIG" else k: v for k, v in
                       correlation.items()}

        for method, corr in correlation.items():
            print(method, round(np.mean(corr), 3), round(np.std(corr), 3))
        print()

        correlation = pd.DataFrame(correlation)
        sns.boxplot(data=correlation, showfliers=False)
        sns.stripplot(data=correlation, color="0.25", s=3)
        plt.xlabel("Feature attribution extraction method")
        plt.ylabel(f"Pearson correlation")
        print(f"Plotted Pearson correlation for {attribution_handler.name} {attribution_type}")
        x = [-0.5, len(methods) - 0.5]
        y = [random_results[0]] * 2
        y_error_min = [random_results[0] - random_results[1]] * 2
        y_error_max = [random_results[0] + random_results[1]] * 2
        plt.plot(x, y, '--', label=f"Random Pearson correlation "
                                   f"({random_results[0]:.3f} +- {random_results[1]:.3f})")
        plt.fill_between(x, y_error_min, y_error_max, alpha=0.3)
        plt.grid(axis='y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'output/plots/{attribution_handler.name}_{attribution_type}_pearson'
                    f'_correlation_comparison{"_subset" if methods_subset is not None else ""}.png', dpi=300)
        plt.clf()


def plot_method_correlation_comparison_all_models_subset(model_handlers):
    methods_subset = ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist']
    f, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
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
            raise TypeError(f"attribution_handler of wrong type, got {type(attribution_handler)} but expected "
                            f"{ImrexAttributionsHandler} or {TITANAttributionsHandler}")

    for ax, correlation_results, random_results, name in zip(axs, correlations, random_correlations, names):
        method_correlation = {}
        for pdb_id, pdb_corr in correlation_results.items():
            for method, method_corr in pdb_corr.items():
                # We just do -1 * correlation, this gives the correlation between feature attribution and residue
                # proximity
                if method in method_correlation:
                    method_correlation[method].append(-method_corr)
                else:
                    method_correlation[method] = [-method_corr]

        correlation = {method: method_correlation[method] for method in methods_subset}
        correlation = {"SHAP" if k == 'SHAP BGdist' else "IG" if k == "VanillaIG" else k: v for k, v in
                       correlation.items()}

        for method, corr in correlation.items():
            print(method, round(np.mean(corr), 3), round(np.std(corr), 3))
        print()

        correlation = pd.DataFrame(correlation)
        sns.boxplot(data=correlation, ax=ax, showfliers=False)
        sns.stripplot(data=correlation, color="0.25", s=3, ax=ax)
        x = [-0.5, len(methods_subset) - 0.5]
        y = [random_results[0]] * 2
        y_error_min = [random_results[0] - random_results[1]] * 2
        y_error_max = [random_results[0] + random_results[1]] * 2
        ax.plot(x, y, '--', label=f"Random correlation\n"
                                  f"({random_results[0]:.3f} +- {random_results[1]:.3f})")
        ax.fill_between(x, y_error_min, y_error_max, alpha=0.3)
        ax.grid(axis='y')
        # ax.set_title(name)
        ax.legend()

    axs[0].set_ylabel('Pearson correlation')
    axs[1].set_xlabel('Feature attribution extraction method')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/plots/pearson_comparison_all_subset.png', dpi=300)
    plt.clf()


def plot_aa_model_comparison(model_handlers, random_index, method, save_post, figsize):
    model_correlation_ps = []
    model_correlation = []
    names_ps = []
    names = []
    for model_handler in model_handlers:
        # per sequence
        correlation = model_handler.get_aa_correlation_ps()
        method_correlation_ep = []
        method_correlation_cdr3 = []
        for pdb_id, methods in correlation.items():
            method_correlation_ep.append(-methods[method][0])
            method_correlation_cdr3.append(-methods[method][1])
        model_correlation_ps.append(method_correlation_ep)
        model_correlation_ps.append(method_correlation_cdr3)
        if save_post == "all":
            names_ps.append(model_handler.display_name + ' epitope')
            names_ps.append(model_handler.display_name + ' CDR3')
        else:
            names_ps.append(model_handler.display_name + '\nepitope')
            names_ps.append(model_handler.display_name + '\nCDR3')
        # full
        correlation = model_handler.get_aa_correlation()
        method_correlation = []
        for pdb_id, methods in correlation.items():
            method_correlation.append(-methods[method])
        model_correlation.append(method_correlation)
        names.append(model_handler.display_name)

    per_seq_palette = [c for c in sns.color_palette("deep", 3) for _ in range(2)]
    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]}, sharey=True)
    sns.boxplot(data=model_correlation, ax=axs[0], showfliers=False)
    sns.stripplot(data=model_correlation, ax=axs[0], color='0.25', s=3)
    sns.boxplot(data=model_correlation_ps, ax=axs[1], palette=per_seq_palette, showfliers=False)
    sns.stripplot(data=model_correlation_ps, ax=axs[1], color='0.25', s=3)

    for model, errors in zip(names, model_correlation):
        print(model, round(np.mean(errors), 3), round(np.std(errors), 3))

    for model, errors in zip(names_ps, model_correlation_ps):
        print(model, round(np.mean(errors), 3), round(np.std(errors), 3))

    axs[0].set_ylabel('Pearson correlation')
    if save_post == "all":
        axs[0].set_xticklabels([split_line(n, 12) for n in names])
        axs[1].set_xticklabels([split_line(n, 12) for n in names_ps])
    else:
        axs[0].set_xticklabels([split_line(n, 15) for n in names])
        axs[1].set_xticklabels([split_line(n, 15) for n in names_ps])

    re_ps = model_handlers[random_index].get_aa_random_correlation_ps()
    re = model_handlers[random_index].get_aa_random_correlation()

    re_ps_ep_range = (re_ps[0][0] + re_ps[0][1], re_ps[0][0] - re_ps[0][1])
    re_ps_cdr3_range = (re_ps[1][0] + re_ps[1][1], re_ps[1][0] - re_ps[1][1])
    re_range = (re[0] + re[1], re[0] - re[1])
    x_ps = [-0.5, len(names_ps) - 0.5]
    x = [-0.5, len(names) - 0.5]
    print(f"Random correlation epitope ({re_ps[0][0]} +- {re_ps[0][1]})")
    print()
    if save_post == "all":
        axs[1].plot(x_ps, [re_ps[0][0]] * 2, '--',
                    label=f"Random correlation epitope ({round(re_ps[0][0], 3):.3f} +- {round(re_ps[0][1], 3)})")
        axs[1].plot(x_ps, [re_ps[1][0]] * 2, '--',
                    label=f"Random correlation CDR3 ({round(re_ps[1][0], 3):.3f} +- {round(re_ps[1][1], 3)})")
    else:
        axs[1].plot(x_ps, [re_ps[0][0]] * 2, '--',
                    label=f"Random correlation epitope\n({round(re_ps[0][0], 3):.3f} +- {round(re_ps[0][1], 3)})")
        axs[1].plot(x_ps, [re_ps[1][0]] * 2, '--',
                    label=f"Random correlation CDR3\n({round(re_ps[1][0], 3):.3f} +- {round(re_ps[1][1], 3)})")
    axs[1].fill_between(x_ps, re_ps_ep_range[0], re_ps_ep_range[1], alpha=0.3)
    axs[1].fill_between(x_ps, re_ps_cdr3_range[0], re_ps_cdr3_range[1], alpha=0.3)
    if save_post == "all":
        axs[0].plot(x, [re[0]] * 2, '--', label=f"Random correlation ({round(re[0], 3):.3f} +- {round(re[1], 3)})")
    else:
        axs[0].plot(x, [re[0]] * 2, '--', label=f"Random correlation\n({round(re[0], 3):.3f} +- {round(re[1], 3)})")

    axs[0].fill_between(x, re_range[0], re_range[1], alpha=0.3)
    if save_post == "all":
        axs[1].legend()
        axs[0].legend()
    else:
        axs[1].legend(fontsize=9)
        axs[0].legend(fontsize=8)
    axs[0].set_xlabel('Model')
    axs[1].set_xlabel('Model + sequence')

    fig.tight_layout()
    axs[0].grid(axis='y')
    axs[1].grid(axis='y')

    plt.savefig(f'output/plots/models_{method}_pearson_correlation_comparison_{save_post}.png', dpi=300)
    plt.close()


def plot_sample_details(model_handlers, method, save_post, dist_i=0):
    model_attributions = [model.get_aa_norm_attributions() for model in model_handlers]
    model_names = [model.display_name for model in model_handlers]
    sequences = model_handlers[dist_i].get_sequences()
    distances = model_handlers[dist_i].get_aa_norm_distances()
    # plt.figure(figsize=(10 / 1.3, 3 / 1.3))
    for pdb_id, dist in distances.items():
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
        plt.savefig(f'output/plots/sample_details/{method}_{pdb_id}_{save_post}.png', dpi=300, bbox_inches='tight')
        plt.clf()


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
        plt.tight_layout()
        plt.savefig(f'output/plots/sample_details/{titan_handler.display_name}_{pdb_id}.png', dpi=300,
                    bbox_inches='tight')
        plt.clf()


def plot_2d_sample_attributions(model_handler, methods, save_name):
    attributions = model_handler.get_norm_attributions()
    sequences = model_handler.get_sequences()
    distances = model_handler.get_norm_distances()

    fig = plt.figure(figsize=(6.4 / 2, 4.8 / 2))
    ax = fig.add_axes([0.05, 0.05, 0.07, 0.9])
    cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical', cmap=plt.get_cmap('Greys'))
    cb.set_label(f"Feature attribution / residue proximity")
    plt.savefig('output/plots/colorbar_grey.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    for pdb_id, attribution in attributions.items():
        ep, cdr3 = sequences[pdb_id]
        fig, axs = plt.subplots(1, len(methods) + 1, sharey=True, figsize=(12 / 1.3, 4.8 / 1.3))
        for i, method in enumerate(methods):
            att = attribution[method]
            grid = axs[i].imshow(att, cmap='Greys', vmin=0, vmax=1)
            axs[i].set_xticks(list(range(len(ep))))
            axs[i].set_xticklabels(ep)
            axs[i].set_title("SHAP" if method == 'SHAP BGdist' else "IG" if method == "VanillaIG" else method)
            axs[i].set_xlabel('epitope')

        dist = distances[pdb_id]
        axs[-1].imshow(dist, cmap='Greys', vmin=0, vmax=1)
        axs[-1].set_xticks(list(range(len(ep))))
        axs[-1].set_xticklabels(ep)
        axs[-1].set_title('Pairwise\nresidue proximity')
        axs[-1].set_xlabel('epitope')

        plt.yticks(list(range(len(cdr3))), cdr3)

        axs[0].set_ylabel('CDR3')
        plt.tight_layout(pad=0.1)
        plt.savefig(f'output/plots/sample_2d_details/{pdb_id}_{save_name}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_2d_ImRex_input(image_path, model_handler):
    input_imgs = {}
    for f in sorted(os.listdir(image_path)):
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
        plt.savefig(f'output/plots/imrex_input_features/{pdb_id}.png', dpi=300)
        plt.close()


def plot_positional_average(model_handlers, method, save_post, dist_i=0):
    model_attributions = {model.display_name: model.get_aa_norm_attributions() for model in model_handlers}
    sequences = model_handlers[dist_i].get_sequences()
    distances = model_handlers[dist_i].get_aa_norm_distances()

    eps = [v[0] for k, v in sequences.items()]
    cdr3s = [v[1] for k, v in sequences.items()]

    max_ep = len(max(eps, key=len))
    max_cdr3 = len(max(cdr3s, key=len))

    pos_distances_ep = []
    pos_distances_cdr3 = []
    pos_model_attributions = {k: [] for k, v in model_attributions.items()}
    for pdb_id, dist in distances.items():
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
    plt.gcf().set_size_inches(10 / 1.3, 3 / 1.3)
    heatmap = np.array(heatmap)
    heatmap = np.ma.masked_where(heatmap == -1, heatmap)
    grid = plt.imshow(heatmap, cmap='Greys')
    plt.xticks(list(range(max_ep)) + list(range(max_ep + 1, max_ep + 1 + max_cdr3)),
               ['$\mathregular{e_{' + str(i) + '}}$' for i in range(max_ep)] +
               ['$\mathregular{c_{' + str(i) + '}}$' for i in range(max_cdr3)])
    plt.yticks(list(range(len(heatmap))), names)
    plt.colorbar(grid, orientation='horizontal', pad=0.2, label=f"{method} feature attribution")
    plt.clim(0, 1)
    plt.tight_layout()
    plt.savefig(f'output/plots/average_pos_attributions_{method}_{save_post}.png', dpi=300, bbox_inches='tight')
    plt.clf()


def main():
    imrex_attributions_handler = ImrexAttributionsHandler(
        name="imrex_nocdr3dup",
        display_name="ImRex",
        model_path="ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/iteration_2/"
                   "2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv-epoch20.h5",
        image_path="data/tcr3d_images/",
        save_folder="data",
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

    titan_strictsplit_handler = TITANAttributionsHandler(
        name="titan_strictsplit",
        display_name="TITAN",
        model_path="TITAN/models/titanData_strictsplit_nocdr3/cv5/titanData_strictsplit_nocdr3_5/",
        tcrs_path='data/tcrs.csv',
        epitopes_path='data/epitopes.csv',
        data_path='data/tcr3d_titan_input.csv',
        save_folder='data'
    )
    sns.set_palette("deep")

    # Fig 2: Correlation comparison of 4 extraction methods for ImRex pairwise, ImRex AA and TITAN
    plot_method_correlation_comparison_all_models_subset([imrex_attributions_handler, titan_strictsplit_handler])

    # Fig 3: Feature attributions for ImRex AA and TITAN and Residue proximity
    plot_sample_details([imrex_attributions_handler, titan_strictsplit_handler], 'SmoothGrad', 'imrextitan')

    # Fig 4: Average feature attributions for ImRex AA and TITAN and Residue proximity
    plot_positional_average([imrex_attributions_handler, titan_strictsplit_handler], 'SmoothGrad', 'imrextitan')

    # Fig 5: Compare correlation between ImRex AA and TITAN and epitope/CDR3
    plot_aa_model_comparison([imrex_attributions_handler, titan_strictsplit_handler], 0, 'SmoothGrad', 'imrextitan',
                             (6.4 / 1.05, 4.8 / 1.05))

    # Fig 6: Model performance
    plot_model_performance()

    # Fig S1: ImRex 2D feature encoding input
    plot_2d_ImRex_input('data/tcr3d_images_new/', imrex_attributions_handler)

    # Fig S2: Same as Fig 2 but with all extraction methods
    plot_method_correlation_comparison(imrex_attributions_handler)
    plot_method_correlation_comparison(titan_strictsplit_handler)

    # Fig S3 and S4: Detailed feature attributions for ImRex and TITAN
    plot_2d_sample_attributions(imrex_attributions_handler, ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist'],
                                "4_methods")
    plot_TITAN_methods_sample_details(titan_strictsplit_handler, imrex_attributions_handler,
                                      ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist'])

    # Fig S5: Average feature attribution for the 3 models
    plot_positional_average([imrex_attributions_handler, titan_strictsplit_handler, titan_on_imrex_data_handler],
                            'SmoothGrad', 'all')

    # Fig S6: Like Fig 5 but on all models
    plot_aa_model_comparison([imrex_attributions_handler, titan_strictsplit_handler, titan_on_imrex_data_handler], 0,
                             'SmoothGrad', 'all', (12.8 / 1.15, 4.8 / 1.15))


if __name__ == "__main__":
    main()
