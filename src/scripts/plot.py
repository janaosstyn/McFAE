import os
import pickle
from typing import List

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from plot_helper_functions import *
from src.imrex_attributions import ImrexAttributionsHandler
from src.titan_attributions import TITANAttributionsHandler
from src.util import aa_add_padding, split_line, imrex_remove_padding


class Plotter:
    def __init__(self):
        self.attribution_handlers: List = []

        self.current_attribution_handler: ImrexAttributionsHandler = None
        self.current_method: [str, None] = None  # SmoothGrad, SHAP,...
        self.current_correlation_method: [str, None] = None  # pearson, spearman
        self.channels: bool = True  # TODO
        self.combined: bool = False

        self.sub_dir: str = ''

    @staticmethod
    def create_imrex_handler(name: str, correction: str = '') -> ImrexAttributionsHandler:
        """
        Create a single ImrexAttributionsHandler.

        Parameters
        ----------
        name: The name for the handler. The display name will be derived thereof.
        correction: Is the data somehow corrected or not? This influences the name.

        Returns
        -------
        The constructed ImrexAttributionsHandler.
        """
        model_path_imrex = "ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/iteration_2/" \
                           "2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv-epoch20.h5"
        model_path_scrambled_eps = "ImRex/models/models/2022-11-30_13-55-56_scrambled_eps/iteration_0/" \
                                   "2022-11-30_13-55-56_scrambled_eps-epoch20.h5"
        model_path_scrambled_cdr3 = "ImRex/models/models/2022-12-01_09-52-59_scrambled_tcrs/iteration_0/" \
                                    "2022-12-01_09-52-59_scrambled_tcrs-epoch20.h5"

        model_path = model_path_imrex if 'scrambled' not in name \
            else model_path_scrambled_eps if 'scrambled_eps' in name \
            else model_path_scrambled_cdr3

        return ImrexAttributionsHandler(
            name=f'{name}{"_" if correction != "" else ""}{correction}',
            display_name=name.replace('imrex', 'ImRex').replace('_', ' ').replace('aa', 'AA').replace('2d', '2D'),
            model_path=model_path,
            image_path="data/tcr3d_images/",
            save_folder="data",
        )

    def get_attribution_handlers(self, correction: str = ''):
        """
        Get a set of three attribution handlers: a regular nocdr3dup, a scrambled eps and a scrambled tcrs variant.

        Parameters
        ----------
        correction: Used to specify a correction (e.g. positional_corrected).
        """
        self.attribution_handlers = [
            self.create_imrex_handler(name='imrex_nocdr3dup', correction=correction),
            self.create_imrex_handler(name='imrex_scrambled_eps', correction=correction),
            self.create_imrex_handler(name='imrex_scrambled_tcrs', correction=correction)
        ]

    def calculate_and_plot_positional_correction(self):
        """
        Runs self.calculate_positional_correction() on uncorrected data.
        Result: positional corrected data.
        """
        self.get_attribution_handlers()
        self.sub_dir = 'uncorrected'
        self.generate_correlation_plots()

        self.calculate_positional_correction()

        self.sub_dir = 'positional_corrected'
        self.get_attribution_handlers(correction='positional_corrections')
        self.generate_correlation_plots()
        exit()

    def calculate_and_plot_aa_1d_corrections(self):
        """
        Runs self.calculate_aa_correction_1d() on uncorrected data.
        Result: AA (1D) corrected data.
        """
        self.get_attribution_handlers()
        self.sub_dir = 'uncorrected'
        self.generate_correlation_plots()

        self.calculate_aa_correction_1d()

        self.sub_dir = 'aa_1d_corrected'
        self.get_attribution_handlers(correction='aa_corrections')
        self.generate_correlation_plots()

    def calculate_and_plot_aa_2d_corrections(self):
        """
        Runs self.calculate_aa_correction_2d() on uncorrected data.
        Result: AA (2D) corrected data.
        """
        self.get_attribution_handlers()
        self.sub_dir = 'uncorrected'
        self.generate_correlation_plots()

        self.calculate_aa_correction_2d()

        self.sub_dir = 'aa_2d_corrected'
        self.get_attribution_handlers(correction='aa_2d_corrections')
        self.generate_correlation_plots()
        self.current_method = 'SmoothGrad'
        self.plot_positional_average()

    def calculate_and_plot_aa_1d_correction_on_positional_correction(self):
        """
        Runs self.calculate_aa_correction_1d() on positional corrected data.
        Result: data that is both positional and AA (1D) corrected.
        """
        self.get_attribution_handlers(correction='positional_corrections')
        self.sub_dir = 'positional_corrected'
        self.generate_correlation_plots()

        self.calculate_aa_correction_1d()

        self.sub_dir = 'positional_aa_corrected'
        self.get_attribution_handlers(correction='positional_aa_corrections')
        self.generate_correlation_plots()

    def calculate_and_plot_positional_correction_on_aa_1d_correction(self):
        """
        Runs self.calculate_positional_correction() on aa corrected data.
        Result: data that is both positional and AA (1D) corrected.
        """
        self.get_attribution_handlers(correction='aa_corrections')
        self.sub_dir = 'aa_1d_corrected'
        self.generate_correlation_plots()

        self.calculate_positional_correction()

        self.sub_dir = 'aa_positional_corrected'
        self.get_attribution_handlers(correction='aa_positional_corrections')
        self.generate_correlation_plots()

    def generate_correlation_plots(self):
        """
        Generate some correlation plots.
        """
        for self.current_correlation_method in ['pearson', 'spearman']:
            for specs in ['AA', 'pairwise']:
                self.plot_method_correlation_comparison_all_models_subset(
                    specs=[specs] * 3
                )
        self.current_correlation_method = None

        self.current_method = None
        for model in self.attribution_handlers:
            self.current_attribution_handler = model
            for self.current_correlation_method in ['pearson', 'spearman']:
                self.plot_method_correlation_comparison()
        self.current_correlation_method = None

    def get_comparison_plot_filename(self, keywords: List[str] = None) -> str:
        """
        Construct a filename based on the given parameters.
        This method is intended to provide a fixed format for output png filenames.

        Parameters
        ----------
        keywords: some characteristic keywords as list of strings (can be None)

        Returns
        -------
        filename: complete filename including appropriate directory
        (the directory is given by the function name of the calling method)
        """
        import inspect
        dir_name = f"output/plots/comparison_plots/{self.sub_dir}/" \
                   f"{inspect.currentframe().f_back.f_code.co_name.replace('plot_', '')}_plots"

        if not os.path.isdir(f'output/plots/comparison_plots/{self.sub_dir}'):
            os.mkdir(f'output/plots/comparison_plots/{self.sub_dir}')

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        return f"{dir_name}/" \
               f"{f'{self.current_method}_' if self.current_method is not None else ''}" \
               f"{f'{self.current_correlation_method}' if self.current_correlation_method is not None else ''}" \
               f"{'' if keywords is None else '_' + '_'.join(keywords)}" \
               f"{f'_ch' if self.channels else ''}" \
               f"{f'_comb' if self.channels and self.combined else ''}" \
               f".png"

    def get_sample_details_filename(self, pdb_id: str, keyword: str = None):
        """
        Obtain a filename for a sample details output figure

        Parameters
        ----------
        pdb_id: the id of the PDB complex
        keyword: an additional keyword to characterise the image

        Returns
        -------
        filename: complete filename including appropriate directory
        (the directory is given by a concatenation of the names of the attribution handlers)
        """
        dir_name = f'output/plots/sample_details/' \
                   f'{"".join([model_handler.name for model_handler in self.attribution_handlers])}'

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        return f"{dir_name}/" \
               f"{pdb_id}" \
               f"_{self.current_method}" \
               f"{'_' + keyword if keyword is not None else ''}" \
               f"{f'_ch' if self.channels else ''}" \
               f"{f'_comb' if self.channels and self.combined else ''}"

    def get_sample_2d_details_filename(self, pdb_id: str) -> str:
        """
        Obtain a filename for a 2D sample, based on the given parameters

        Parameters
        ----------
        pdb_id: the id of the PDB complex

        Returns
        -------
        filename: complete filename including appropriate directory
        (the directory is given by the name of the attribution_handler)
        """
        dir_name = f'output/plots/sample_2d_details/{str(self.current_attribution_handler)}'

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        return f"{dir_name}/" \
               f"{pdb_id}" \
               f"_{self.current_method}" \
               f"{f'_ch' if self.channels else ''}" \
               f"{f'_comb' if self.channels and self.combined else ''}" \
               f".png"

    def get_imrex_input_features_filename(self, pdb_id: str):
        """
        Obtain a filename for the imrex input features visualization for the given PDB complex

        Parameters
        ----------
        pdb_id: the id of the PDB complex

        Returns
        -------
        filename: complete filename including appropriate directory
        (the directory is given by the name of the attribution_handler)
        """
        dir_name = f'output/plots/{str(self.current_attribution_handler)}'

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        return f"{dir_name}/" \
               f"{pdb_id}" \
               f".png"

    def plot_method_correlation_comparison(self, methods_subset: [List[str], None] = None) -> None:
        """
        Create a plot that compares correlation between the different feature attribution extraction methods.

        Parameters
        ----------
        methods_subset: a list of methods to plot (optionally, if not provided all methods are plotted)
        """
        channels_true = self.channels

        # differentiate based on the type of the attributions handler
        if isinstance(self.current_attribution_handler, ImrexAttributionsHandler):
            if 'correction' not in ' '.join(handler.name for handler in self.attribution_handlers):
                attribution_types = ['aa', 'pair-wise']
                correlations = [
                    self.current_attribution_handler.get_aa_correlation(self.current_correlation_method),
                    self.current_attribution_handler.get_correlation(self.current_correlation_method)
                ]
                random_correlations = [
                    self.current_attribution_handler.get_aa_random_correlation(self.current_correlation_method),
                    self.current_attribution_handler.get_random_correlation(self.current_correlation_method)
                ]
            else:
                attribution_types = ['aa']
                correlations = [
                    self.current_attribution_handler.get_aa_correlation(self.current_correlation_method)
                ]
                random_correlations = [
                    self.current_attribution_handler.get_aa_random_correlation(self.current_correlation_method)
                ]
        elif isinstance(self.current_attribution_handler, TITANAttributionsHandler):
            attribution_types = ['aa']
            correlations = [self.current_attribution_handler.get_aa_correlation(self.current_correlation_method)]
            random_correlations = [
                self.current_attribution_handler.get_aa_random_correlation(self.current_correlation_method)
            ]
        else:
            raise TypeError(
                f"attribution_handler of wrong type, got {type(self.current_attribution_handler)} but expected "
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

            has_channels = False
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

                    if key == "Combi" and not self.combined and self.channels:
                        continue

                    if key is not None and key not in method_correlation[method]:
                        has_channels = True
                        if isinstance(method_correlation[method], list):
                            method_correlation[method] = {key: [-method_corr]}
                        else:
                            method_correlation[method][key] = [-method_corr]
                    elif key is not None:
                        method_correlation[method][key].append(-method_corr)
                    else:
                        method_correlation[method].append(-method_corr)
            self.channels = channels_true and has_channels

            # SHAP BGdist should be displayed as SHAP, VanillaIG as IG
            correlation = {method: method_correlation[method] for method in methods}
            correlation = {"SHAP" if k == 'SHAP BGdist' else "IG" if k == "VanillaIG" else k: v for k, v in
                           correlation.items()}

            f, axs = plt.subplots(
                nrows=3 if self.channels else 1,
                ncols=3 if self.channels else 1,
                figsize=(12 if self.channels else 11, 12 if self.channels else 3),
                sharex=True,
                sharey=True
            )

            dataframes = []
            if self.channels:
                for method in correlation:
                    corr_df = pd.DataFrame(correlation[method])
                    dataframes.append(corr_df)
                name_axs = [axs[j][k] for j in range(3) for k in range(3)]
                x_labels = list(correlation.keys())
                axs[1][0].set_ylabel(
                    "Pearson correlation" if self.current_correlation_method == 'pearson'
                    else "Spearman correlation"
                )
            else:
                dataframes = [
                    pd.DataFrame(correlation if 'Combined' not in correlation.keys() else correlation['Combined'])]
                name_axs = [axs]
                x_labels = ["Feature attribution extraction method"]
                axs.set_ylabel(
                    "Pearson correlation" if self.current_correlation_method == 'pearson'
                    else "Spearman correlation"
                )

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

            if self.channels:
                axs[2][1].set_xlabel(axs[2][1].get_xlabel() + "\nFeature attribution extraction method")

            plt.tight_layout()
            plt.savefig(
                fname=self.get_comparison_plot_filename(
                    keywords=[attribution_type, self.current_attribution_handler.name] +
                             (['subset'] if methods_subset is not None else []),
                ),
                dpi=300
            )
            plt.clf()
            plt.close()

        self.channels = channels_true

    def calculate_positional_correction(self):
        """
        Calculate (and plot) positional correction.
        Store aa_attributions and aa_norm_attributions to file.
        """
        for filename in ['aa_norm_attributions', 'aa_attributions']:
            # get the model attributions for both handlers
            raw_model_attributions = {
                model.display_name:
                    model.get_aa_norm_attributions() if 'norm' in filename
                    else model.get_aa_attributions() if 'aa' in filename
                    else model.get_attributions()
                for model in self.attribution_handlers
            }
            # if channels were split, tune the model_attributions dictionary
            model_attributions, self.channels = get_model_attributions_dictionary(
                model_attributions=raw_model_attributions,
                model_names=[handler.display_name for handler in self.attribution_handlers]
            )

            sequences = self.attribution_handlers[0].get_sequences()

            display_name_vs_name = {
                handler.display_name: handler.name.replace('_corrections', '')
                for handler in self.attribution_handlers
            }

            for model_name, model_attribution in model_attributions.items():
                positional_correction = dict()

                first_channel_key = list(model_attribution.keys())[0]
                first_pdb_id = list(model_attribution[first_channel_key].keys())[0]
                for method in model_attribution[first_channel_key][first_pdb_id]:
                    print(f'{display_name_vs_name[model_name]}', method)
                    self.current_method = method
                    positional_correction[method] = self.plot_positional_average()

                pdb_list = list(model_attribution[first_channel_key].keys())
                values = [
                    s for pdb_id, s in sequences.items()
                    if pdb_list is None or pdb_id in pdb_list
                ]
                max_ep_length, max_cdr3_length = [len(max(values, key=lambda x: len(x[i]))[i]) for i in [0, 1]]

                output_model_attributions = dict()
                for channel, channel_attributions in model_attribution.items():
                    for pdb_id, pdb_attributions in channel_attributions.items():
                        ep, cdr3 = sequences[pdb_id]
                        channel_pdb_id = f'{pdb_id}_{channel.replace("Ch ", "") if "Ch" in channel else "combined"}'
                        output_model_attributions[channel_pdb_id] = dict()
                        for method, method_attributions in pdb_attributions.items():
                            attribution = model_attribution[channel][pdb_id][method]
                            attribution = np.concatenate((
                                aa_add_padding(attribution[:len(ep)], max_ep_length),
                                [-1],
                                aa_add_padding(attribution[len(ep):], max_cdr3_length)
                            ))
                            attribution = attribution - positional_correction[method][channel]
                            attribution[max_ep_length] = np.nan
                            attribution = attribution[~np.isnan(attribution)]
                            output_model_attributions[channel_pdb_id][method] = attribution

                if not os.path.isdir(f'data/{display_name_vs_name[model_name]}_positional_corrections'):
                    os.mkdir(f'data/{display_name_vs_name[model_name]}_positional_corrections')
                pickle.dump(
                    output_model_attributions,
                    open(f'data/{display_name_vs_name[model_name]}_positional_corrections/{filename}.p', 'wb')
                )
            print()

    def calculate_aa_correction_1d(self):
        """
        Calculate AA (1D) corrections.
        Store aa_attributions and aa_norm_attributions to file.
        """
        for filename in ['aa_norm_attributions', 'aa_attributions']:
            # get the model attributions for both handlers
            raw_model_attributions = {
                model.display_name:
                    model.get_aa_norm_attributions() if 'norm' in filename
                    else model.get_aa_attributions() if 'aa' in filename
                    else model.get_attributions()
                for model in self.attribution_handlers
            }
            # if channels were split, tune the model_attributions dictionary
            model_attributions, self.channels = get_model_attributions_dictionary(
                model_attributions=raw_model_attributions,
                model_names=[handler.display_name for handler in self.attribution_handlers]
            )

            sequences = self.attribution_handlers[0].get_sequences()

            display_name_vs_name = {
                handler.display_name: handler.name.replace('_corrections', '')
                for handler in self.attribution_handlers
            }

            for model_name, model_attribution in model_attributions.items():
                aa_correction = dict()

                first_channel_key = list(model_attribution.keys())[0]
                first_pdb_id = list(model_attribution[first_channel_key].keys())[0]
                for method in model_attribution[first_channel_key][first_pdb_id]:
                    print(f'{display_name_vs_name[model_name]}', method)
                    self.current_method = method
                    aa_correction[method] = self.plot_aa_average_1d()

                output_model_attributions = dict()
                for channel, channel_attributions in model_attribution.items():
                    for pdb_id, pdb_attributions in channel_attributions.items():
                        ep, cdr3 = sequences[pdb_id]
                        channel_pdb_id = f'{pdb_id}_{channel.replace("Ch ", "") if "Ch" in channel else "combined"}'
                        output_model_attributions[channel_pdb_id] = dict()
                        for method, method_attributions in pdb_attributions.items():
                            attribution = model_attribution[channel][pdb_id][method]
                            for i, aa in enumerate(ep + cdr3):
                                attribution[i] = attribution[i] - aa_correction[method][channel][aa]

                            output_model_attributions[channel_pdb_id][method] = attribution

                if not os.path.isdir(f'data/{display_name_vs_name[model_name]}_aa_corrections'):
                    os.mkdir(f'data/{display_name_vs_name[model_name]}_aa_corrections')
                pickle.dump(
                    output_model_attributions,
                    open(f'data/{display_name_vs_name[model_name]}_aa_corrections/{filename}.p', 'wb')
                )
                print(f'data/{display_name_vs_name[model_name]}_aa_corrections/{filename}.p')
            print()

    def calculate_aa_correction_2d(self):
        """
        Calculate AA-positional (2D) correction.
        Store aa_attributions and aa_norm_attributions to file.
        """
        for filename in ['aa_norm_attributions', 'aa_attributions']:
            # get the model attributions for both handlers
            raw_model_attributions = {
                model.display_name:
                    model.get_aa_norm_attributions() if 'norm' in filename
                    else model.get_aa_attributions() if 'aa' in filename
                    else model.get_attributions()
                for model in self.attribution_handlers
            }
            # if channels were split, tune the model_attributions dictionary
            model_attributions, self.channels = get_model_attributions_dictionary(
                model_attributions=raw_model_attributions,
                model_names=[handler.display_name for handler in self.attribution_handlers]
            )

            sequences = self.attribution_handlers[0].get_sequences()

            display_name_vs_name = {
                handler.display_name: handler.name
                for handler in self.attribution_handlers
            }

            for model_name, model_attribution in model_attributions.items():
                aa_correction = dict()

                first_channel_key = list(model_attribution.keys())[0]
                first_pdb_id = list(model_attribution[first_channel_key].keys())[0]
                for method in model_attribution[first_channel_key][first_pdb_id]:
                    print(f'{display_name_vs_name[model_name]}', method)
                    self.current_method = method
                    aa_correction[method] = self.plot_aa_average_2d()

                pdb_list = list(model_attribution[first_channel_key].keys())
                values = [
                    s for pdb_id, s in sequences.items()
                    if pdb_list is None or pdb_id in pdb_list
                ]
                max_ep_length, max_cdr3_length = [len(max(values, key=lambda x: len(x[i]))[i]) for i in [0, 1]]

                output_model_attributions = dict()
                for channel, channel_attributions in model_attribution.items():
                    for pdb_id, pdb_attributions in channel_attributions.items():
                        ep, cdr3 = sequences[pdb_id]
                        sequence = ep + cdr3
                        channel_pdb_id = f'{pdb_id}_{channel.replace("Ch ", "") if "Ch" in channel else "combined"}'
                        output_model_attributions[channel_pdb_id] = dict()
                        for method, method_attributions in pdb_attributions.items():
                            attribution = model_attribution[channel][pdb_id][method]
                            attribution = np.concatenate((
                                aa_add_padding(attribution[:len(ep)], max_ep_length),
                                [np.nan],
                                aa_add_padding(attribution[len(ep):], max_cdr3_length)
                            ))

                            aa_index = 0
                            for i in range(attribution.size):
                                if np.isnan(attribution[i]):
                                    continue
                                attribution[i] = attribution[i] - aa_correction[method][channel][sequence[aa_index]][i]
                                aa_index += 1

                            attribution[max_ep_length] = np.nan
                            attribution = attribution[~np.isnan(attribution)]
                            output_model_attributions[channel_pdb_id][method] = attribution

                if not os.path.isdir(f'data/{display_name_vs_name[model_name]}_aa_2d_corrections'):
                    os.mkdir(f'data/{display_name_vs_name[model_name]}_aa_2d_corrections')
                pickle.dump(
                    output_model_attributions,
                    open(f'data/{display_name_vs_name[model_name]}_aa_2d_corrections/{filename}.p', 'wb')
                )
            print()

    def plot_method_correlation_comparison_all_models_subset(self, specs) -> None:
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
        """
        methods_subset = ['Vanilla', 'VanillaIG', 'SmoothGrad', 'SHAP BGdist']
        correlations = []
        random_correlations = []
        names = [handler.display_name for handler in self.attribution_handlers]
        for attribution_handler, spec in zip(self.attribution_handlers, specs):
            if spec == 'AA':
                correlations.append(attribution_handler.get_aa_correlation(self.current_correlation_method))
                random_correlations.append(
                    attribution_handler.get_aa_random_correlation(self.current_correlation_method))
            elif 'correction' in ' '.join([handler.name for handler in self.attribution_handlers]):
                return
            else:
                correlations.append(attribution_handler.get_correlation(self.current_correlation_method))
                random_correlations.append(attribution_handler.get_random_correlation(self.current_correlation_method))

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

        for i, (ax, corr_result, random_result, name) in enumerate(zip(axs, correlations, random_correlations, names)):
            # first create a dictionary
            #       * case 1 (no channels): maps method name to correlation values
            #       * case 2 (channels): maps method name to a map with channels that map to correlation values
            method_correlation = dict()
            for pdb_id, pdb_corr in corr_result.items():
                for method, method_corr in pdb_corr.items():
                    if method not in methods_subset:
                        continue

                    if method not in method_correlation:
                        method_correlation[method] = dict() if has_channels else list()

                    if has_channels:  # case 2
                        channel = "Combi" if "combined" in pdb_id.split("_")[1] else f'Ch {pdb_id.split("_")[1]}'
                        if not self.combined and channel == "Combi":
                            continue

                        if channel not in method_correlation[method]:
                            method_correlation[method][channel] = []

                        method_correlation[method][channel].append(-method_corr)
                    else:  # case 1
                        method_correlation[method].append(-method_corr)

            method_correlation = {
                "SHAP" if k == 'SHAP BGdist' else "IG" if k == "VanillaIG" else k: v
                for k, v in method_correlation.items()
            }

            # then create one or more Pandas dataframes and create plots from them
            if not has_channels:
                # case 1: one frame
                dataframes = [pd.DataFrame(method_correlation)]
                name_axs = [ax]

                axs[0].set_ylabel(
                    'Pearson correlation' if self.current_correlation_method == 'pearson'
                    else 'Spearman correlation'
                )
                axs[1].set_xlabel('Feature attribution extraction method')
            else:
                # case 2: one frame per method
                dataframes = [pd.DataFrame(method_correlation[method]) for method in method_correlation]
                name_axs = [ax[0], ax[1], ax[2], ax[3]]

                axs[i][0].set_ylabel(
                    f'Pearson correlation\n{name}' if self.current_correlation_method == 'pearson'
                    else f'Spearman correlation\n{name}'
                )
                if i == len(names) - 1:
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
                name_ax.plot(x, y, '--',
                             label=f"Random correlation\n({random_result[0]:.3f} +- {random_result[1]:.3f})")
                name_ax.fill_between(x, y_error_min, y_error_max, alpha=0.3)
                name_ax.grid(axis='y')
                name_ax.legend()

        plt.tight_layout()
        plt.savefig(
            fname=self.get_comparison_plot_filename(keywords=[specs[0]])
        )
        plt.clf()
        plt.close()

    def plot_aa_pearson_correlation_model_comparison(self, random_index: int, image_specific_instruction: str) -> None:
        """
        Compare epitope with CDR3 for the listed models (self.current_attribution_handlers)

        Parameters
        ----------
        random_index: random index for the random correlation calculation
        image_specific_instruction: a string that determines some aspects of the image
        """
        model_correlation_per_sequence = []
        model_correlation = []
        names_per_sequence = []
        names = []
        self.channels = False
        for model_handler in self.attribution_handlers:
            # per sequence
            self.channels, collected_names, model_correlations = get_per_sequence_correlation(
                model_handler=model_handler,
                method=self.current_method,
                is_combined=self.combined,
                shows_all=image_specific_instruction == "all"
            )
            names_per_sequence.extend(collected_names)
            model_correlation_per_sequence.extend(model_correlations)

            # full
            model_correlations = get_correlation(
                model_handler=model_handler,
                method=self.current_method,
                is_combined=self.combined,
                has_channels=self.channels
            )
            names.extend(collected_names)
            model_correlation.extend(model_correlations)

        fig_size = (14, 6) if self.channels else (6.1, 4.6)
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
        sns.boxplot(data=model_correlation_per_sequence, ax=axs[1], palette=per_seq_palette, showfliers=False)
        sns.stripplot(data=model_correlation_per_sequence, ax=axs[1], color='0.25', s=3)

        axs[0].set_ylabel(
            'Pearson correlation' if self.current_correlation_method == 'pearson'
            else 'Spearman correlation'
        )
        axs[0].set_xticklabels(
            [split_line(n, 12 if image_specific_instruction == 'all' else 15) for n in names]
        )
        axs[1].set_xticklabels(
            [split_line(n, 12 if image_specific_instruction == 'all' else 15) for n in names_per_sequence]
        )

        re_ps = self.attribution_handlers[random_index].get_aa_random_correlation_ps()
        re = self.attribution_handlers[random_index].get_aa_random_correlation()

        re_ps_ep_range = (re_ps[0][0] + re_ps[0][1], re_ps[0][0] - re_ps[0][1])
        re_ps_cdr3_range = (re_ps[1][0] + re_ps[1][1], re_ps[1][0] - re_ps[1][1])
        re_range = (re[0] + re[1], re[0] - re[1])
        x_ps = [-0.5, len(names_per_sequence) - 0.5]
        x = [-0.5, len(names) - 0.5]

        sep = ' ' if image_specific_instruction == 'all' else '\n'
        axs[1].plot(x_ps, [re_ps[0][0]] * 2, '--',
                    label=f"Random correlation epitope{sep}({round(re_ps[0][0], 3):.3f} +- {round(re_ps[0][1], 3)})")
        axs[1].plot(x_ps, [re_ps[1][0]] * 2, '--',
                    label=f"Random correlation CDR3{sep}({round(re_ps[1][0], 3):.3f} +- {round(re_ps[1][1], 3)})")

        axs[1].fill_between(x_ps, re_ps_ep_range[0], re_ps_ep_range[1], alpha=0.3)
        axs[1].fill_between(x_ps, re_ps_cdr3_range[0], re_ps_cdr3_range[1], alpha=0.3)

        axs[0].plot(x, [re[0]] * 2, '--', label=f"Random correlation{sep}({round(re[0], 3):.3f} +- {round(re[1], 3)})")

        axs[0].fill_between(x, re_range[0], re_range[1], alpha=0.3)
        if image_specific_instruction == "all":
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
            fname=self.get_comparison_plot_filename(),
            dpi=300
        )
        plt.close()

    def plot_sample_details(self) -> None:
        """
        Creates a feature attribution heat map for each pdb complex.
        """
        # get the model attributions for both handlers
        model_attributions = {
            model.display_name: model.get_aa_norm_attributions()
            for model in self.attribution_handlers[:1]
        }
        # if channels were split, tune the model_attributions dictionary
        model_attributions, self.channels = get_model_attributions_dictionary(
            model_attributions=model_attributions,
            model_names=[handler.display_name for handler in self.attribution_handlers[:1]]
        )

        # sequences and distances are the same for both handlers
        sequences = self.attribution_handlers[0].get_sequences()
        distances = self.attribution_handlers[0].get_aa_norm_distances()

        from plot_helper_heatmap import SampleDetailsHeatmap
        heatmap = SampleDetailsHeatmap()
        for pdb_id, dist in distances.items():
            ep, cdr3 = sequences[pdb_id]
            heatmap.reset()
            heatmap.initialize_rows(row_names=model_attributions[list(model_attributions.keys())[0]].keys())
            heatmap.set_sequence(ep_sequence=ep, cdr3_sequence=cdr3)
            for model_name, model_attribution in model_attributions.items():
                for channel, channel_attribution in model_attribution.items():
                    if pdb_id not in channel_attribution.keys():
                        break
                    attribution = channel_attribution[pdb_id][self.current_method]
                    heatmap.set_ep_values(row_name=channel, values=attribution[:len(ep)])
                    heatmap.set_cdr3_values(row_name=channel, values=attribution[len(ep):])
            # Imrex
            heatmap.compose_heatmap()
            heatmap.create_plot(
                color_bar_label=f"{self.current_method} feature attribution",
                filename=self.get_sample_details_filename(pdb_id=pdb_id)
            )

    def plot_2d_sample_details(self, methods_subset: List[str]) -> None:
        """
        Create a plot showing 2D sample details for each PDB-complex

        Parameters
        ----------
        methods_subset: a list of methods to plot
        """
        attributions = self.current_attribution_handler.get_norm_attributions()
        sequences = self.current_attribution_handler.get_sequences()
        distances = self.current_attribution_handler.get_norm_distances()
        self.channels = '_' in list(attributions.keys())[0]

        # gradient gray color bar
        fig = plt.figure(figsize=(6.4 / 2, 4.8 / 2))
        ax = fig.add_axes([0.05, 0.05, 0.07, 0.9])
        cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical', cmap=plt.get_cmap('Greys'))
        cb.set_label(f"Feature attribution / residue proximity")
        plt.savefig('output/plots/colorbar_grey.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

        for pdb_id, attribution in attributions.items():
            if self.channels and ('combined' in pdb_id or int(pdb_id.split('_')[1]) > 0):
                continue
            ep, cdr3 = sequences[pdb_id.split('_')[0]]
            rows = 4 if self.channels and len(methods_subset) > 1 else 1
            columns = 5 if self.channels and len(methods_subset) == 1 else len(methods_subset) + 1
            if self.combined and len(methods_subset) == 1:
                columns += 1
            fig, axs = plt.subplots(
                nrows=rows,
                ncols=columns,
                sharex=True,
                sharey=True,
                figsize=(2 * columns, 3.75 * rows)
            )

            if self.channels and len(methods_subset) > 1:
                for row in range(4):
                    channel_attribution = attributions[pdb_id.replace('_0', f'_{row}')]
                    for column, method in enumerate(methods_subset):
                        att = channel_attribution[method]
                        axs[row][column] = set_att_plot_specs(
                            axs=axs[row][column], ep=ep, att=att, method=method, x_label=False, title=row == 0
                        )
                        if column == 0:
                            axs[row][column].set_ylabel(f'CDR3 Ch {row}')

                    dist = distances[pdb_id.split('_')[0]]
                    axs[row][-1] = set_dist_plot_specs(axs=axs[-1], ep=ep, dist=dist, x_label=False, title=row == 0)

                for i in range(columns):
                    axs[0][i].set_xlabel('epitope')

            elif self.channels:
                method = methods_subset[0]
                for row in range(4):
                    channel_attribution = attributions[pdb_id.replace('_0', f'_{row}')]
                    att = channel_attribution[method]
                    axs[row] = set_att_plot_specs(axs=axs[row], ep=ep, att=att, method=method, x_label=False)

                if self.combined:
                    channel_attribution = attributions[pdb_id.replace('_0', f'_combined')]
                    att = channel_attribution[method]
                    axs[-2] = set_att_plot_specs(axs=axs[-2], ep=ep, att=att, method=method)

                dist = distances[pdb_id.split('_')[0]]
                axs[-1] = set_dist_plot_specs(axs=axs[-1], ep=ep, dist=dist, x_label=False)
                for i in range(columns):
                    axs[i].set_xlabel('epitope')
                axs[0].set_ylabel('CDR3')

            else:
                for column, method in enumerate(methods_subset):
                    att = attribution[method]
                    axs[column] = set_att_plot_specs(axs=axs[column], ep=ep, att=att, method=method)

                dist = distances[pdb_id]
                axs[-1] = set_dist_plot_specs(axs=axs[-1], ep=ep, dist=dist)
                axs[0].set_ylabel('CDR3')

            plt.yticks(list(range(len(cdr3))), cdr3)
            plt.tight_layout(pad=0.1)
            plt.savefig(
                fname=self.get_sample_2d_details_filename(
                    pdb_id=pdb_id.split("_")[0],
                ),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)

    def plot_2d_imrex_input(self, image_path: str) -> None:
        """
        Plot the 2D ImRex input

        Parameters
        ----------
        image_path: the path to the .pkl image input
        """
        input_imgs = {}
        for f in sorted(os.listdir(image_path)):
            if not f.endswith('.pkl'):
                continue
            input_imgs[f[:-4]] = tf.cast(tf.convert_to_tensor(pickle.load(open(image_path + f, 'rb'))), tf.float32)

        sequences = self.current_attribution_handler.get_sequences()
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
            plt.savefig(
                fname=self.get_imrex_input_features_filename(pdb_id=pdb_id),
                dpi=300
            )
            plt.close()

    def plot_positional_average(self) -> Dict:
        """
        Creates a feature attribution heat map for each pdb complex.
        """

        # get the model attributions for both handlers
        model_attributions = {
            model.display_name: model.get_aa_norm_attributions()
            for model in self.attribution_handlers
        }
        # sequences and distances are the same for both handlers
        sequences = self.attribution_handlers[0].get_sequences()
        distances = self.attribution_handlers[0].get_aa_norm_distances()

        # if channels were split, tune the model_attributions dictionary
        model_attributions, self.channels = get_model_attributions_dictionary(
            model_attributions=model_attributions,
            model_names=[handler.display_name for handler in self.attribution_handlers]
        )
        # get the pdbs that occur in model attributions
        pdb_list = list(model_attributions[self.attribution_handlers[0].display_name]['Ch 0'].keys())

        from plot_helper_heatmap import PositionalHeatmap
        heatmap = PositionalHeatmap()
        heatmap.extract_max_ep_cdr3_length(sequences=sequences, pdb_subset=pdb_list)
        heatmap.initialize_rows(
            row_names=list(
                model_attributions[self.attribution_handlers[0].display_name].keys()
            ) + ['Residue proximity']
        )
        heatmap_scrambled = heatmap.copy_initialization()

        for pdb_id, dist in distances.items():
            if pdb_id not in pdb_list:
                continue
            ep, cdr3 = sequences[pdb_id]

            # Residue proximity
            heatmap.add_ep_attributions(
                row_name='Residue proximity', attributions=aa_add_padding(dist[:len(ep)], heatmap.max_ep_length)
            )
            heatmap.add_cdr3_attributions(
                row_name='Residue proximity', attributions=aa_add_padding(dist[len(ep):], heatmap.max_cdr3_length)
            )
            heatmap_scrambled.add_ep_attributions(
                row_name='Residue proximity', attributions=aa_add_padding(dist[:len(ep)], heatmap.max_ep_length)
            )
            heatmap_scrambled.add_cdr3_attributions(
                row_name='Residue proximity', attributions=aa_add_padding(dist[len(ep):], heatmap.max_cdr3_length)
            )

            # Attributions
            for model_name, model_attribution in model_attributions.items():
                for channel, channel_attribution in model_attribution.items():
                    attribution = channel_attribution[pdb_id][self.current_method]

                    if 'scrambled' not in model_name:
                        heatmap.add_ep_attributions(
                            row_name=channel,
                            attributions=aa_add_padding(attribution[:len(ep)], heatmap.max_ep_length)
                        )
                        heatmap.add_cdr3_attributions(
                            row_name=channel,
                            attributions=aa_add_padding(attribution[len(ep):], heatmap.max_cdr3_length)
                        )
                    elif 'scrambled eps' in model_name:
                        heatmap_scrambled.add_ep_attributions(
                            row_name=channel,
                            attributions=aa_add_padding(attribution[:len(ep)], heatmap.max_ep_length)
                        )
                    elif 'scrambled tcrs' in model_name:
                        heatmap_scrambled.add_cdr3_attributions(
                            row_name=channel,
                            attributions=aa_add_padding(attribution[len(ep):], heatmap.max_cdr3_length)
                        )

        # Imrex
        heatmap.compose_heatmap()
        heatmap.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename()
        )
        # Scrambled: absolut plot
        heatmap_scrambled.compose_heatmap()
        heatmap_scrambled.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename(keywords=['scrambled'])
        )
        # Scrambled: correction amount
        heatmap_scrambled.compose_heatmap_correction()
        heatmap_scrambled.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename(keywords=['correction'])
        )
        # Difference Imrex - scrambled
        heatmap.compose_heatmap(scrambled_heatmap=heatmap_scrambled)
        heatmap.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename(keywords=['diff'])
        )
        return heatmap_scrambled.heatmap

    def plot_aa_average_1d(self) -> Dict:
        """
        Creates a feature attribution heat map for each pdb complex.
        """

        # get the model attributions for both handlers
        model_attributions = {
            model.display_name: model.get_aa_norm_attributions()
            for model in self.attribution_handlers
        }
        # sequences are the same for all handlers
        sequences = self.attribution_handlers[0].get_sequences()

        # if channels were split, tune the model_attributions dictionary
        model_attributions, self.channels = get_model_attributions_dictionary(
            model_attributions=model_attributions,
            model_names=[handler.display_name for handler in self.attribution_handlers]
        )

        # get the pdbs that occur in model attributions
        pdb_list = list(model_attributions[self.attribution_handlers[0].display_name]['Ch 0'].keys())

        from plot_helper_heatmap import AAHeatmap1D

        heatmap = AAHeatmap1D()
        heatmap.extract_aa(sequences=sequences, pdb_id_subset=pdb_list)
        heatmap.initialize_rows(row_names=model_attributions[list(model_attributions.keys())[0]])

        heatmap_scrambled = AAHeatmap1D()
        heatmap_scrambled.extract_aa(sequences=sequences, pdb_id_subset=pdb_list)
        heatmap_scrambled.initialize_rows(row_names=model_attributions[list(model_attributions.keys())[0]])

        for pdb_id, sequence in sequences.items():
            if pdb_id not in pdb_list:
                continue
            ep, cdr3 = sequence
            ep_cdr3 = ep + cdr3

            for model_name, model_attribution in model_attributions.items():
                for channel, channel_attribution in model_attribution.items():
                    attributions = channel_attribution[pdb_id][self.current_method].tolist()

                    for i in range(len(attributions)):
                        if 'scrambled' not in model_name:
                            heatmap.add_aa_attribution(
                                row_name=channel, aa=ep_cdr3[i], attribution=attributions[i]
                            )
                        elif 'scrambled eps' in model_name and i < len(ep):
                            heatmap_scrambled.add_aa_attribution(
                                row_name=channel, aa=ep_cdr3[i], attribution=attributions[i]
                            )
                        elif 'scrambled tcrs' in model_name and i > len(ep):
                            heatmap_scrambled.add_aa_attribution(
                                row_name=channel, aa=ep_cdr3[i], attribution=attributions[i]
                            )

        # Imrex
        heatmap.compose_heatmap()
        heatmap.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename()
        )
        # Scrambled: absolut plot
        heatmap_scrambled.compose_heatmap()
        heatmap_scrambled.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename(keywords=['scrambled'])
        )
        # Scrambled: correction amount
        heatmap_scrambled.compose_heatmap_correction()
        heatmap_scrambled.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename(keywords=['correction'])
        )
        # Difference Imrex - scrambled
        heatmap.compose_heatmap(scrambled_heatmap=heatmap_scrambled)
        heatmap.create_plot(
            color_bar_label=f"{self.current_method} feature attribution",
            filename=self.get_comparison_plot_filename(keywords=['diff'])
        )

        return_heatmap = dict()
        for key in heatmap_scrambled.heatmap:
            return_heatmap[key] = dict()
            for i, aa in enumerate(heatmap_scrambled.aa):
                return_heatmap[key][aa] = heatmap_scrambled.heatmap[key][i]

        return return_heatmap

    def plot_aa_average_2d(self) -> Dict:
        """
        Creates a feature attribution heat map for each pdb complex.
        """

        # get the model attributions for both handlers
        model_attributions = {
            model.display_name: model.get_aa_norm_attributions()
            for model in self.attribution_handlers
        }
        # sequences are the same for all handlers
        sequences = self.attribution_handlers[0].get_sequences()

        # if channels were split, tune the model_attributions dictionary
        model_attributions, self.channels = get_model_attributions_dictionary(
            model_attributions=model_attributions,
            model_names=[handler.display_name for handler in self.attribution_handlers]
        )

        # get the pdbs that occur in model attributions
        pdb_list = list(model_attributions[self.attribution_handlers[0].display_name]['Ch 0'].keys())

        values = [
            s for pdb_id, s in sequences.items()
            if pdb_list is None or pdb_id in pdb_list
        ]
        max_ep_length, max_cdr3_length = [len(max(values, key=lambda x: len(x[i]))[i]) for i in [0, 1]]

        from plot_helper_heatmap import AAHeatmap2D

        corrections = dict()
        for channel in model_attributions[list(model_attributions.keys())[0]].keys():
            heatmap = AAHeatmap2D()
            heatmap.extract_max_ep_cdr3_length(sequences=sequences, pdb_subset=pdb_list)
            heatmap.initialize_aa_rows(sequences=sequences, pdb_id_subset=pdb_list)

            heatmap_scrambled = heatmap.copy_initialization()

            for pdb_id, sequence in sequences.items():
                if pdb_id not in pdb_list:
                    continue
                ep, cdr3 = sequence
                ep_cdr3 = ep + cdr3

                for model_name, model_attribution in model_attributions.items():
                    channel_attribution = model_attributions[model_name][channel]
                    attributions = channel_attribution[pdb_id][self.current_method].tolist()

                    attributions = np.concatenate((
                        aa_add_padding(attributions[:len(ep)], max_ep_length),
                        aa_add_padding(attributions[len(ep):], max_cdr3_length)
                    ))

                    aa_index = 0
                    for i in range(len(attributions)):
                        if np.isnan(attributions[i]):
                            continue

                        if 'scrambled' not in model_name:
                            if i < max_ep_length:
                                heatmap.add_ep_aa_attribution(
                                    aa=ep_cdr3[aa_index], position=i, attribution=attributions[i]
                                )
                            else:
                                heatmap.add_cdr3_aa_attribution(
                                    aa=ep_cdr3[aa_index], position=i - max_ep_length, attribution=attributions[i]
                                )
                        elif 'scrambled eps' in model_name and i < max_ep_length:
                            heatmap_scrambled.add_ep_aa_attribution(
                                aa=ep_cdr3[aa_index], position=i, attribution=attributions[i]
                            )
                        elif 'scrambled tcrs' in model_name and i >= max_ep_length:
                            heatmap_scrambled.add_cdr3_aa_attribution(
                                aa=ep_cdr3[aa_index], position=i - max_ep_length, attribution=attributions[i]
                            )
                        aa_index += 1

            # Imrex
            heatmap.compose_heatmap()
            heatmap.create_plot(
                color_bar_label=f"{self.current_method} feature attribution",
                filename=self.get_comparison_plot_filename(keywords=[channel])
            )
            # Scrambled: absolut plot
            heatmap_scrambled.compose_heatmap()
            heatmap_scrambled.create_plot(
                color_bar_label=f"{self.current_method} feature attribution",
                filename=self.get_comparison_plot_filename(keywords=[channel, 'scrambled'])
            )
            # Scrambled: correction amount
            heatmap_scrambled.compose_heatmap_correction()
            heatmap_scrambled.create_plot(
                color_bar_label=f"{self.current_method} feature attribution",
                filename=self.get_comparison_plot_filename(keywords=[channel, 'correction'])
            )
            # Difference Imrex - scrambled
            heatmap.compose_heatmap(scrambled_heatmap=heatmap_scrambled)
            heatmap.create_plot(
                color_bar_label=f"{self.current_method} feature attribution",
                filename=self.get_comparison_plot_filename(keywords=[channel, 'diff'])
            )

            corrections[channel] = heatmap_scrambled.heatmap
            for key in corrections[channel].keys():
                corrections[channel][key][max_ep_length] = np.nan

        return corrections


# TODO: below methods involve other models than ImRex. They were ignored for channel split
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
        plt.close()


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
        plt.close()
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


if __name__ == "__main__":
    plotter = Plotter()
    plotter.combined = True
    plotter.calculate_and_plot_positional_correction()
    plotter.calculate_and_plot_aa_1d_corrections()
    plotter.calculate_and_plot_aa_2d_corrections()
    plotter.calculate_and_plot_aa_1d_correction_on_positional_correction()
    plotter.calculate_and_plot_positional_correction_on_aa_1d_correction()
