import os
import pickle

import numpy as np
import pandas as pd
import saliency.core as saliency
import shap
import tensorflow as tf
from scipy.stats import pearsonr

from src.util import imrex_remove_padding, imgs_to_list_of_feature_lists, generate_path_inputs, setup_logger, \
    get_distance_matrices, normalize_2d, rmse, matrix_to_aa, list_feature_list_to_list_imgs, img_to_feature_list, \
    get_mean_feature_values, integral_approximation


class ImrexAttributionsHandler:
    """
    Class that manages feature attribution extraction from any pretrained ImRex model. This class calculates and
    saves all required (intermediary) results, e.g. featura attributions, distance matrices and RMSE's.
    """

    def __init__(self, name, display_name, model_path, image_path, save_folder):
        """
        Parameters
        ----------
        name            Internal name, results will be saved in the folder [save_folder]/[name]
        display_name    Name displayed by e.g. plot.py
        model_path      Path to the ImRex model
        image_path      Path to the ImRex input used for feature attribution extraction
        save_folder     Path to the save folder, results will be saved in the folder [save_folder]/[name]
        """
        self.name = name
        self.display_name = display_name
        self.model_path = model_path
        self.image_path = image_path
        self.save_folder = save_folder

        self.model = None
        self.input_images = None
        self.attributions = None
        self.norm_attributions = None
        self.distances = None
        self.norm_distances = None
        self.errors = None
        self.random_error = None
        self.correlation = None
        self.random_correlation = None

        self.aa_attributions = None
        self.aa_norm_attributions = None
        self.aa_distances = None
        self.aa_norm_distances = None
        self.aa_errors = None
        self.aa_errors_ps = None
        self.aa_random_error = None
        self.aa_random_error_ps = None
        self.aa_correlation = None
        self.aa_correlation_ps = None
        self.aa_random_correlation = None
        self.aa_random_correlation_ps = None

        # Create folder to save results if it does not exists yet
        if not os.path.exists(f"{self.save_folder}/{self.name}"):
            os.makedirs(f"{self.save_folder}/{self.name}")

        self.logger = setup_logger(self.name)

    @staticmethod
    def dict_key(pdb_id, channel):
        return f'{pdb_id}_{channel}'

    @staticmethod
    def split_feature_attribution_matrix(matrix, w, h):
        if len(matrix.shape) == 2 and matrix.shape[-1] == 4:
            return [imrex_remove_padding(matrix[:, channel], w, h) for channel in range(4)]
        elif len(matrix.shape) == 3 and matrix.shape[-1] == 4:
            return [imrex_remove_padding(matrix[:, :, channel], w, h) for channel in range(4)]
        elif len(matrix.shape) == 4 and matrix.shape[-1] == 4:
            return [imrex_remove_padding(matrix[:, :, :, channel], w, h) for channel in range(4)]
        else:
            return [imrex_remove_padding(matrix, w, h)]

    @staticmethod
    def image_channel(image, channel):
        if channel is not None:
            image = image.numpy()
            for i in range(4):
                image[:, :, i] = image[:, :, channel]
            image = tf.convert_to_tensor(image)
        return image

    def get_sequences(self):
        """
        Returns
        -------
        Epitope and CDR3 sequence for each PDB complex
        """

        tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv")
        sequences = {}
        for pdb_id, cdr3, ep in zip(tcr3df['PDB_ID'], tcr3df['cdr3'], tcr3df['antigen.epitope']):
            sequences[pdb_id] = (ep, cdr3)
        return sequences

    def get_attributions(self, overwrite=False, split_channels=False):
        """
        Calculate and save all feature attributions, loaded from file if already saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite       If True, attributions will be re-calculated and overwrite the old saved result
        split_channels  If True, attribution calculation will be performed for each input channel separately

        Returns
        -------
        A dict with attributions for each PDB complex
        """

        def attributions():
            # Get data and ImRex model
            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv")
            self.model, self.input_images = self.imrex_setup()
            print(self.model)

            # Calculate SHAP feature attributions with 'mean' and 'dist' background methods
            if split_channels:
                if os.path.isfile('temp_shaps_mean_split.txt') and os.path.isfile('temp_shaps_dist_split.txt'):
                    dim = tuple(
                        [len(self.input_images)] + list(self.input_images[list(self.input_images.keys())[0]].shape))
                    shaps_mean_calc = np.loadtxt('temp_shaps_mean_split.txt').reshape(dim)
                    shaps_dist_calc = np.loadtxt('temp_shaps_dist_split.txt').reshape(dim)
                else:
                    shaps_mean_calc = self.get_shap_attributions(background='mean', split_channels=split_channels)
                    np.savetxt('temp_shaps_mean_split.txt', shaps_mean_calc.reshape(shaps_mean_calc.shape[0], -1))
                    shaps_dist_calc = self.get_shap_attributions(background='dist', split_channels=split_channels)
                    np.savetxt('temp_shaps_dist_split.txt', shaps_dist_calc.reshape(shaps_dist_calc.shape[0], -1))
            else:
                if os.path.isfile('temp_shaps_mean_combined.txt') and os.path.isfile('temp_shaps_dist_combined.txt'):
                    dim = tuple(
                        [len(self.input_images)] + list(self.input_images[list(self.input_images.keys())[0]].shape)[:-1])
                    shaps_mean_calc = np.loadtxt('temp_shaps_mean_combined.txt').reshape(dim)
                    shaps_dist_calc = np.loadtxt('temp_shaps_dist_combined.txt').reshape(dim)
                else:
                    shaps_mean_calc = self.get_shap_attributions(background='mean', split_channels=split_channels)
                    np.savetxt('temp_shaps_mean_combined.txt', shaps_mean_calc.reshape(shaps_mean_calc.shape[0], -1))
                    shaps_dist_calc = self.get_shap_attributions(background='dist', split_channels=split_channels)
                    np.savetxt('temp_shaps_dist_combined.txt', shaps_dist_calc.reshape(shaps_dist_calc.shape[0], -1))

            attribution_dict = {}

            property_zip = list(zip(
                tcr3df['PDB_ID'],
                tcr3df['cdr3'],
                tcr3df['antigen.epitope'],
                shaps_mean_calc,
                shaps_dist_calc
            ))

            for pdb_id, cdr3, ep, shap_mean, shap_dist in property_zip:
                pdb_dict = {
                    'SHAP mean': shap_mean,
                    'SHAP BGdist': shap_dist,
                    'IG': self.get_ig_attribution(input_img=self.input_images[pdb_id], split_channels=split_channels)
                }

                saliency_methods = ['Vanilla', 'SmoothGrad', 'VanillaIG', 'SmoothGradIG', 'GuidedIG', 'XRAI',
                                    'BlurIG', 'SmoothGradBlurIG']

                # Add attributions of all saliency methods to dict
                for method in saliency_methods:
                    pdb_dict[method] = self.get_saliency_attribution(
                        im=self.input_images[pdb_id], method=method, split_channels=split_channels)
                    # if split_channels:
                    #     pdb_dict[method] = np.moveaxis(pdb_dict[method], 0, 2)
                    # else:
                    #     pdb_dict[method] = pdb_dict[method].reshape(pdb_dict[method].shape[1:])

                # Split channels if applicable + remove padding
                if split_channels:
                    for method in pdb_dict:
                        pdb_dict[method] = self.split_feature_attribution_matrix(
                            matrix=pdb_dict[method],
                            w=len(cdr3),
                            h=len(ep)
                        )

                    for method in pdb_dict:
                        for channel in range(4):
                            if self.dict_key(pdb_id=pdb_id, channel=channel) not in attribution_dict:
                                attribution_dict[self.dict_key(pdb_id=pdb_id, channel=channel)] = {}

                            if method == 'XRAI':
                                attribution_dict[self.dict_key(pdb_id=pdb_id, channel=channel)][method] = \
                                    pdb_dict[method][0]
                            else:
                                attribution_dict[self.dict_key(pdb_id=pdb_id, channel=channel)][method] = \
                                    pdb_dict[method][channel]
                else:
                    for method in pdb_dict:
                        pdb_dict[method] = imrex_remove_padding(pdb_dict[method], len(cdr3), len(ep))
                    attribution_dict[pdb_id] = pdb_dict

                self.logger.info(f'{pdb_id} done')

            if split_channels:
                attribution_dict_combined = self.get_attributions(overwrite=overwrite, split_channels=False)
                attribution_dict_combined = {
                    f'{k}_combined': v
                    for k, v in attribution_dict_combined.items()
                }
                attribution_dict.update(attribution_dict_combined)

            return attribution_dict

        self.attributions = self.__handle_getter(attributions, self.attributions, overwrite,
                                                 f"{self.name}/attributions.p", 'attributions')
        return self.attributions

    def get_distances(self, overwrite=False):
        """
        Calculate and save all distance matrices, loaded from file if already saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, distance matrices will be re-calculated and overwrite the old saved result

        Returns
        -------
        A dict with a distance matrix for each PDB complex
        """
        self.distances = self.__handle_getter(get_distance_matrices, self.distances, overwrite, "distance_matrices.p",
                                              'distances')

        return self.distances

    def get_norm_distances(self, overwrite=False):
        """
        Calculate and save all normalized distance matrices, loaded from file if already saved previously
        (and overwrite=False). Normalization is done by taking 1/value for each value in the distance matrix.

        Parameters
        ----------
        overwrite   If True, normalized distance matrices will be re-calculated and overwrite the old saved result, the
                    original distances will not be re-calculated.

        Returns
        -------
        A dict with the normalized distance matrix for each PDB complex
        """

        def norm_distances():
            distances = self.get_distances()
            norm_distances = {}
            for pdb_id, dist_matrix in distances.items():
                norm_distances[pdb_id] = normalize_2d(1 / dist_matrix)
            return norm_distances

        self.norm_distances = self.__handle_getter(norm_distances, self.norm_distances, overwrite,
                                                   "norm_distance_matrices.p", 'normalized distances')
        return self.norm_distances

    def get_errors(self, overwrite=False):
        """
        Calculate and save all RMSE's between the attributions and distances, loaded from file if already saved
        previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, RMSE's will be re-calculated and overwrite the old saved result, the original distances and
                    attributions will not be re-calculated.

        Returns
        -------
        A dict with the RMSE for each PDB complex
        """

        def errors():
            attributions = self.get_attributions()
            distances = self.get_distances()
            error_dict = {}
            for pdb_id, methods in attributions.items():
                dist = distances[pdb_id.split('_')[0]]
                error_pdb = {}
                for method, attribution in methods.items():
                    error_pdb[method] = rmse(dist, attribution)
                error_dict[pdb_id] = error_pdb
                self.logger.info(f'{pdb_id} done')
            return error_dict

        self.errors = self.__handle_getter(errors, self.errors, overwrite, f"{self.name}/errors.p", 'errors')
        return self.errors

    def get_random_error(self, overwrite=False):
        """
        Calculate and save the random RMSE by taking the RMSE between a random feature attribution matrix and the
        distance matrix of each PDB complex 1000 times, loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, the random error will be re-calculated and overwrite the old saved result, the original
                    distances will not be re-calculated.

        Returns
        -------
        The tuple (mean_random_error, std_random_error)
        """

        def random_error():
            distances = self.get_distances()
            random_errors = []
            for pdb_id, dist_m in distances.items():
                # Repeat 1000 times
                for i in range(1000):
                    # Get a random matix with same shape as distance matrix
                    random_m = np.random.rand(*dist_m.shape)
                    # calculate RMSE
                    random_errors.append(rmse(dist_m, random_m))
                self.logger.info(f'{pdb_id} done')
            # Return mean and std
            return np.mean(random_errors), np.std(random_errors)

        self.random_error = self.__handle_getter(random_error, self.random_error, overwrite, "imrex_random_error.p",
                                                 'random_error')
        return self.random_error

    def get_correlation(self, overwrite=False):
        """
        Calculate and save the correlation between the attributions and distances, loaded from file if already saved
        previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, the correlation will be re-calculated and overwrite the old saved result, the original
                    distances and attributions will not be re-calculated.

        Returns
        -------
        A dict with the Correlation for each PDB complex
        """

        def correlation():
            attributions = self.get_attributions()
            distances = self.get_distances()
            pearson_corr_dict = {}
            for pdb_id, methods in attributions.items():
                dist = distances[pdb_id.split('_')[0]]
                pearson_corr_pdb = {}
                for method, attribution in methods.items():
                    pearson_corr_pdb[method] = pearsonr(dist.flatten(), attribution.flatten())[0]
                pearson_corr_dict[pdb_id] = pearson_corr_pdb
                self.logger.info(f'{pdb_id} done')
            return pearson_corr_dict

        self.correlation = self.__handle_getter(correlation, self.correlation, overwrite, f"{self.name}/correlation.p",
                                                f'correlation')
        return self.correlation

    def get_random_correlation(self, overwrite=False):
        """
        Calculate and save the random correlation by taking the correlation between a random feature attribution matrix
        and the distance matrix of each PDB complex 1000 times, loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, the random correlation will be re-calculated and overwrite the old saved result, the
                    original distances will not be re-calculated.

        Returns
        -------
        The tuple (mean_random_corr, std_random_corr)
        """

        def random_correlation():
            distances = self.get_distances()
            random_pearson_corr = []
            for pdb_id, dist_m in distances.items():
                # Repeat 1000 times
                for i in range(1000):
                    # Get a random matrix with same shape as distance matrix
                    random_m = np.random.rand(*dist_m.shape)
                    # calculate correlation
                    random_pearson_corr.append(pearsonr(dist_m.flatten(), random_m.flatten())[0])
                self.logger.info(f'{pdb_id} done')
            # Return mean and std
            return np.mean(random_pearson_corr), np.std(random_pearson_corr)

        self.random_correlation = self.__handle_getter(random_correlation, self.random_correlation, overwrite,
                                                       "imrex_random_correlation.p",
                                                       'random correlation')
        return self.random_correlation

    def get_aa_attributions(self, overwrite=False):
        """
        Calculate and save the feature attributions merged per AA, loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, AA feature attributions will be re-calculated and overwrite the old saved result,
                    the original attributions will not be re-calculated.

        Returns
        -------
        A dict with the AA feature attributions for each PDB complex
        """

        def aa_attributions():
            attributions = self.get_attributions()
            aa_attributions = {}
            for pdb_id, methods in attributions.items():
                pdb_attributions = {}
                for method, attribution in methods.items():
                    pdb_attributions[method] = matrix_to_aa(attribution, 'max')
                aa_attributions[pdb_id] = pdb_attributions
            return aa_attributions

        self.aa_attributions = self.__handle_getter(aa_attributions, self.aa_attributions, overwrite,
                                                    f"{self.name}/aa_attributions.p", 'AA attributions')
        return self.aa_attributions

    def get_norm_attributions(self, overwrite=False):
        """
        Calculate and save the normalized feature attributions, loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, normalized feature attributions will be re-calculated and overwrite the old saved result,
                    the original attributions will not be re-calculated.

        Returns
        -------
        A dict with the normalized feature attributions for each PDB complex
        """

        def norm_attributions():
            attributions = self.get_attributions()
            norm_attributions = {}
            for pdb_id, methods in attributions.items():
                norm_attributions[pdb_id] = {}
                for method, attribution in methods.items():
                    norm_attributions[pdb_id][method] = normalize_2d(attribution)
            return norm_attributions

        self.norm_attributions = self.__handle_getter(norm_attributions, self.norm_attributions, overwrite,
                                                      f"{self.name}/norm_attributions.p", 'Normalized attributions')
        return self.norm_attributions

    def get_aa_norm_attributions(self, overwrite=False):
        """
        Calculate and save the normalized AA feature attributions, loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, normalized AA feature attributions will be re-calculated and overwrite the old saved
                    result, the original AA feature attributions will not be re-calculated.

        Returns
        -------
        A dict with the normalized AA feature attributions for each PDB complex
        """

        def aa_norm_attributions():
            aa_attributions = self.get_aa_attributions()
            aa_norm_attributions = {}
            for pdb_id, methods in aa_attributions.items():
                aa_norm_attributions[pdb_id] = {}
                for method, attribution in methods.items():
                    aa_norm_attributions[pdb_id][method] = normalize_2d(attribution)
            return aa_norm_attributions

        self.aa_norm_attributions = self.__handle_getter(aa_norm_attributions, self.aa_norm_attributions, overwrite,
                                                         f"{self.name}/aa_norm_attributions.p",
                                                         'AA normalized attributions')
        return self.aa_norm_attributions

    def get_aa_distances(self, overwrite=False):
        """
        Calculate and save the distance matrices merged per AA, loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, AA distances will be re-calculated and overwrite the old saved result, the original
                    distances will not be re-calculated.

        Returns
        -------
        A dict with the AA distances for each PDB complex
        """

        def aa_distances():
            distances = self.get_distances()
            aa_distances = {}
            for pdb_id, dist_matrix in distances.items():
                aa_distances[pdb_id] = matrix_to_aa(dist_matrix, 'min')
            return aa_distances

        self.aa_distances = self.__handle_getter(aa_distances, self.aa_distances, overwrite, "aa_distances.p",
                                                 'AA distances')
        return self.aa_distances

    def get_aa_norm_distances(self, overwrite=False):
        """
        Calculate and save the normalized AA distances, loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, normalized AA distances will be re-calculated and overwrite the old saved result, the
                    original AA distances will not be re-calculated.

        Returns
        -------
        A dict with the normalized AA distances for each PDB complex
        """

        def aa_norm_distances():
            aa_distances = self.get_aa_distances()
            aa_norm_distances = {}
            for pdb_id, dist_matrix in aa_distances.items():
                aa_norm_distances[pdb_id] = normalize_2d(1 / dist_matrix)
            return aa_norm_distances

        self.aa_norm_distances = self.__handle_getter(aa_norm_distances, self.aa_norm_distances, overwrite,
                                                      "aa_norm_distances.p", 'AA normalized distances')
        return self.aa_norm_distances

    def get_aa_errors(self, overwrite=False):
        """
        Calculate and save the RMSE's between the AA distance and AA feature attributions, loaded from file if already
        saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, AA RMSE's will be re-calculated and overwrite the old saved result, the original AA
                    distances and AA attributions will not be re-calculated.

        Returns
        -------
        A dict with the AA RMSE for each PDB complex
        """

        def aa_errors():
            aa_attributions = self.get_aa_attributions()
            aa_distances = self.get_aa_distances()
            error_dict = {}
            for pdb_id, methods in aa_attributions.items():
                aa_dist = aa_distances[pdb_id.split('_')[0]]
                error_pdb = {}
                for method, attribution in methods.items():
                    error_pdb[method] = rmse(aa_dist, attribution)
                error_dict[pdb_id] = error_pdb
            return error_dict

        self.aa_errors = self.__handle_getter(aa_errors, self.aa_errors, overwrite, f"{self.name}/aa_errors.p",
                                              'AA errors')
        return self.aa_errors

    def get_aa_errors_ps(self, overwrite=False):
        """
        Calculate and save the AA RMSE's per sequence (epitope and CDR3) separately, loaded from file if already saved
        previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, AA RMSE's per sequence will be re-calculated and overwrite the old saved result, the
                    original AA distances and AA attributions will not be re-calculated.

        Returns
        -------
        A dict with the AA RMSE per sequence for each PDB complex
        """

        def aa_errors_ps():
            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv", index_col='PDB_ID')
            aa_attributions = self.get_aa_attributions()
            aa_distances = self.get_aa_distances()
            error_ps_dict = {}
            for pdb_id, methods in aa_attributions.items():
                aa_dist = aa_distances[pdb_id.split('_')[0]]
                # Get starting index of epitope
                seq_split = len(tcr3df.loc[pdb_id.split('_')[0]]['antigen.epitope'])
                error_ps_dict[pdb_id] = {}
                for method, attribution in methods.items():
                    # Split distance and attributions arrays in epitope and CDR3
                    ep_aa_dist = aa_dist[:seq_split]
                    cdr3_aa_dist = aa_dist[seq_split:]
                    ep_attribution = attribution[:seq_split]
                    cdr3_attribution = attribution[seq_split:]
                    # Calculate rmse separately for epitope and CDR3
                    error_ps_dict[pdb_id][method] = (rmse(ep_aa_dist, ep_attribution),
                                                     rmse(cdr3_aa_dist, cdr3_attribution))
            return error_ps_dict

        self.aa_errors_ps = self.__handle_getter(aa_errors_ps, self.aa_errors_ps, overwrite,
                                                 f"{self.name}/aa_errors_ps.p", 'AA errors per sequence')
        return self.aa_errors_ps

    def get_aa_random_error(self, overwrite=False):
        """
        Calculate and save the random RMSE between the AA distance and AA feature attributions, loaded from file if
        already saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, the AA random RMSE will be re-calculated and overwrite the old saved result, the original
                    AA distances will not be re-calculated.

        Returns
        -------
        A dict with the AA random RMSE for each PDB complex
        """

        def aa_random_error():
            errors = []
            aa_distances = self.get_aa_distances()
            for pdb_id, aa_dist in aa_distances.items():
                for i in range(1000):
                    # Get random array with same length as distance array
                    random_a = np.random.rand(len(aa_dist))
                    errors.append(rmse(aa_dist, random_a))
            return np.mean(errors), np.std(errors)

        self.aa_random_error = self.__handle_getter(aa_random_error, self.aa_random_error, overwrite,
                                                    f"aa_random_error.p", 'AA random error')
        return self.aa_random_error

    def get_aa_random_error_ps(self, overwrite=False):
        """
        Calculate and save the AA random RMSE's per sequence (epitope and CDR3), loaded from file if already saved
        previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, the AA random RMSE per sequence will be re-calculated and overwrite the old saved result,
                    the original AA distances will not be re-calculated.

        Returns
        -------
        A dict with the AA random RMSE per sequence for each PDB complex
        """

        def aa_random_error_ps():
            errors_ep = []
            errors_cdr3 = []
            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv", index_col='PDB_ID')
            aa_distances = self.get_aa_distances()
            for pdb_id, aa_dist in aa_distances.items():
                ep_len = len(tcr3df.loc[pdb_id]['antigen.epitope'])
                cdr3_len = len(tcr3df.loc[pdb_id]['cdr3'])
                for i in range(1000):
                    random_a_ep = np.random.rand(ep_len)
                    aa_dist_ep = aa_dist[:ep_len]
                    errors_ep.append(rmse(aa_dist_ep, random_a_ep))
                    random_a_cdr3 = np.random.rand(cdr3_len)
                    aa_dist_cdr3 = aa_dist[ep_len:]
                    errors_cdr3.append(rmse(aa_dist_cdr3, random_a_cdr3))
                print(f'{pdb_id} done')
            return (np.mean(errors_ep), np.std(errors_ep)), (np.mean(errors_cdr3), np.std(errors_cdr3))

        self.aa_random_error_ps = self.__handle_getter(aa_random_error_ps, self.aa_random_error_ps, overwrite,
                                                       f"aa_random_error_ps.p", 'AA random error per sequence')
        return self.aa_random_error_ps

    def get_aa_correlation(self, overwrite=False):
        """
        Calculate and save the correlation between the AA distance and AA feature attributions, loaded from file if
        already saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, AA correlation will be re-calculated and overwrite the old saved result, the original AA
                    distances and AA attributions will not be re-calculated.

        Returns
        -------
        A dict with the AA correlation for each PDB complex
        """

        def aa_correlation():
            aa_attributions = self.get_aa_attributions()
            aa_distances = self.get_aa_distances()
            pearson_correlation_dict = {}
            for pdb_id, methods in aa_attributions.items():
                aa_dist = aa_distances[pdb_id.split('_')[0]]
                pearson_corr_pdb = {}
                for method, attribution in methods.items():
                    pearson_corr_pdb[method] = pearsonr(aa_dist.flatten(), attribution.flatten())[0]
                pearson_correlation_dict[pdb_id] = pearson_corr_pdb
            return pearson_correlation_dict

        self.aa_correlation = self.__handle_getter(aa_correlation, self.aa_correlation, overwrite,
                                                   f"{self.name}/aa_correlation.p", 'AA correlation')
        return self.aa_correlation

    def get_aa_correlation_ps(self, overwrite=False):
        """
        Calculate and save the AA correlation per sequence (epitope and CDR3) separately, loaded from file if already
        saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, AA correlation per sequence will be re-calculated and overwrite the old saved result, the
                    original AA distances and AA attributions will not be re-calculated.

        Returns
        -------
        A dict with the AA correlation per sequence for each PDB complex
        """

        def aa_correlation_ps():
            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv", index_col='PDB_ID')
            aa_attributions = self.get_aa_attributions()
            aa_distances = self.get_aa_distances()
            pearson_correlation_dict = {}
            for pdb_id, methods in aa_attributions.items():
                aa_dist = aa_distances[pdb_id.split('_')[0]]
                # Get starting index of epitope
                seq_split = len(tcr3df.loc[pdb_id.split('_')[0]]['antigen.epitope'])
                pearson_correlation_dict[pdb_id] = {}
                for method, attribution in methods.items():
                    # Split distance and attributions arrays in epitope and CDR3
                    ep_aa_dist = aa_dist[:seq_split]
                    cdr3_aa_dist = aa_dist[seq_split:]
                    ep_attribution = attribution[:seq_split]
                    cdr3_attribution = attribution[seq_split:]
                    # Calculate correlation separately for epitope and CDR3
                    pearson_correlation_dict[pdb_id][method] = (
                        pearsonr(ep_aa_dist.flatten(), ep_attribution.flatten())[0],
                        pearsonr(cdr3_aa_dist.flatten(), cdr3_attribution.flatten())[0])
            return pearson_correlation_dict

        self.aa_correlation_ps = self.__handle_getter(aa_correlation_ps, self.aa_correlation_ps, overwrite,
                                                      f"{self.name}/aa_correlation_ps.p", 'AA correlation per sequence')
        return self.aa_correlation_ps

    def get_aa_random_correlation(self, overwrite=False):
        """
        Calculate and save the random correlation between the AA distance and AA feature attributions, loaded from file
        if already saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, the AA random correlation will be re-calculated and overwrite the old saved result, the
                    original AA distances will not be re-calculated.

        Returns
        -------
        A dict with the AA random correlation for each PDB complex
        """

        def aa_random_correlation():
            pearson_correlation = []
            aa_distances = self.get_aa_distances()
            for pdb_id, aa_dist in aa_distances.items():
                for i in range(1000):
                    # Get random array with same length as distance array
                    random_a = np.random.rand(len(aa_dist))
                    pearson_correlation.append(pearsonr(aa_dist.flatten(), random_a.flatten())[0])
            return np.mean(pearson_correlation), np.std(pearson_correlation)

        self.aa_random_correlation = self.__handle_getter(aa_random_correlation, self.aa_random_correlation, overwrite,
                                                          f"aa_random_correlation.p", 'AA random correlation')
        return self.aa_random_correlation

    def get_aa_random_correlation_ps(self, overwrite=False):
        """
        Calculate and save the AA random correlation per sequence (epitope and CDR3), loaded from file if already saved
        previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, the AA random correlation per sequence will be re-calculated and overwrite the old saved
                    result, the original AA distances will not be re-calculated.

        Returns
        -------
        A dict with the AA random correlation per sequence for each PDB complex
        """

        def aa_random_correlation_ps():
            pearson_correlation_ep = []
            pearson_correlation_cdr3 = []
            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv", index_col='PDB_ID')
            aa_distances = self.get_aa_distances()
            for pdb_id, aa_dist in aa_distances.items():
                ep_len = len(tcr3df.loc[pdb_id]['antigen.epitope'])
                cdr3_len = len(tcr3df.loc[pdb_id]['cdr3'])
                for i in range(1000):
                    random_a_ep = np.random.rand(ep_len)
                    aa_dist_ep = aa_dist[:ep_len]
                    random_a_cdr3 = np.random.rand(cdr3_len)
                    aa_dist_cdr3 = aa_dist[ep_len:]
                    pearson_correlation_ep.append(pearsonr(aa_dist_ep.flatten(), random_a_ep.flatten())[0])
                    pearson_correlation_cdr3.append(pearsonr(aa_dist_cdr3.flatten(), random_a_cdr3.flatten())[0])
                print(f'{pdb_id} done')
            return (np.mean(pearson_correlation_ep), np.std(pearson_correlation_ep)), (
                np.mean(pearson_correlation_cdr3), np.std(pearson_correlation_cdr3))

        self.aa_random_correlation_ps = self.__handle_getter(aa_random_correlation_ps, self.aa_random_correlation_ps,
                                                             overwrite,
                                                             f"aa_random_correlation_ps.p",
                                                             'AA random correlation per sequence')
        return self.aa_random_correlation_ps

    def set_all(self, overwrite=False, split_channels=False):
        """
        Calculate and save all results for this model, results are loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, all results will be re-calculated and overwrite the old saved results.
        """
        self.get_attributions(overwrite, split_channels)
        self.get_distances(overwrite)
        self.get_norm_distances(overwrite)
        self.get_errors(overwrite)
        self.get_random_error(overwrite)
        self.get_aa_attributions(overwrite)
        self.get_norm_attributions(overwrite)
        self.get_aa_norm_attributions(overwrite)
        self.get_aa_distances(overwrite)
        self.get_aa_norm_distances(overwrite)
        self.get_aa_errors(overwrite)
        self.get_aa_errors_ps(overwrite)
        self.get_aa_random_error(overwrite)
        self.get_aa_random_error_ps(overwrite)
        self.get_correlation(overwrite)
        self.get_random_correlation(overwrite)
        self.get_aa_correlation(overwrite)
        self.get_aa_correlation_ps(overwrite)
        self.get_aa_random_correlation(overwrite)
        self.get_aa_random_correlation_ps(overwrite)

    def imrex_setup(self):
        """
        Loads pretrained ImRex model and input images

        Returns
        -------
        Imrex model, dict: input for each PDB complex
        """
        model = tf.keras.models.load_model(self.model_path)
        input_imgs = {}
        for f in sorted(os.listdir(self.image_path)):
            if f.endswith('.zip'):
                continue
            input_imgs[f[:-4]] = tf.cast(tf.convert_to_tensor(pickle.load(open(self.image_path + f, 'rb'))), tf.float32)
        return model, input_imgs

    def make_prediction(self, img):
        """
        Make a single prediction with the ImRex model

        Parameters
        ----------
        img     The input image

        Returns
        -------
        The predicted binding class (0 or 1)
        """
        # Prepare input shape
        image_batch = tf.expand_dims(img, 0)
        # Make prediction
        prob = self.model(image_batch)
        prediction = tf.math.round(prob)
        return prediction

    def get_shap_attributions(self, background, split_channels):
        """
        Calculate the SHAP feature attributions based on the given background method

        Parameters
        ----------
        background      'mean' or 'dist', with 'mean' the average input is taken as background, with 'dist' the
                        background is sampled from the distribution of all samples
        split_channels  If True, attribution calculation will be performed for each input channel separately

        Returns
        -------
        A feature attribution matrix for each input sample
        """

        def shap_f(z):
            # Convert the feature input to image input and make a prediction
            return self.model.predict(list_feature_list_to_list_imgs(z))

        # Make an array of the input images
        input_imgs = np.array(list(self.input_images.values()))

        if background == 'mean':
            # Set up the explainer with average background: the average values of the input images is calculated and
            # converted to a feature list
            explainer = shap.KernelExplainer(shap_f,
                                             img_to_feature_list(get_mean_feature_values(input_imgs))[None, ...])
        elif background == 'dist':
            # Set up the explainer with background distribution: the input images are converted to a list of feature
            # lists
            explainer = shap.KernelExplainer(shap_f, imgs_to_list_of_feature_lists(input_imgs))
        else:
            return

        # Calculate the SHAP values
        shap_values = explainer.shap_values(imgs_to_list_of_feature_lists(input_imgs), nsamples=1000)
        # Convert them back to a list of images
        attributions = list_feature_list_to_list_imgs(shap_values)
        # Take the absolute value and merge per feature channel
        if split_channels:
            return tf.math.abs(attributions).numpy()
        else:
            return tf.reduce_sum(tf.math.abs(attributions), axis=-1).numpy()

    def get_ig_attribution(self, input_img, split_channels):
        """
        Calculate the Integrated Gradients (IG) feature attribution for a single sample

        Parameters
        ----------
        input_img   The input for which to calculate the IG feature attributions
        split_channels     todo

        Returns
        -------
        The IG feature attribution matrix for this sample
        """

        # Baseline for CMYK format. The picture is white in this 4-channel case
        baseline = tf.zeros(shape=(20, 11, 4))

        # Linear interpolation path between baseline and input images at 50 alpha intervals
        m_steps = 50
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

        # generate the interpolated images
        path_inputs = generate_path_inputs(baseline=baseline, input=input_img, alphas=alphas)

        # Calculate the gradients for each image along the interpolation path with respect to the correct output
        with tf.GradientTape() as tape:
            tape.watch(path_inputs)
            outputs = self.model(path_inputs)
        path_gradients = tape.gradient(outputs, path_inputs)

        # Apply Riemann Trapozoidal integral approximation
        ig = integral_approximation(gradients=path_gradients, method='riemann_trapezoidal')

        attributions = (input_img - baseline) * ig

        # Take the absolute value and merge per feature channel
        if split_channels:
            return tf.math.abs(attributions).numpy()
        else:
            return tf.reduce_sum(tf.math.abs(attributions), axis=-1).numpy()

    def get_saliency_attribution(self, im, method, split_channels):
        """
        Calculate the feature attributions for a single input with a specific 'saliency' method

        Parameters
        ----------
        im                 Input image
        method             Saliency method to use for feature attribution extraction
        split_channels     todo

        Returns
        -------
        Feature attribution matrix for this sample
        """
        # Baseline for CMYK format. The picture is white in this 4-channel case
        baseline = tf.zeros(shape=(20, 11, 4))

        # Definition of the function to compute the gradients, it will be passed to the various saliency methods
        # functions
        def call_model_function(images, call_model_args=None, expected_keys=None):
            images = tf.convert_to_tensor(images)
            with tf.GradientTape() as tape:
                tape.watch(images)
                output_layer = self.model(images)
                gradients = np.array(tape.gradient(output_layer, images))
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

        prediction = int(self.make_prediction(im).numpy())
        call_model_args = {'class_idx_str': prediction}

        # Compute saliency attribution mask
        if method == 'Vanilla':
            saliency_object = saliency.GradientSaliency()
            mask = saliency_object.GetMask(im, call_model_function, call_model_args)
        elif method == 'SmoothGrad':
            saliency_object = saliency.GradientSaliency()
            mask = saliency_object.GetSmoothedMask(im, call_model_function, call_model_args)
        elif method == 'VanillaIG':
            saliency_object = saliency.IntegratedGradients()
            mask = saliency_object.GetMask(im, call_model_function, call_model_args,
                                           x_steps=50, x_baseline=baseline, batch_size=128)
        elif method == 'SmoothGradIG':
            saliency_object = saliency.IntegratedGradients()
            mask = saliency_object.GetSmoothedMask(im, call_model_function, call_model_args,
                                                   x_steps=50, x_baseline=baseline, batch_size=128)
        elif method == 'GuidedIG':
            saliency_object = saliency.GuidedIG()
            mask = saliency_object.GetMask(im, call_model_function, call_model_args, x_steps=50,
                                           x_baseline=baseline, max_dist=1.0, fraction=0.5)
        elif method == 'XRAI':
            saliency_object = saliency.XRAI()
            mask = saliency_object.GetMask(im.numpy(), call_model_function, call_model_args, batch_size=128)
        elif method == 'BlurIG':
            saliency_object = saliency.BlurIG()  # Blur Integrated Gradients
            mask = saliency_object.GetMask(im, call_model_function, call_model_args, batch_size=128)
        elif method == 'SmoothGradBlurIG':
            saliency_object = saliency.BlurIG()  # Blur Integrated Gradients
            mask = saliency_object.GetSmoothedMask(im, call_model_function, call_model_args, batch_size=128)
        else:
            self.logger.error(f'Attribution method {method} not recognized')
            return

        if method != 'XRAI':
            if split_channels:
                if isinstance(mask, np.ndarray):
                    return mask
                else:
                    return np.array(mask)  # TODO!!
            else:
                return saliency.VisualizeImageGrayscale(mask)
        else:
            return mask

    def __handle_getter(self, func, self_ret, overwrite, path, name):
        """
        Function that manages saving and loading all intermediate results.
        If the internal variable is already set and overwrite=False: return internal variable
        If overwrite=True or this result was not saved yet: calculate the result, save to file and set internal variable
        If overwrite=False and result was already saved: load result from file and set internal variable

        Parameters
        ----------
        func        Function to call when calculation is required
        self_ret    Internal variable to return when already set
        overwrite   If True, result will always be (re-)calculated
        path        Path (relative to the save_folder) on which to save or from which to load the intermediate result
        name        Name of the calculation, only used for logging

        Returns
        -------
        The result
        """
        if self_ret is not None and not overwrite:
            return self_ret
        path = f"{self.save_folder}/{path}"
        if not os.path.exists(path) or overwrite:
            self.logger.info(f"Overwrite=True, re-calculating {name}" if overwrite else
                             f"Calculating {name} for the first time")
            ret = func()
            pickle.dump(ret, open(path, 'wb'))
            return ret

        else:
            self.logger.info(f"Loading {name} from file")
            return pickle.load(open(path, 'rb'))


def main():
    imrex_attributions = ImrexAttributionsHandler(
        name="imrex_nocdr3dup",
        display_name="ImRex",
        model_path="ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/iteration_2/"
                   "2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv-epoch20.h5",
        image_path="data/tcr3d_images/",
        save_folder="data"
    )
    imrex_attributions.set_all(overwrite=True, split_channels=True)


if __name__ == "__main__":
    main()
