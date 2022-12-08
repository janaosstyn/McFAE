import logging
import os
import pickle

import numpy as np
import pandas as pd
import saliency.core as saliency
import shap
import torch
from scipy.stats import pearsonr, spearmanr

from TITAN.scripts.flexible_model_eval import load_data, load_model
from src.util import concatted_inputs_to_input_pair_lists, duplicate_input_pair_lists, aa_remove_padding, rmse, \
    setup_logger, normalize_2d, correlation_nan, p_value_stats


class TITANAttributionsHandler:
    """
    Class that manages feature attribution extraction from any pretrained TITAN model. This class calculates and
    saves all required (intermediary) results, e.g. featura attributions and RMSE's.
    """

    def __init__(self, name, display_name, model_path, tcrs_path, epitopes_path, data_path, save_folder):
        """
        Parameters
        ----------
        name            Internal name, results will be saved in the folder [save_folder]/[name]
        display_name    Name displayed by e.g. plot.py
        model_path      Path to the TITAN model
        tcrs_path       Path to the TITAN TCRs file
        epitopes_path   Path to the TITAN epitopes file
        data_path       Path to the input data file
        save_folder     Path to the save folder, results will be saved in the folder [save_folder]/[name]
        """
        self.name = name
        self.display_name = display_name
        self.model_path = model_path
        self.tcrs_path = tcrs_path
        self.epitopes_path = epitopes_path
        self.data_path = data_path
        self.save_folder = save_folder

        self.model = None
        self.attributions = None
        self.norm_attributions = None
        self.errors = None
        self.errors_ps = None
        self.aa_distances = None
        self.correlation = None
        self.aa_correlation_ps = None
        self.random_correlation = None
        self.spearmanc = None
        self.aa_spearmanc_ps = None
        self.random_spearmanc = None

        # Create folder to save results if it does not exists yet
        if not os.path.exists(f"{self.save_folder}/{self.name}"):
            os.makedirs(f"{self.save_folder}/{self.name}")

        self.logger = setup_logger(self.name)

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

    def get_aa_attributions(self, overwrite=False):
        """
        Calculate and save all feature attributions, loaded from file if already saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, attributions will be re-calculated and overwrite the old saved result

        Returns
        -------
        A dict with attributions for each PDB complex
        """

        def attributions():
            # Get data and setup TITAN model
            test_loader, params, smiles_language, protein_language = load_data(self.data_path, self.tcrs_path,
                                                                               self.epitopes_path, self.model_path)
            self.model = load_model('bimodal_mca', self.model_path, params, smiles_language, protein_language)
            self.model.to(device='cpu')
            self.model.zero_grad()
            self.model.eval()
            self.model.requires_grad_(False)

            all_data_pairs = np.array(
                [np.concatenate((l.cpu().numpy(), r.cpu().numpy())) for ls, rs, ys in test_loader for l, r, y in
                 zip(ls, rs, ys)])

            saliency_attributions = self.get_saliency_attributions(test_loader)

            # Calculate SHAP attributions
            shap_attributions = self.get_shap_attributions(all_data_pairs)

            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv")
            all_attributions = {}
            for shap_attribution, saliency_attribution, input_data, pdb_id in zip(
                    shap_attributions, saliency_attributions, all_data_pairs, tcr3df['PDB_ID']):
                pdb_attributions = {'SHAP BGdist': aa_remove_padding(np.abs(shap_attribution), input_data)}
                for method, saliency_att in saliency_attribution.items():
                    pdb_attributions[method] = aa_remove_padding(saliency_att, input_data)
                all_attributions[pdb_id] = pdb_attributions
            return all_attributions

        self.attributions = self.__handle_getter(attributions, self.attributions, overwrite,
                                                 f"{self.name}/attributions.p", 'attributions')
        return self.attributions

    def get_aa_norm_attributions(self, overwrite=False):
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
            aa_attributions = self.get_aa_attributions()
            aa_norm_attributions = {}
            for pdb_id, methods in aa_attributions.items():
                aa_norm_attributions[pdb_id] = {}
                for method, attribution in methods.items():
                    aa_norm_attributions[pdb_id][method] = normalize_2d(attribution)
            return aa_norm_attributions

        self.norm_attributions = self.__handle_getter(norm_attributions, self.norm_attributions, overwrite,
                                                      f"{self.name}/norm_attributions.p", 'normalized attributions')
        return self.norm_attributions

    def get_aa_distances(self, overwrite=False):
        """
        Load the AA distances from file, this function can only be called if those are already calculated by the
        ImrexAttributionsHandler

        Parameters
        ----------
        overwrite   Not used, this function only loads AA distance from file

        Returns
        -------
        A dict with the AA distances for each PDB complex
        """

        def aa_distances():
            print("Calculating AA distance must be done with ImrexAttributionsHandler!")
            return None

        self.aa_distances = self.__handle_getter(aa_distances, self.aa_distances, overwrite, "aa_distances.p",
                                                 'AA distances')
        return self.aa_distances

    def get_aa_errors(self, overwrite=False):
        """
        Calculate and save the RMSE's between the AA distance and feature attributions, loaded from file if already
        saved previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, RMSE's will be re-calculated and overwrite the old saved result, the original AA
                    distances and attributions will not be re-calculated.

        Returns
        -------
        A dict with the RMSE for each PDB complex
        """

        def errors():
            attributions = self.get_aa_attributions()
            aa_distances = self.get_aa_distances()
            error_dict = {}
            for pdb_id, methods in attributions.items():
                aa_dist = aa_distances[pdb_id]
                error_pdb = {}
                for method, attribution in methods.items():
                    error_pdb[method] = rmse(aa_dist, attribution)
                error_dict[pdb_id] = error_pdb
            return error_dict

        self.errors = self.__handle_getter(errors, self.errors, overwrite, f"{self.name}/errors.p", 'errors')
        return self.errors

    def get_aa_errors_ps(self, overwrite=False):
        """
        Calculate and save the RMSE's per sequence (epitope and CDR3) separately, loaded from file if already saved
        previously (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, RMSE's per sequence will be re-calculated and overwrite the old saved result, the
                    original AA distances and attributions will not be re-calculated.

        Returns
        -------
        A dict with the RMSE per sequence for each PDB complex
        """

        def errors_ps():
            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv", index_col='PDB_ID')
            aa_attributions = self.get_aa_attributions()
            aa_distances = self.get_aa_distances()
            error_ps_dict = {}
            for pdb_id, methods in aa_attributions.items():
                aa_dist = aa_distances[pdb_id]
                seq_split = len(tcr3df.loc[pdb_id]['antigen.epitope'])
                error_ps_dict[pdb_id] = {}
                for method, attribution in methods.items():
                    ep_aa_dist = aa_dist[:seq_split]
                    cdr3_aa_dist = aa_dist[seq_split:]
                    ep_attribution = attribution[:seq_split]
                    cdr3_attribution = attribution[seq_split:]
                    error_ps_dict[pdb_id][method] = (rmse(ep_aa_dist, ep_attribution),
                                                     rmse(cdr3_aa_dist, cdr3_attribution))
            return error_ps_dict

        self.errors_ps = self.__handle_getter(errors_ps, self.errors_ps, overwrite,
                                              f"{self.name}/errors_ps.p", 'errors per sequence')
        return self.errors_ps

    def get_aa_correlation(self, correlation_method='pearson', overwrite=False):
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

        def correlation():
            attributions = self.get_aa_attributions()
            distances = self.get_aa_distances()
            corr_dict = {}
            method_p = {m: [] for m in list(attributions.values())[0].keys()}
            for pdb_id, methods in attributions.items():
                dist = distances[pdb_id]
                corr_pdb = {}
                for method, attribution in methods.items():
                    if correlation_method == 'pearson':
                        corr_pdb[method], p = correlation_nan(pearsonr, dist, attribution, with_p=True)
                        method_p[method].append(p)
                    elif correlation_method == 'spearman':
                        corr_pdb[method], p = correlation_nan(spearmanr, dist, attribution, with_p=True)
                        method_p[method].append(p)
                corr_dict[pdb_id] = corr_pdb
            p_value_stats(self.name, method_p)
            return corr_dict

        if correlation_method == 'pearson':
            self.correlation = self.__handle_getter(correlation, self.correlation, overwrite,
                                                    f"{self.name}/correlation.p", f'correlation')
            return self.correlation
        elif correlation_method == 'spearman':
            self.spearmanc = self.__handle_getter(correlation, self.spearmanc, overwrite, f"{self.name}/spearmanc.p",
                                                  f'spearman correlation')
            return self.spearmanc

    def get_aa_correlation_ps(self, correlation_method='pearson', overwrite=False):
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
            correlation_dict = {}
            for pdb_id, methods in aa_attributions.items():
                aa_dist = aa_distances[pdb_id]
                # Get starting index of epitope
                seq_split = len(tcr3df.loc[pdb_id]['antigen.epitope'])
                correlation_dict[pdb_id] = {}
                for method, attribution in methods.items():
                    # Split distance and attributions arrays in epitope and CDR3
                    ep_aa_dist = aa_dist[:seq_split]
                    cdr3_aa_dist = aa_dist[seq_split:]
                    ep_attribution = attribution[:seq_split]
                    cdr3_attribution = attribution[seq_split:]
                    # Calculate correlation separately for epitope and CDR3
                    if correlation_method == 'pearson':
                        correlation_dict[pdb_id][method] = (
                            correlation_nan(pearsonr, ep_aa_dist.flatten(), ep_attribution.flatten()),
                            correlation_nan(pearsonr, cdr3_aa_dist.flatten(), cdr3_attribution.flatten()))
                    elif correlation_method == 'spearman':
                        correlation_dict[pdb_id][method] = (
                            correlation_nan(spearmanr, ep_aa_dist.flatten(), ep_attribution.flatten()),
                            correlation_nan(spearmanr, cdr3_aa_dist.flatten(), cdr3_attribution.flatten()))
            return correlation_dict

        if correlation_method == 'pearson':
            self.aa_correlation_ps = self.__handle_getter(aa_correlation_ps, self.aa_correlation_ps, overwrite,
                                                          f"{self.name}/aa_correlation_ps.p",
                                                          'AA correlation per sequence')
            return self.aa_correlation_ps
        elif correlation_method == 'spearman':
            self.aa_spearmanc_ps = self.__handle_getter(aa_correlation_ps, self.aa_spearmanc_ps, overwrite,
                                                        f"{self.name}/aa_spearmanc_ps.p",
                                                        'AA spearman correlation per sequence')
            return self.aa_spearmanc_ps

    def get_aa_random_correlation(self, correlation_method='pearson', overwrite=False):
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

        def random_correlation():
            raise NotImplementedError('Random correlation should be calculated from ImRex')

        if correlation_method == 'pearson':
            self.random_correlation = self.__handle_getter(random_correlation, self.random_correlation, overwrite,
                                                           "aa_random_correlation.p", 'random correlation')
            return self.random_correlation
        elif correlation_method == 'spearman':
            self.random_spearmanc = self.__handle_getter(random_correlation, self.random_spearmanc, overwrite,
                                                         "aa_random_spearmanc.p", 'random spearman correlation')
            return self.random_spearmanc

    def set_all(self, overwrite=False):
        """
        Calculate and save all results for this model, results are loaded from file if already saved previously
        (and overwrite=False)

        Parameters
        ----------
        overwrite   If True, all results will be re-calculated and overwrite the old saved results.
        """
        self.get_aa_attributions(overwrite)
        self.get_aa_norm_attributions(overwrite)
        self.get_aa_distances(overwrite)
        self.get_aa_errors(overwrite)
        self.get_aa_errors_ps(overwrite)
        for cm in ['pearson', 'spearman']:
            self.get_aa_correlation(cm, overwrite)
            self.get_aa_correlation_ps(cm, overwrite)
            self.get_aa_random_correlation(cm, overwrite)

    def get_shap_attributions(self, inputs):
        """
        Calculate the SHAP feature attributions for the given inputs

        Parameters
        ----------
        inputs  All input samples

        Returns
        -------
        A feature attribution array for each input sample
        """
        logging.getLogger('shap').setLevel(30)

        def shap_f(z):
            l, r = concatted_inputs_to_input_pair_lists(z)

            # TITAN cannot handle a single input sample. If this is the case, we will duplicate the sample and input
            # this but only use the first output at the end
            if l.shape[0] == 1:
                l, r = duplicate_input_pair_lists(l, r)
                return self.model(l, r)[0][0].numpy()
            return self.model(l, r)[0].numpy().squeeze()

        explainer = shap.KernelExplainer(shap_f, inputs)
        shap_values = explainer.shap_values(inputs, nsamples=1000)
        return shap_values

    def get_saliency_attributions(self, test_loader):
        """
        Calculate the gradient (saliency) feature attributions with a set of different methods for the inputs in the
        given dataloader

        Parameters
        ----------
        test_loader  A dataloader containing the input samples

        Returns
        -------
        A feature attribution array for each input sample
        """
        attributions = []
        i = 0
        logging.getLogger('saliency.core.xrai').setLevel(logging.WARNING)
        for ls, rs, ys in test_loader:
            for l, r in zip(ls.cpu(), rs.cpu()):
                embed_l = self.model.ligand_embedding(l.to(torch.int64))
                embed_r = self.model.receptor_embedding(r.to(torch.int64))
                method_attributions = {}
                for method in ['Vanilla', 'SmoothGrad', 'VanillaIG', 'SmoothGradIG', 'GuidedIG', 'XRAI', 'BlurIG',
                               'SmoothGradBlurIG']:
                    method_attributions[method] = self._get_saliency_attribution(embed_l, embed_r, method)
                attributions.append(method_attributions)
                self.logger.info(f'Sample {i} done')
                i += 1
        return attributions

    def _get_saliency_attribution(self, embed_l, embed_r, method):
        """
        Calculate the gradient (saliency) feature attributions for the given input

        Parameters
        ----------
        embed_l  The embedded ligand input
        embed_r  The embedded receptor input
        method   The Saliency method to use

        Returns
        -------
        Feature attribution array for the input sample extracted with the given method
        """

        def call_model_function(batch_concat_input, call_model_args=None, expected_keys=None):
            # SET EMBEDDING TYPE TO PREDEFINED, the regular forward function won't use them anymore!
            ligand_embedding_type_backup = self.model.ligand_embedding_type
            receptor_embedding_type_backup = self.model.receptor_embedding_type
            self.model.ligand_embedding_type = 'predefined'
            self.model.receptor_embedding_type = 'predefined'

            if len(batch_concat_input.shape) > 3:
                blur_ig = True
                batch_l = torch.tensor(batch_concat_input[:, :, :25]).squeeze(1).requires_grad_(True)
                batch_r = torch.tensor(batch_concat_input[:, :, 25:]).squeeze(1).requires_grad_(True)
            else:
                blur_ig = False
                batch_l = torch.tensor(batch_concat_input[:, :25]).requires_grad_(True)
                batch_r = torch.tensor(batch_concat_input[:, 25:]).requires_grad_(True)

            if batch_l.size(0) == 1:
                out = self.model(batch_l.repeat(2, 1, 1), batch_r.repeat(2, 1, 1))[0][0]
            else:
                out = self.model(batch_l, batch_r)[0].squeeze()

            # REVERT EMBEDDING TYPE CHANGE
            self.model.ligand_embedding_type = ligand_embedding_type_backup
            self.model.receptor_embedding_type = receptor_embedding_type_backup

            gradient_ligand, gradient_receptor = torch.autograd.grad(out, [batch_l, batch_r],
                                                                     grad_outputs=torch.ones_like(out))
            gradients = torch.concat((gradient_ligand, gradient_receptor), 1)

            if blur_ig:
                gradients = gradients.unsqueeze(1)

            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients.numpy()}

        # removed baseline, this defaults to zero anyway
        if method == "Vanilla":
            saliency_object = saliency.GradientSaliency()
            mask = saliency_object.GetMask(torch.concat([embed_l, embed_r]).numpy(), call_model_function)
        elif method == 'SmoothGrad':
            saliency_object = saliency.GradientSaliency()
            mask = saliency_object.GetSmoothedMask(torch.concat([embed_l, embed_r]).numpy(), call_model_function)
        elif method == 'VanillaIG':
            saliency_object = saliency.IntegratedGradients()
            mask = saliency_object.GetMask(torch.concat([embed_l, embed_r]).numpy(), call_model_function,
                                           x_steps=50, batch_size=128)
        elif method == 'SmoothGradIG':
            saliency_object = saliency.IntegratedGradients()
            mask = saliency_object.GetSmoothedMask(torch.concat([embed_l, embed_r]).numpy(), call_model_function,
                                                   x_steps=50, batch_size=128)
        elif method == 'GuidedIG':
            saliency_object = saliency.GuidedIG()
            mask = saliency_object.GetMask(torch.concat([embed_l, embed_r]).numpy(), call_model_function, x_steps=50,
                                           max_dist=1.0, fraction=0.5)
        elif method == 'XRAI':
            saliency_object = saliency.XRAI()
            mask = saliency_object.GetMask(torch.concat([embed_l, embed_r]).numpy(), call_model_function,
                                           batch_size=128)
        elif method == 'BlurIG':
            saliency_object = saliency.BlurIG()  # Blur Integrated Gradients
            mask = saliency_object.GetMask(torch.concat([embed_l, embed_r]).unsqueeze(0).numpy(), call_model_function,
                                           batch_size=128)
            mask = mask.squeeze(0)
        elif method == 'SmoothGradBlurIG':
            saliency_object = saliency.BlurIG()  # Blur Integrated Gradients
            mask = saliency_object.GetSmoothedMask(torch.concat([embed_l, embed_r]).unsqueeze(0).numpy(),
                                                   call_model_function, batch_size=128)
            mask = mask.squeeze(0)

        else:
            self.logger.error(f'Attribution method {method} not recognized')
            return

        mask = np.abs(mask).sum(axis=1)
        if method != 'XRAI':
            vmax = np.percentile(mask, 99)
            vmin = np.min(mask)
            return np.clip((mask - vmin) / (vmax - vmin), 0, 1)
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
            print(f"Overwrite=True, re-calculating {name}" if overwrite else
                  f"Calculating {name} for the first time")
            ret = func()
            if ret is not None:
                pickle.dump(ret, open(path, 'wb'))
            return ret

        else:
            print(f"Loading {name} from file")
            return pickle.load(open(path, 'rb'))


def main():
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

    titan_on_imrex_data_handler.set_all()
    titan_strictsplit_handler.set_all()


if __name__ == "__main__":
    main()
