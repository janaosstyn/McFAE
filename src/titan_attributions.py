import logging
import os
import pickle

import numpy as np
import pandas as pd
import shap

from TITAN.scripts.flexible_model_eval import load_data, load_model
from src.util import concatted_inputs_to_input_pair_lists, duplicate_input_pair_lists, aa_remove_padding, rmse, \
    setup_logger, normalize_2d


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

            # Calculate SHAP attributions
            shap_attributions = self.get_shap_attributions(all_data_pairs)

            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv")
            all_attributions = {}
            for shap_attribution, input_data, pdb_id in zip(shap_attributions, all_data_pairs, tcr3df['PDB_ID']):
                pdb_attributions = {'SHAP BGdist': aa_remove_padding(np.abs(shap_attribution), input_data)}
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
