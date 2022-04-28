import os
import pickle
import sys
import numpy as np
import pandas as pd
import shap
import logging

sys.path.insert(0, os.path.abspath('.'))
from TITAN.scripts.flexible_model_eval import load_data, load_model
from src.util import concatted_inputs_to_input_pair_lists, duplicate_input_pair_lists, aa_remove_padding, rmse, \
    setup_logger, normalize_2d


class TITANAttributionsHandler:
    def __init__(self, name, display_name, model_path, tcrs_path, epitopes_path, data_path, save_folder):
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

        if not os.path.exists(f"{self.save_folder}/{self.name}"):
            os.makedirs(f"{self.save_folder}/{self.name}")

        self.logger = setup_logger(self.name)

    def get_sequences(self):
        tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv")
        sequences = {}
        for pdb_id, cdr3, ep in zip(tcr3df['PDB_ID'], tcr3df['cdr3'], tcr3df['antigen.epitope']):
            sequences[pdb_id] = (ep, cdr3)
        return sequences

    def get_aa_attributions(self, overwrite=False):
        def attributions():
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
        def aa_distances():
            print("Calculating AA distance must be done with ImrexAttributionsHandler!")
            return None

        self.aa_distances = self.__handle_getter(aa_distances, self.aa_distances, overwrite, "aa_distances.p",
                                                 'AA distances')
        return self.aa_distances

    def get_aa_errors(self, overwrite=False):
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
        self.get_aa_attributions(overwrite)
        self.get_aa_norm_attributions(overwrite)
        self.get_aa_distances(overwrite)
        self.get_aa_errors(overwrite)
        self.get_aa_errors_ps(overwrite)

    def get_shap_attributions(self, inputs):
        logging.getLogger('shap').setLevel(30)

        def shap_f(z):
            l, r = concatted_inputs_to_input_pair_lists(z)
            if l.shape[0] == 1:
                l, r = duplicate_input_pair_lists(l, r)
                return self.model(l, r)[0][0].numpy()
            return self.model(l, r)[0].numpy().squeeze()

        explainer = shap.KernelExplainer(shap_f, inputs)
        shap_values = explainer.shap_values(inputs, nsamples=1000)
        return shap_values

    def __handle_getter(self, func, self_ret, overwrite, path, name):
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
