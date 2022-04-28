import pickle
import shap
import os
import sys
import pandas as pd
import numpy as np
import saliency.core as saliency
import tensorflow as tf

sys.path.insert(0, os.path.abspath('.'))
from src.util import imrex_remove_padding, imgs_to_list_of_feature_lists, generate_path_inputs, setup_logger, \
    get_distance_matrices, normalize_2d, rmse, matrix_to_aa, list_feature_list_to_list_imgs, img_to_feature_list, \
    get_mean_feature_values, integral_approximation


class ImrexAttributionsHandler:
    def __init__(self, name, display_name, model_path, image_path, save_folder):
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

        self.aa_attributions = None
        self.aa_norm_attributions = None
        self.aa_distances = None
        self.aa_norm_distances = None
        self.aa_errors = None
        self.aa_errors_ps = None
        self.aa_random_error = None
        self.aa_random_error_ps = None

        if not os.path.exists(f"{self.save_folder}/{self.name}"):
            os.makedirs(f"{self.save_folder}/{self.name}")

        self.logger = setup_logger(self.name)

    def get_sequences(self):
        tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv")
        sequences = {}
        for pdb_id, cdr3, ep in zip(tcr3df['PDB_ID'], tcr3df['cdr3'], tcr3df['antigen.epitope']):
            sequences[pdb_id] = (ep, cdr3)
        return sequences

    def get_attributions(self, overwrite=False):
        def attributions():
            tcr3df = pd.read_csv(f"{self.save_folder}/tcr3d_imrex_output.csv")
            self.model, self.input_images = self.imrex_setup()

            shaps_mean = self.get_shap_attributions('mean')
            shaps_dist = self.get_shap_attributions('dist')

            attributions = {}
            for pdb_id, cdr3, ep, shap_mean, shap_dist in zip(tcr3df['PDB_ID'], tcr3df['cdr3'],
                                                              tcr3df['antigen.epitope'], shaps_mean, shaps_dist):

                manual_ig = self.get_ig_attribution(self.input_images[pdb_id])

                img_attributions = {'SHAP mean': imrex_remove_padding(shap_mean, len(cdr3), len(ep)),
                                    'SHAP BGdist': imrex_remove_padding(shap_dist, len(cdr3), len(ep)),
                                    'IG': imrex_remove_padding(manual_ig, len(cdr3), len(ep))}

                saliency_methods = ['Vanilla', 'SmoothGrad', 'VanillaIG', 'SmoothGradIG', 'GuidedIG', 'XRAI',
                                    'BlurIG', 'SmoothGradBlurIG']
                for method in saliency_methods:
                    saliency_attribution = self.get_saliency_attribution(self.input_images[pdb_id], method)
                    img_attributions[method] = imrex_remove_padding(saliency_attribution, len(cdr3), len(ep))

                attributions[pdb_id] = img_attributions
                self.logger.info(f'{pdb_id} done')
            return attributions

        self.attributions = self.__handle_getter(attributions, self.attributions, overwrite,
                                                 f"{self.name}/attributions.p", 'attributions')
        return self.attributions

    def get_distances(self, overwrite=False):
        self.distances = self.__handle_getter(get_distance_matrices, self.distances, overwrite, "distance_matrices.p",
                                              'distances')

        return self.distances

    def get_norm_distances(self, overwrite=False):
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
        def errors():
            attributions = self.get_attributions()
            distances = self.get_distances()
            error_dict = {}
            for pdb_id, methods in attributions.items():
                dist = distances[pdb_id]
                error_pdb = {}
                for method, attribution in methods.items():
                    error_pdb[method] = rmse(dist, attribution)
                error_dict[pdb_id] = error_pdb
                self.logger.info(f'{pdb_id} done')
            return error_dict

        self.errors = self.__handle_getter(errors, self.errors, overwrite, f"{self.name}/errors.p", 'errors')
        return self.errors

    def get_random_error(self, overwrite=False):
        def random_error():
            distances = self.get_distances()
            random_errors = []
            for pdb_id, dist_m in distances.items():
                for i in range(1000):
                    random_m = np.random.rand(*dist_m.shape)
                    random_errors.append(rmse(dist_m, random_m))
                self.logger.info(f'{pdb_id} done')
            return np.mean(random_errors), np.std(random_errors)

        self.random_error = self.__handle_getter(random_error, self.random_error, overwrite, "imrex_random_error.p",
                                                 'random_error')
        return self.random_error

    def get_aa_attributions(self, overwrite=False):
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
        def aa_errors():
            aa_attributions = self.get_aa_attributions()
            aa_distances = self.get_aa_distances()
            error_dict = {}
            for pdb_id, methods in aa_attributions.items():
                aa_dist = aa_distances[pdb_id]
                error_pdb = {}
                for method, attribution in methods.items():
                    error_pdb[method] = rmse(aa_dist, attribution)
                error_dict[pdb_id] = error_pdb
            return error_dict

        self.aa_errors = self.__handle_getter(aa_errors, self.aa_errors, overwrite, f"{self.name}/aa_errors.p",
                                              'AA errors')
        return self.aa_errors

    def get_aa_errors_ps(self, overwrite=False):
        def aa_errors_ps():
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

        self.aa_errors_ps = self.__handle_getter(aa_errors_ps, self.aa_errors_ps, overwrite,
                                                 f"{self.name}/aa_errors_ps.p", 'AA errors per sequence')
        return self.aa_errors_ps

    def get_aa_random_error(self, overwrite=False):
        def aa_random_error():
            errors = []
            aa_distances = self.get_aa_distances()
            for pdb_id, aa_dist in aa_distances.items():
                for i in range(1000):
                    random_a = np.random.rand(len(aa_dist))
                    errors.append(rmse(aa_dist, random_a))
            return np.mean(errors), np.std(errors)

        self.aa_random_error = self.__handle_getter(aa_random_error, self.aa_random_error, overwrite,
                                                    f"aa_random_error.p", 'AA random error')
        return self.aa_random_error

    def get_aa_random_error_ps(self, overwrite=False):
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

    def set_all(self, overwrite=False):
        self.get_attributions(overwrite)
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

    def imrex_setup(self):
        model = tf.keras.models.load_model(self.model_path)
        input_imgs = {}
        for f in sorted(os.listdir(self.image_path)):
            input_imgs[f[:-4]] = tf.cast(tf.convert_to_tensor(pickle.load(open(self.image_path + f, 'rb'))), tf.float32)
        return model, input_imgs

    def make_prediction(self, img):
        image_batch = tf.expand_dims(img, 0)
        prob = self.model(image_batch)
        prediction = tf.math.round(prob)
        return prediction

    def get_shap_attributions(self, background):
        def shap_f(z):
            return self.model.predict(list_feature_list_to_list_imgs(z))

        input_imgs = np.array(list(self.input_images.values()))
        if background == 'mean':
            explainer = shap.KernelExplainer(shap_f,
                                             img_to_feature_list(get_mean_feature_values(input_imgs))[None, ...])
        elif background == 'dist':
            explainer = shap.KernelExplainer(shap_f, imgs_to_list_of_feature_lists(input_imgs))
        else:
            return
        shap_values = explainer.shap_values(imgs_to_list_of_feature_lists(input_imgs), nsamples=1000)
        attributions = list_feature_list_to_list_imgs(shap_values)
        return tf.reduce_sum(tf.math.abs(attributions), axis=-1).numpy()

    def get_ig_attribution(self, input_img):
        baseline = tf.zeros(shape=(20, 11, 4))  # Baseline for CMYK format. The picture is white in this 4-channel case

        # ### Linear interpolation path between baseline and input images at alpha intervals
        m_steps = 50  # number of interpolation steps chosen
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)  # levels of transparency for each step

        # generate the interpolated images
        path_inputs = generate_path_inputs(baseline=baseline, input=input_img, alphas=alphas)

        # Let's calculate the gradients for each image along the interpolation path with respect to the correct output
        with tf.GradientTape() as tape:
            tape.watch(path_inputs)  # define on what to calculate the gradients
            outputs = self.model(path_inputs)

        path_gradients = tape.gradient(outputs, path_inputs)

        ig = integral_approximation(gradients=path_gradients, method='riemann_trapezoidal')

        attributions = (input_img - baseline) * ig

        # Sum of the attributions across color channels for visualization.
        # The attribution mask shape is a grayscale image with height and width
        # equal to the original image.
        attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
        return attribution_mask.numpy()

    def get_saliency_attribution(self, im, method):
        # definition of the baseline
        baseline = tf.zeros(shape=(20, 11, 4))  # Baseline for CMYK format. The picture is white in this 4-channel case

        # definition of the function to compute the gradients, it will be passed to the various saliency methods
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
            mask = saliency_object.GetMask(im, call_model_function, call_model_args, x_steps=50, x_baseline=baseline,
                                           max_dist=1.0, fraction=0.5)
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
            return saliency.VisualizeImageGrayscale(mask)
        else:
            return mask

    def __handle_getter(self, func, self_ret, overwrite, path, name):
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
    imrex_attributions.set_all()


if __name__ == "__main__":
    main()
