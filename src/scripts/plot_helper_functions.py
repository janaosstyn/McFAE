from typing import Dict


def get_model_attributions_dictionary(model_attributions, model_names) -> [Dict, bool]:
    """
    Given a dictionary of model attributions where channel + pdb_id are concatenated, obtain a dictionary that splits
    the IDs in channel and pdb_id.

    Parameters
    ----------
    model_attributions: a dictionary with model attributions
    model_names: model names (for dictionary keys)

    Returns
    -------
    The modified dictionary and a boolean indicating whether the original dictionary indeed implicitly contained
    channels (if not, the dictionary is returned unmodified).
    """
    if '_' in list(model_attributions[list(model_attributions.keys())[0]].keys())[0]:
        return {
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
            for model_name in model_names
        }, True
    else:
        return model_attributions, False


def get_per_sequence_correlation(model_handler, method, is_combined, shows_all):
    """
    Get correlation values for each sequence.

    Parameters
    ----------
    model_handler: a model handler from which correlation information is fetched
    method: a feature attribution extraction method
    is_combined: indicates whether a 'combined' channel is included
    shows_all: indicates whether the x-labels should contain new line

    Returns
    -------
    A boolean that indicates whether the data is channeled, a list of x-labels for the plot and a list of correlations
    """
    channels = False
    model_correlation_per_sequence = []
    names_per_sequence = []

    correlation = model_handler.get_aa_correlation_ps()
    display_names = []
    per_pdb_correlations = []  # correlation_ep, correlation_cdr3, pdb_id
    for pdb_id, methods in correlation.items():
        per_pdb_correlations.append((
            -methods[method][0],
            -methods[method][1],
            pdb_id
        ))

    correlation_dict_per_sequence = dict()
    if '_' in per_pdb_correlations[-1][2]:
        channels = True
        for channel in range(4):
            correlation_dict_per_sequence[f'Ch {channel}'] = [
                correlation_tuple for correlation_tuple in per_pdb_correlations
                if 'combined' not in correlation_tuple[2] and int(correlation_tuple[2].split('_')[1]) == channel
            ]
            display_names.append(f'Ch {channel}')
        if is_combined:
            correlation_dict_per_sequence[f'Combi'] = [
                correlation_tuple for correlation_tuple in per_pdb_correlations
            ]
            display_names.append(f'Combi')
    else:
        correlation_dict_per_sequence[''] = per_pdb_correlations
        display_names.append(model_handler.display_name)

    for name in range(len(display_names)):
        model_correlation_per_sequence.append([tup[0] for tup in correlation_dict_per_sequence[name]])
        model_correlation_per_sequence.append([tup[1] for tup in correlation_dict_per_sequence[name]])

        for sub_name in ['epitope', 'CDR3']:
            names_per_sequence.append(
                display_names[name] +
                (f' {sub_name}' if not shows_all else ('\n' + f'{sub_name}'))
            )

    return channels, names_per_sequence, model_correlation_per_sequence


def get_correlation(model_handler, method, is_combined, has_channels):
    """
    Get correlations.

    Parameters
    ----------
    model_handler: a model handler from which correlation information is fetched
    method: a feature attribution extraction method
    is_combined: indicates whether a 'combined' channel is included
    has_channels: data is channeled

    Returns
    -------
    Correlation values
    """
    correlation = model_handler.get_aa_correlation()
    correlation_dict = dict()
    if has_channels:
        for channel in range(4):
            correlation_dict[f'Ch {channel}'] = [
                -methods[method]
                for pdb_id, methods in correlation.items()
                if "combined" not in pdb_id and int(pdb_id.split('_')[1]) == channel
            ]
        if is_combined:
            correlation_dict['Combined'] = [
                -methods[method]
                for pdb_id, methods in correlation.items()
                if "combined" in pdb_id
            ]
    else:
        correlation_dict[''] = [
            -methods[method]
            for methods in correlation.values()
        ]

    return list(correlation_dict.values())


def set_att_plot_specs(axs, ep, att, method, x_label=True, title=True):
    """
    Small helper function to create and partially fulfil an attribute heatmap image.

    Parameters
    ----------
    axs: matplotlib axes object
    ep: epitope AA sequence
    att: attribute value
    method: feature extraction method (for title)
    x_label: set x-axis label
    title: set title

    Returns
    -------
    Modified axs
    """
    axs.imshow(att, cmap='Greys', vmin=0, vmax=1)
    axs.set_xticks(list(range(len(ep))))
    axs.set_xticklabels(ep)
    if title:
        axs.set_title("SHAP" if method == 'SHAP BGdist' else "IG" if method == "VanillaIG" else method)
    if x_label:
        axs.set_xlabel('epitope')
    return axs


def set_dist_plot_specs(axs, ep, dist, x_label=True, title=True):
    """
    Small helper function to create and partially fulfil a distance plot.

    Parameters
    ----------
    axs: matplotlib axes object
    ep: epitope AA sequence
    dist: distances to plot as heatmap
    x_label: set x-axis label
    title: set title

    Returns
    -------

    """
    axs.imshow(dist, cmap='Greys', vmin=0, vmax=1)
    axs.set_xticks(list(range(len(ep))))
    axs.set_xticklabels(ep)
    if title:
        axs.set_title('Pairwise\nresidue proximity')
    if x_label:
        axs.set_xlabel('epitope')
    return axs
