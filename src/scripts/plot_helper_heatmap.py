import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


class Heatmap:
    def __init__(self, x):
        """
        Constructor

        Parameters
        ----------
        x: What is on the x-axis? ('AA' or 'ep cdr3')
        """
        self.heatmap: Dict[str, np.ndarray] = dict()
        self.min_val, self.max_val = 0, 1

        self.x = x

    def initialize_rows(self, row_names: List[str]) -> None:
        """
        Fill the heatmap with zero filled ndarray rows with the given names.
        Parameters
        ----------
        row_names: List of row names
        """

    def compose_heatmap(self) -> None:
        """
        Combine the different sub heatmaps into a single heatmap, where both sub heatmaps are separated by a -1
        column.
        """

    def create_plot(self, color_bar_label: str, filename: str) -> None:
        """
        Create the heatmap plot.

        Parameters
        ----------
        color_bar_label: The label for the color bar
        filename: The filename for the output figure
        """

    def calculate_diff(
            self,
            subtract: [Dict[str, np.ndarray], np.ndarray],
            normalize: bool = False,
            max_ep_length: int = None
    ):
        """
        Helper function that subtracts two heatmaps.

        Parameters
        ----------
        subtract: Either a heatmap (in the form of a dictionary, like self.heatmap) or a np.ndarray, representing the
            mean per key
        normalize: If true, the result is normalized
        max_ep_length: The separator position where -1 will be put, only for x='ep cdr3'
        """
        self.min_val, self.max_val = math.inf, -math.inf

        for key in self.heatmap:
            self.heatmap[key] = self.heatmap[key] - subtract[key]
            if self.x == 'ep cdr3':
                # ensure that this position is not min or max
                self.heatmap[key][max_ep_length] = np.mean(self.heatmap[key])
            self.min_val = min(self.min_val, np.nanmin(self.heatmap[key]))
            self.max_val = max(self.max_val, np.nanmax(self.heatmap[key]))
            if self.x == 'ep cdr3':
                # now set position to -1
                self.heatmap[key][max_ep_length] = -1

        if normalize:
            for key in self.heatmap:
                self.heatmap[key] = (self.heatmap[key] - self.min_val) / (self.max_val - self.min_val)
                if self.x == 'ep cdr3':
                    # position might not be -1 anymore
                    self.heatmap[key][max_ep_length] = -1

            self.min_val, self.max_val = 0, 1

    def write_plot(
            self,
            heatmap: np.ndarray,
            x_ticks: List[int],
            x_labels: List[str],
            color_bar_label: str,
            filename: str
    ) -> None:
        """
        Create a 1D heatmap plot, given a series of parameters

        Parameters
        ----------
        heatmap: The heatmap to plot
        x_ticks: The x-ticks
        x_labels: The labels for the x-axis
        color_bar_label: The label for the color bar
        filename: The filename for the output figure
        """
        y_ticks = list(range(len(self.heatmap)))
        y_labels = self.heatmap.keys()

        if 'correction' in filename:
            self.min_val = min(self.min_val, -self.max_val)
            self.max_val = max(self.max_val, -self.min_val)

        plt.gcf().set_size_inches(8, 3 if heatmap.shape[0] < 5 else 8)
        plt.colorbar(
            plt.imshow(X=heatmap, cmap='coolwarm' if 'correction' in filename else 'Greys'),
            orientation='horizontal',
            pad=0.05,
            label=color_bar_label
        )
        plt.xticks(ticks=x_ticks, labels=x_labels, fontsize='small')
        plt.yticks(ticks=y_ticks, labels=y_labels)
        plt.clim(vmin=0, vmax=1)
        plt.clim(vmin=self.min_val, vmax=self.max_val)
        plt.savefig(fname=filename, dpi=300, bbox_inches='tight')
        plt.close()


class AAHeatmap1D(Heatmap):
    def __init__(self):
        Heatmap.__init__(self, x='AA')

        self.aa_heatmap: Dict[str, Dict[str, List[float]]] = dict()

        self.aa: List[str] = list()
        self.aa_index: Dict[str, int] = dict()

    def copy_initialization(self) -> 'AAHeatmap1D':
        """
        Copy the data members to avoid duplicate initialization.

        Returns
        -------
        A new PositionalHeatmap2 with the same data member values
        """
        copied = AAHeatmap1D()
        copied.aa = self.aa
        copied.aa_index = self.aa_index

        for row_name in self.aa_heatmap.keys():
            copied.aa_heatmap[row_name] = {aa: [] for aa in self.aa}

        return copied

    def extract_aa(self, sequences: Dict[str, Tuple[str, str]], pdb_id_subset: List[str] = None) -> None:
        """
        Extract the different AA occurring across the sequences.

        Parameters
        ----------
        sequences: Dictionary that maps pdb IDs to its epitope and cdr3 sequences.
        pdb_id_subset: If not None, this list contains a subset of all pdb IDs. The pdb IDs that are not in this list
            are ignored.
        """
        flattened_sequences = [
            sequence for pdb_id, sequence_pair in sequences.items() for sequence in sequence_pair
            if pdb_id_subset is None or pdb_id in pdb_id_subset
        ]
        str_sequences = ''.join(flattened_sequences)

        self.aa = sorted(list(set([letter for letter in str_sequences])))
        self.aa_index = {aa: i for i, aa in enumerate(self.aa)}

    def initialize_rows(self, row_names: List[str]) -> None:
        """
        Fill the heatmap with the given names with a dictionary with for each AA an empty list
        Parameters
        ----------
        row_names: List of row names
        """
        for row_name in row_names:
            self.aa_heatmap[row_name] = {aa: [] for aa in self.aa}

    def add_aa_attribution(self, row_name: str, aa: str, attribution: float) -> None:
        """
        Add an AA attribution to the given row.

        Parameters
        ----------
        row_name: Name of the row in the heatmap
        aa: Amino acid
        attribution: Attribution value for amino acid
        """
        self.aa_heatmap[row_name][aa].append(attribution)

    def compose_heatmap(self, scrambled_heatmap=None) -> None:
        """
        Combine different AA into a single heatmap.

        Parameters
        ----------
        scrambled_heatmap: If not None, the difference between both is calculated.
        """
        for key in self.aa_heatmap:
            self.heatmap[key] = np.array([
                np.nanmean(self.aa_heatmap[key][aa])
                for aa in self.aa_heatmap[key].keys()
            ])

        if scrambled_heatmap is not None:
            scrambled_heatmap.compose_heatmap_correction()
            self.calculate_diff(subtract=scrambled_heatmap.heatmap, normalize=True)

    def compose_heatmap_correction(self) -> None:
        """
        Combine a "correction" heatmap.
        This correction is formed by the values minus the mean of the values
        """
        self.compose_heatmap()

        correction = {
            key: np.nanmean(self.heatmap[key])
            for key in self.heatmap.keys()
        }
        self.calculate_diff(subtract=correction)

    def create_plot(self, color_bar_label: str, filename: str) -> None:
        """
        Create the heatmap plot.

        Parameters
        ----------
        color_bar_label: The label for the color bar
        filename: The filename for the output figure
        """
        heatmap = np.array(list(self.heatmap.values()))

        x_ticks = list(range(len(self.aa)))
        x_labels = self.aa

        self.write_plot(
            heatmap=heatmap,
            x_ticks=x_ticks,
            x_labels=x_labels,
            color_bar_label=color_bar_label,
            filename=filename
        )


class EpCDR3OnXAxisHeatmap(Heatmap):
    def __init__(self):
        Heatmap.__init__(self, x='ep cdr3')

        # precise dict construction depends on subclass
        self.ep_heatmap: Dict = dict()
        self.cdr3_heatmap: Dict = dict()

        self.max_ep_length: int = 0
        self.max_cdr3_length: int = 0

    def general_copy_initialization(self, copied, ep_fill, cdr3_fill):
        """
        Copy the data members to avoid duplicate initialization.

        Parameters
        ----------
        copied: The object to which to copy the data.
        ep_fill: How to fill the rows in sef.ep_heatmap
        cdr3_fill: How to fill the rows in sef.cdr3_heatmap

        Returns
        -------
        A new Heatmap of the same type with the same data member values
        """
        copied.max_ep_length = self.max_ep_length
        copied.max_cdr3_length = self.max_cdr3_length

        import copy
        for row_name in self.ep_heatmap.keys():
            copied.ep_heatmap[row_name] = copy.deepcopy(ep_fill)
            copied.cdr3_heatmap[row_name] = copy.deepcopy(cdr3_fill)

        return copied

    def extract_max_ep_cdr3_length(self, sequences: Dict[str, Tuple[str, str]], pdb_subset: List[str] = None) -> None:
        """
        Extract the maximum epitope and cdr3 length among the sequences.

        Parameters
        ----------
        sequences: Dictionary that maps pdb IDs to its epitope and cdr3 sequences.
        pdb_subset: If not None, this list contains a subset of all pdb IDs. The pdb IDs that are not in this list
            are ignored.
        """
        values = [
            s for pdb_id, s in sequences.items()
            if pdb_subset is None or pdb_id in pdb_subset
        ]
        self.max_ep_length, self.max_cdr3_length = [len(max(values, key=lambda x: len(x[i]))[i]) for i in [0, 1]]

    def general_compose_heatmap(self, scrambled_heatmap=None, ignore_key: str = None):
        """
        Prepare the heatmap for compose_heatmap.
        """
        heatmaps_copy = [heatmap.copy() for heatmap in [self.ep_heatmap, self.cdr3_heatmap]]
        for heatmap in heatmaps_copy:
            for key in heatmap:
                if isinstance(heatmap[key], list) or isinstance(heatmap[key], np.ndarray):
                    heatmap[key] = np.nanmean(heatmap[key], axis=0)
                else:
                    heatmap[key] = np.array([
                        np.nanmean(heatmap[key][position])
                        for position in heatmap[key].keys()
                    ])
        ep_heatmap_copy, cdr3_heatmap_copy = heatmaps_copy

        self.heatmap = {
            key: np.concatenate((ep_heatmap_copy[key], [-1], cdr3_heatmap_copy[key]))
            for key in ep_heatmap_copy.keys()
        }

        if scrambled_heatmap is not None:
            scrambled_heatmap.compose_heatmap_correction()
            if ignore_key is not None:
                scrambled_heatmap.heatmap[ignore_key] = np.zeros(self.heatmap[ignore_key].size)
            self.calculate_diff(subtract=scrambled_heatmap.heatmap, normalize=True, max_ep_length=self.max_ep_length)

    def general_compose_heatmap_correction(self, ignore_key: str = None) -> None:
        """
        Combine a "correction" heatmap.
        This correction is formed by the values minus the mean of the values
        """
        self.compose_heatmap()
        if ignore_key:
            self.heatmap.pop(ignore_key)

        correction = {
            key: np.nanmean(self.heatmap[key])
            for key in self.heatmap.keys()
        }
        self.calculate_diff(subtract=correction, max_ep_length=self.max_ep_length)

    def create_plot(self, color_bar_label: str, filename: str) -> None:
        """
        Create the heatmap plot.

        Parameters
        ----------
        color_bar_label: The label for the color bar
        filename: The filename for the output figure
        """
        heatmap = np.array(list(self.heatmap.values()))
        heatmap = np.ma.masked_where(heatmap == -1, heatmap)

        x_ticks = list(range(self.max_ep_length + self.max_cdr3_length + 1))
        x_ticks.remove(self.max_ep_length)  # the -1 column
        x_labels = [
            '$\mathregular{e_{' + str(i + 1) + '}}$' for i in range(self.max_ep_length)
        ] + [
            '$\mathregular{c_{' + str(i + 1) + '}}$' for i in range(self.max_cdr3_length)
        ]

        self.write_plot(
            heatmap=heatmap,
            x_ticks=x_ticks,
            x_labels=x_labels,
            color_bar_label=color_bar_label,
            filename=filename
        )


class AAHeatmap2D(EpCDR3OnXAxisHeatmap):
    def __init__(self):
        EpCDR3OnXAxisHeatmap.__init__(self)

    def copy_initialization(self) -> 'AAHeatmap2D':
        """
        Copy the data members to avoid duplicate initialization.

        Returns
        -------
        A new AAHeatmap2D with the same data member values
        """
        return self.general_copy_initialization(
            copied=AAHeatmap2D(),
            ep_fill={i: [] for i in range(self.max_ep_length)},
            cdr3_fill={i: [] for i in range(self.max_cdr3_length)}
        )

    def initialize_aa_rows(self, sequences: Dict[str, Tuple[str, str]], pdb_id_subset: List[str] = None) -> None:
        """
        Extract the different AA occurring across the sequences, and create rows in the heatmap

        Parameters
        ----------
        sequences: Dictionary that maps pdb IDs to its epitope and cdr3 sequences.
        pdb_id_subset: If not None, this list contains a subset of all pdb IDs. The pdb IDs that are not in this list
            are ignored.
        """
        flattened_sequences = [
            sequence for pdb_id, sequence_pair in sequences.items() for sequence in sequence_pair
            if pdb_id_subset is None or pdb_id in pdb_id_subset
        ]
        str_sequences = ''.join(flattened_sequences)

        for row_name in sorted(list(set([letter for letter in str_sequences]))):
            self.ep_heatmap[row_name] = {i: [] for i in range(self.max_ep_length)}
            self.cdr3_heatmap[row_name] = {i: [] for i in range(self.max_cdr3_length)}

    def add_ep_aa_attribution(self, aa: str, position: int, attribution: float) -> None:
        """
        Add an AA attribution for a given epitope position

        Parameters
        ----------
        aa: Amino acid
        position: The position in the ep
        attribution: Attribution value for amino acid
        """
        self.ep_heatmap[aa][position].append(attribution)

    def add_cdr3_aa_attribution(self, aa: str, position: int, attribution: float) -> None:
        """
        Add an AA attribution for a given CDR3 position

        Parameters
        ----------
        aa: Amino acid
        position: The position in the ep
        attribution: Attribution value for amino acid
        """
        self.cdr3_heatmap[aa][position].append(attribution)

    def compose_heatmap(self, scrambled_heatmap=None) -> None:
        """
        Combine different AA into a single heatmap.

        Parameters
        ----------
        scrambled_heatmap: If not None, the difference between both is calculated.
        """
        self.general_compose_heatmap(scrambled_heatmap=scrambled_heatmap)

    def compose_heatmap_correction(self) -> None:
        """
        Combine a "correction" heatmap.
        This correction is formed by the values minus the mean of the values
        """
        self.general_compose_heatmap_correction()


class PositionalHeatmap(EpCDR3OnXAxisHeatmap):
    def __init__(self):
        EpCDR3OnXAxisHeatmap.__init__(self)

    def copy_initialization(self) -> 'PositionalHeatmap':
        """
        Copy the data members to avoid duplicate initialization.

        Returns
        -------
        A new PositionalHeatmap with the same data member values
        """
        return self.general_copy_initialization(
            copied=PositionalHeatmap(),
            ep_fill=[],
            cdr3_fill=[]
        )

    def initialize_rows(self, row_names: List[str]) -> None:
        """
        Fill the heatmap with the given names with an empty array.
        Parameters
        ----------
        row_names: List of row names
        """
        for row_name in row_names:
            self.ep_heatmap[row_name] = []
            self.cdr3_heatmap[row_name] = []

    def add_ep_attributions(self, row_name: str, attributions: np.ndarray) -> None:
        """
        Add attributions for an epitope

        Parameters
        ----------
        row_name: Name of the row where to add the attribution values
        attributions: A numpy array with attributions value for amino acid
        """
        self.ep_heatmap[row_name].append(attributions)

    def add_cdr3_attributions(self, row_name: str, attributions: np.ndarray) -> None:
        """
        Add attributions for a CDR3

        Parameters
        ----------
        row_name: Name of the row where to add the attribution values
        attributions: A numpy array with attributions value for amino acid
        """
        self.cdr3_heatmap[row_name].append(attributions)

    def compose_heatmap(self, scrambled_heatmap=None) -> None:
        """
        Combine different AA into a single heatmap.

        Parameters
        ----------
        scrambled_heatmap: If not None, the difference between both is calculated.
        """
        self.general_compose_heatmap(scrambled_heatmap=scrambled_heatmap, ignore_key='Residue proximity')

    def compose_heatmap_correction(self) -> None:
        """
        Combine a "correction" heatmap.
        This correction is formed by the values minus the mean of the values
        """
        self.general_compose_heatmap_correction(ignore_key='Residue proximity')


class SampleDetailsHeatmap(Heatmap):
    def __init__(self):
        Heatmap.__init__(self, x='ep cdr3')

        self.ep_heatmap: Dict[str, np.ndarray] = dict()
        self.cdr3_heatmap: Dict[str, np.ndarray] = dict()

        self.ep_sequence: str = None
        self.cdr3_sequence: str = None

    def copy_initialization(self) -> 'SampleDetailsHeatmap':
        """
        Copy the data members to avoid duplicate initialization.

        Returns
        -------
        A new SampleDetailsHeatmap with the same data member values
        """
        copied = SampleDetailsHeatmap()
        copied.ep_sequence = self.ep_sequence
        copied.cdr3_sequence = self.cdr3_sequence

        copied.initialize_rows(row_names=self.ep_heatmap.keys())

        return copied

    def reset(self) -> None:
        """
        Reset the data members
        """
        self.ep_heatmap: Dict[str, np.ndarray] = dict()
        self.cdr3_heatmap: Dict[str, np.ndarray] = dict()
        self.ep_sequence: str = None
        self.cdr3_sequence: str = None

    def initialize_rows(self, row_names: List[str]) -> None:
        """
        Fill the heatmap with the given names with an empty array.
        Parameters
        ----------
        row_names: List of row names
        """
        for row_name in row_names:
            self.ep_heatmap[row_name] = np.zeros(0)
            self.cdr3_heatmap[row_name] = np.zeros(0)

    def set_ep_values(self, row_name: str, values: np.ndarray) -> None:
        """
        Set the epitope value for a given row

        Parameters
        ----------
        row_name: Name of the row (= key in dictionary)
        values: Values to set
        """
        self.ep_heatmap[row_name] = values

    def set_cdr3_values(self, row_name: str, values: np.ndarray) -> None:
        """
        Set the cdr3 value for a given row

        Parameters
        ----------
        row_name: Name of the row (= key in dictionary)
        values: Values to set
        """
        self.cdr3_heatmap[row_name] = values

    def set_sequence(self, ep_sequence: str, cdr3_sequence: str) -> None:
        """
        Set the sequences (used for x-tick labels)

        Parameters
        ----------
        ep_sequence: The epitope sequence
        cdr3_sequence: The CDR3 sequence
        """
        self.ep_sequence = ep_sequence
        self.cdr3_sequence = cdr3_sequence

    def compose_heatmap(self, scrambled_heatmap = None) -> None:
        """
        Combine different AA into a single heatmap.
        """
        self.heatmap = {
            row_name: np.concatenate((self.ep_heatmap[row_name], [-1], self.cdr3_heatmap[row_name]))
            for row_name in self.ep_heatmap.keys()
        }

        if scrambled_heatmap is not None:
            scrambled_heatmap.compose_heatmap_correction()
            self.calculate_diff(subtract=scrambled_heatmap.heatmap, normalize=True)

    def compose_heatmap_correction(self) -> None:
        """
        Combine a "correction" heatmap.
        This correction is formed by the values minus the mean of the values
        """
        self.compose_heatmap()

        correction = {
            key: np.nanmean(self.heatmap[key])
            for key in self.heatmap.keys()
        }
        self.calculate_diff(subtract=correction, max_ep_length=len(self.ep_sequence))

    def create_plot(self, color_bar_label: str, filename: str) -> None:
        """
        Create the heatmap plot.

        Parameters
        ----------
        color_bar_label: The label for the color bar
        filename: The filename for the output figure
        """
        heatmap = np.array(list(self.heatmap.values()))
        heatmap = np.ma.masked_where(heatmap == -1, heatmap)

        x_ticks = list(range(len(self.ep_sequence + self.cdr3_sequence) + 1))
        x_ticks.remove(len(self.ep_sequence))  # the -1 column
        x_labels = [c for c in self.ep_sequence + self.cdr3_sequence]

        self.write_plot(
            heatmap=heatmap,
            x_ticks=x_ticks,
            x_labels=x_labels,
            color_bar_label=color_bar_label,
            filename=filename
        )


