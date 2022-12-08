import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from matplotlib import pyplot as plt

from src.imrex_attributions import ImrexAttributionsHandler


def pdb_stats():
    df = pd.read_csv('data/tcr3d_imrex_output.csv')
    print(len(df['antigen.epitope'].unique()))
    print(len(df['cdr3'].unique()))


def tcrdist_comparison():
    df = pd.read_csv('data/tcr3d_imrex_output.csv')
    df = df.drop_duplicates(subset=['cdr3', 'antigen.epitope'])
    most_ep = df['antigen.epitope'].value_counts(sort=True)
    centroids = {'ELAGIGILTV': 'CAWSETGLGTGELFF', 'GILGFVFTL': 'CASSIRSSYEQYF'}
    for i in range(2):
        df_ep = df[df['antigen.epitope'] == most_ep.index[i]]
        # tcrdist_comparison_ep(df_ep)
        imrex_per_ep_attributions(most_ep.index[i], df_ep, centroids[most_ep.index[i]])


def tcrdist_comparison_ep(df):
    df_full = pd.read_csv('data/MHCI.csv')
    vdjdb = pd.read_csv('ImRex/data/interim/vdjdb-2022-03-30/vdjdb.txt', sep='\t')

    v_genes = []
    for pdb_id in df['PDB_ID']:
        v_genes.append(df_full[df_full['PDB_ID'] == pdb_id].iloc[0]['TRBV_gene'] + '*01')
    df = df.assign(v_b_gene=v_genes)

    j_genes = []
    for cdr3 in df['cdr3'].values:
        print(f">seq{cdr3}\n{cdr3}")
    for cdr3, ep in zip(df['cdr3'], df['antigen.epitope']):
        print(cdr3, ep)
        vdjdb_sample = vdjdb[(vdjdb['cdr3'] == cdr3) & (vdjdb['antigen.epitope'] == ep)]
        # print(vdjdb_sample)
        vdjdb_sample = vdjdb_sample.drop_duplicates(
            subset=['v.segm', 'j.segm'])
        # print(vdjdb_sample)
        if not vdjdb_sample.empty:
            j_genes.append(vdjdb_sample.iloc[0]['j.segm'])
            for v_gene, j_gene in zip(vdjdb_sample['v.segm'], vdjdb_sample['j.segm']):
                print(v_gene, j_gene)
        else:
            j_genes.append(None)
            print('Not found in VDJDB')
        print()
    df = df.assign(j_b_gene=j_genes)
    df['cdr3_b_aa'] = df['cdr3']
    df['epitope'] = df['antigen.epitope']

    from tcrdist.repertoire import TCRrep
    tr = TCRrep(cell_df=df,
                organism='human',
                chains=['beta'],
                db_file='alphabeta_gammadelta_db.tsv')
    # print(tr.clone_df)
    from tcrdist.rep_diff import hcluster_diff, member_summ
    tr.hcluster_df, tr.Z = \
        hcluster_diff(clone_df=tr.clone_df,
                      pwmat=tr.pw_beta,
                      x_cols=['epitope'],
                      count_col='count')
    # print(tr.hcluster_df)
    # exit()

    from tcrsampler.sampler import TCRsampler

    t = TCRsampler()
    # t.download_background_file("ruggiero_human_sampler.zip")
    tcrsampler_beta = TCRsampler(default_background='ruggiero_human_beta_t.tsv.sampler.tsv')

    from tcrdist.adpt_funcs import get_centroid_seq
    from tcrdist.summarize import _select
    from palmotif import compute_pal_motif, svg_logo

    """Beta Chain"""
    svgs_beta = list()
    for i, r in tr.hcluster_df.iterrows():

        dfnode = tr.clone_df.iloc[r['neighbors_i'],]
        if dfnode.shape[0] > 2:
            centroid, *_ = get_centroid_seq(df=dfnode)
        else:
            centroid = dfnode['cdr3_b_aa'].to_list()[0]
        print(f"BETA-CHAIN: {centroid}")

        gene_usage_beta = dfnode.groupby(['v_b_gene', 'j_b_gene']).size()
        sampled_rep = tcrsampler_beta.sample(gene_usage_beta.reset_index().to_dict('split')['data'],
                                             flatten=True, depth=10)
        sampled_rep = [x for x in sampled_rep if x is not None]
        motif, stat = compute_pal_motif(
            seqs=_select(df=tr.clone_df,
                         iloc_rows=r['neighbors_i'],
                         col='cdr3_b_aa'),
            refs=sampled_rep,
            centroid=centroid)
        svg_logo(motif, filename=f'output/logos/beta_{df["antigen.epitope"].iloc[0]}_{i}.svg')
        svgs_beta.append(svg_logo(motif, return_str=True))

    """Add Beta SVG graphics to hcluster_df"""
    tr.hcluster_df['svg_beta'] = svgs_beta
    res_summary = member_summ(res_df=tr.hcluster_df,
                              clone_df=tr.clone_df,
                              addl_cols=['epitope'])

    tr.hcluster_df_detailed = \
        pd.concat([tr.hcluster_df, res_summary], axis=1)
    """
    Write D3 html for interactive denogram graphic. 
    Specify desired tooltips.
    """
    from hierdiff import plot_hclust_props
    html = plot_hclust_props(tr.Z,
                             title='PA Epitope Example',
                             res=tr.hcluster_df_detailed,
                             tooltip_cols=['cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'svg_beta'],
                             alpha=0.00001, colors=['blue', 'gray'],
                             alpha_col='pvalue')

    with open(f'output/hierdiff_example_{df["antigen.epitope"].iloc[0]}.html', 'w') as fh:
        fh.write(html)


def imrex_per_ep_attributions(ep, df_ep, centroid):
    imrex_attributions = ImrexAttributionsHandler(
        name="imrex_nocdr3dup",
        display_name="ImRex",
        model_path="ImRex/models/models/2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv/iteration_2/"
                   "2022-01-06_11-03-43_nocdr3dup_default_epgrouped5cv-epoch20.h5",
        image_path="data/tcr3d_images/",
        save_folder="data"
    )

    model_attributions = imrex_attributions.get_aa_norm_attributions()
    sequences = imrex_attributions.get_sequences()
    model_attributions = {k: model_attributions[k] for k in df_ep['PDB_ID']}
    sequences = {k: sequences[k] for k in df_ep['PDB_ID']}

    eps = [v[0] for k, v in sequences.items()]
    cdr3s = [v[1] for k, v in sequences.items()]

    max_ep = len(max(eps, key=len))
    # max_cdr3 = len(max(cdr3s, key=len))

    sg_attributions = []
    # for pdb_id, attr in model_attributions.items():
    #     ep = sequences[pdb_id][0]
    #     sg_attributions.append((aa_add_padding(attr['SmoothGrad'][:len(ep)], max_ep),
    #                             aa_add_padding(attr['SmoothGrad'][len(ep):], max_cdr3)))
    for pdb_id, attr in model_attributions.items():
        ep = sequences[pdb_id][0]
        sg_attributions.append((attr['SmoothGrad'][:len(ep)], attr['SmoothGrad'][len(ep):]))
    names = ['ImRex']

    # pos_attributions_ep = [a[0] for a in sg_attributions]
    pos_attributions_cdr3 = [a[1] for a in sg_attributions]

    print(pos_attributions_cdr3)

    al_attributions_cdr3 = []
    for cdr3, attribution in zip(cdr3s, pos_attributions_cdr3):
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -3
        aligner.extend_gap_score = -3
        cdr3_a, centroid_a = aligner.align(cdr3, centroid)[0].aligned
        al_seq = [np.nan for _ in range(len(centroid))]
        for cdr3_t, centroid_t in zip(cdr3_a, centroid_a):
            al_seq[centroid_t[0]:centroid_t[1]] = attribution[cdr3_t[0]:cdr3_t[1]]
        al_attributions_cdr3.append(al_seq)

        print(aligner.align(cdr3, centroid)[0])
        print(cdr3_a, centroid_a)
        print(al_seq)
        print()

    # exit()

    heatmap = [np.nanmean(al_attributions_cdr3, 0)]

    plt.gcf().set_size_inches(10 / 1.1, 3 / 1.1)
    heatmap = np.array(heatmap)
    heatmap = np.ma.masked_where(heatmap == -1, heatmap)
    grid = plt.imshow(heatmap, cmap='Greys')
    plt.xticks(list(range(len(centroid))), ['$\mathregular{c_{' + str(i + 1) + '}}$' for i in range(len(centroid))])
    plt.yticks(list(range(len(heatmap))), names)
    plt.colorbar(grid, orientation='horizontal', pad=0.2, label=f"SmoothGrad feature attribution")
    plt.clim(0, 1)
    plt.tight_layout()
    plt.savefig(f'output/plots/average_pos_attributions_SmoothGrad_{ep}.png', dpi=300, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    tcrdist_comparison()
    # pdb_stats()
