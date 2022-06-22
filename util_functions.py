import os
import itertools
import numpy as np
import pandas as pd
import requests as rq

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Align.Applications import ClustalwCommandline

from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_multiple
from biotite.sequence.graphics import plot_alignment_type_based

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
plt.rcParams['svg.fonttype'] = 'none'

lab = pd.read_csv('seqs/csv/Short_len_upar.csv', index_col=0)
lab.Prot = lab.Prot.str.replace(' ', '_').str.upper()
lab = lab.Prot.to_dict()


def retrieve_uniprot_data_by_acess(acess):
    """
    Retrieve protein by ID from UniProt
    """
    requestURL = f"https://www.ebi.ac.uk/proteins/api/proteins/{acess}"
    r = rq.get(requestURL, headers={"Accept": "application/json"})

    if not r.ok:
        r.raise_for_status()
        return 0
    responseBody = r.json()

    try:
        features = responseBody['features']
    except KeyError:
        type_ = None
        start, end = np.nan, np.nan
    else:
        st_nd = [(start := i['begin'], end := i['end']) for i in features
                 if i.get('description') and i['description'] == 'UPAR/Ly6']
        type_ = 'DOMAIN'
        if not st_nd:
            type_ = 'CHAIN'
            st_nd = [(start := i['begin'], end := i['end'])
                     for i in features if i['type'] == 'CHAIN']

    try:
        prot_name = responseBody['protein']['submittedName'][0]
        prot_name = prot_name['fullName']['value']

    except (KeyError, IndexError):
        prot_name = responseBody['protein']['recommendedName']
        prot_name = prot_name['fullName']['value']

    return {'sequence': responseBody['sequence']['sequence'],
            'id': responseBody['id'],
            'protein_name': prot_name,
            'org': responseBody['organism']['names'][0]['value'],
            'start': start,
            'end': end,
            'type_': type_,
            'acess': acess}


def get_domain(x):
    if hasattr(x, 'start') and x.type_ == 'DOMAIN':
        n_st = x.start - 6
    else:
        n_st = 0
    n_end = len(x.sequence)
    result = []
    for st, en in itertools.product(
        np.arange(n_st, n_end - 20),
        np.arange(n_st + 20, n_end + 1)
    ):
        if len((string := x.sequence[st: en])) > 45:
            if ('CN' == string[-2:]
                    and string[2] == 'C'
                    and string.count('C') % 2 == 0
                    and string.count('C') >= 8
                    and string.count('C') <= 12
                    and len(string) < 100):

                result.append(string)

    return max(result, key=lambda x: len(x)) if result else np.nan


def rename_for_lab(series: pd.Series):
    lab_name = lab.get(series.acess)
    if lab_name:
        series.id = lab_name + '_' + series.id.split('_')[1]
    return series


def save_df_to_fasta(df, filename, seq_col='sequence', org='org',
                     protein_name='protein_name'):
    req_list = []
    for ind, row in df.iterrows():
        req_list.append(SeqRecord(seq=Seq(row[seq_col]),
                                  id=row.id,
                                  name=row[protein_name],
                                  description=('| '
                                               + row[protein_name]
                                               + ' | '
                                               + row[org])
                                  ))
    with open(filename, 'w') as f:
        SeqIO.write(req_list, f, 'fasta')


def read_blast(name, org):
    df = pd.read_csv(f'seqs/csv/{name}', index_col=0)
    bl_name = name.split('_', 1)[0]
    bl = pd.read_csv(f'blast/{bl_name}_blast_short.txt',
                     index_col=0, sep='\t',
                     header=None).index.unique().tolist()
    df = df.loc[bl].reset_index()
    df['org'] = org
    df['genes'] = df.genes + '_|_' + df.org.str.replace(' ', '_')
    df['domain'] = df.apply(get_domain, axis=1)
    return df.drop_duplicates('domain').set_index('genes')
# [df.sequence.str.len() < 190]


def align(in_, out):
    fasta, align = in_, out

    cmd = ClustalwCommandline('clustalw', infile=fasta, align=True,
                              matrix='matrix.txt', pwmatrix='matrix.txt',
                              type='PROTEIN', outfile=align, quiet=True)
    x = cmd()
    for i in x:
        print(i)

    os.rename(fasta.rsplit('.', 1)[0] + '.dnd',
              align.rsplit('.', 1)[0] + '.dnd')


def print_heatmap(path, organism):
    sp = 'Star protein'
    lp = 'Ly6/uPAR protein'
    ev = 'e-value'
    bs = 'bit score'
    df = pd.read_csv(path, sep='\t', header=None).rename(columns={
        0: sp,
        1: lp,
        10: ev,
        11: bs
    })
    kwargs = {'cmap': 'mako', 'linecolor': 'white', "linewidths": 0.1}
    fig, ax = plt.subplots(1, 2, figsize=(27.5, 10))
    sns.heatmap(data=df.pivot(index=lp, columns=sp, values=ev).fillna(1),
                ax=ax[0], vmax=1e-05, **kwargs)

    kwargs['cmap'] = kwargs['cmap'] + '_r'
    sns.heatmap(data=df.pivot(index=lp, columns=sp, values=bs).fillna(0),
                ax=ax[1], **kwargs)
    for i in ax:
        i.set_xticklabels(i.get_xticklabels(), rotation=45,
                          ha="right", va='center_baseline')
    fig.suptitle(organism, fontsize=30)
    plt.tight_layout()
    plt.savefig(f"result/img/{path.split('/')[-1].rsplit('.', 1)[0]}.svg",
                format='svg')
    plt.show()


def similarity(x: str, y: str):
    d = {
        '-': 0,
        'A': 1, 'F': 1, 'H': 1, 'I': 1, 'L': 1, 'M': 1,
        'P': 1, 'R': 1, 'V': 1, 'W': 1, 'X': 1,
        'C': 2, 'D': 3, 'E': 3,
        'G': 4, 'N': 4, 'Q': 4, 'S': 4, 'T': 4, 'Y': 4,
        'K': 5
    }
    res = [1 for i, j in enumerate(x) if d[j] == d[y[i]]]
    return f'{round(sum(res) * 100 / len(x), 1)}%'


def get_aln(gene: str, org: str, star_blast: pd.DataFrame,
            df: pd.DataFrame, uniprot: pd.DataFrame):
    name = gene.replace(' ', '_') + '_|_' + org.replace(' ', '_')
    seq = star_blast.loc[[name]]
    ly6upar_ser = uniprot.set_index('id').loc[df.loc[gene, 'hit ly6 protein']]

    if isinstance(ly6upar_ser, pd.Series):
        ly6upar_ser = pd.DataFrame(ly6upar_ser).T
    return pd.concat(
        [ly6upar_ser, seq]
        ).fillna('').reset_index().rename(columns={'index': 'id'})


def get_pictures(star_blast, df, uniprot, organism):

    star_genes = df.index.unique().tolist()
    color_map = "-=#ffffff\nA=#7eff00\nC=#ffe300\nD=#ff0000\nE=#ff0000\nF=#7eff00\nG=#ff00e4\nH=#7eff00\nI=#7eff00\nK=#007bff\nL=#7eff00\nM=#7eff00\nN=#ff00e4\nP=#7eff00\nQ=#ff00e4\nR=#007bff\nS=#ff00e4\nT=#ff00e4\nV=#7eff00\nW=#7eff00\nX=#7eff00\nY=#ff00e4"
    color_map = {i.split('=')[0]: i.split('=')[1]
                 for i in color_map.split('\n')}
    for gene in star_genes:
        try:
            tmp = get_aln(gene, org=organism, star_blast=star_blast,
                          df=df, uniprot=uniprot)
        except KeyError as e:
            print(gene, 'protein has duplicated domain\n', e)
            continue
        print_aln_lystar(gene, organism, tmp, color_map)


def get_genes_for_point_aln(star_name):
    df = pd.read_csv(f'blast/{star_name}_blast_short.txt',
                     sep='\t', header=None)
    cols = [
        'star gene',
        'hit ly6 protein',
        'ident percent',
        'alignment length',
        'mismatch',
        'gapopen',
        'alignment start aa from star protein',
        'alignment end aa from star protein',
        'alignment start aa from ly6 protein',
        'alignment end aa from ly6 protein',
        'blast e-value',
        'blast bit score',
    ]
    df.columns = cols
    df = df.set_index(cols[0])
    return df


def get_combs_of_loops(loops):
    try:
        combs = [
            (loops[1], loops[2]),
            (loops[0], loops[4]),
            (loops[3], loops[5]),
            (loops[6], loops[9]),
            (loops[7], loops[8]),
            (loops[10], loops[11]),
        ]
    except IndexError:
        try:
            combs = [
                (loops[1], loops[2]),
                (loops[0], loops[4]),
                (loops[3], loops[5]),
                (loops[6], loops[7]),
                0,
                (loops[8], loops[9]),
            ]
        except IndexError:
            combs = [
                (loops[1], loops[3]),
                (loops[0], loops[2]),
                0,
                (loops[4], loops[5]),
                0,
                (loops[6], loops[7]),

            ]
    return combs


def print_loops(loops, ax, pos=-0.1, color='r'):
    if pos > 0:
        j = pos + 1.4
        st = j + 1
        add_v = -1
    else:
        j = -1.4
        st = j - 1
        add_v = 1

    if pos > 0:
        ls = {v: k for k, v in dict(enumerate(
              get_combs_of_loops(loops[1]))).items()}
        combs = sorted(set(ls.keys()) - set(get_combs_of_loops(loops[0])),
                       key=lambda x: ls.get(x))
    else:
        combs = get_combs_of_loops(loops)

    for i in combs:
        if i:
            ax.plot(i, [j, j], color=color)
            ax.plot([i[0]] * 2, [pos, j], color=color)
            ax.plot([i[1]] * 2, [pos, j], color=color)

        if pos > 0:
            j += -add_v if j < st else add_v
        else:
            j += -add_v if j > st else add_v


def print_aln_lystar(gene, organism, tmp, color_map):
    gene = gene.split('_|_')[0]
    sequence_dict = {k: ProteinSequence(v)
                     for k, v in tmp.set_index('id').domain.to_dict().items()}
    headers = list(sequence_dict.keys())
    sequences = list(sequence_dict.values())

    # Perform a multiple sequence alignment
    matrix = pd.read_csv('matrix.txt', sep='\s+', skiprows=1, index_col=0)
    matrix = matrix.melt(ignore_index=False)

    alph1 = sequences[0].get_alphabet()
    alph2 = sequences[0].get_alphabet()
    matrix_dict = matrix.set_index('variable', append=True).value.to_dict()

    matrix = SubstitutionMatrix(alph1, alph2, matrix_dict)
    alignment, order, _, _ = align_multiple(sequences, matrix)

    # Order alignment according to guide tree
    alignment = alignment[:, order.tolist()]

    headers = [headers[i].replace('_', ' ') for i in order]
    headers = [name for i, name in enumerate(headers)]

    fig, ax = plt.subplots(1, 1, figsize=(
        len(max(sequences, key=len)) / 3, 4)
        )

    loops = plot_alignment_type_based(
        ax, alignment, labels=headers,
        symbols_per_line=len(alignment),
        label_size=15, symbol_size=12, number_size=15,
        similarity_kwargs={'func': similarity,
                           'label': 'Similarity',
                           'refseq': 0},
        show_similarity=True, color_symbols=True,
        color_scheme=list(map(color_map.get, sequences[0].get_alphabet()))
    )
    adj = 0.6
    color = 'orange'
    print_loops(loops[0], ax, -adj, color)

    if loops[0] != loops[1]:
        print_loops(loops, ax, len(sequences) + adj, color)

    ax.set_aspect('equal', share=True)

    plt.tight_layout()
    fig.savefig(f'result/img/a{organism.split(" ")[1]}_{gene}_alignment.eps', format='eps')
    fig.savefig(f'result/img/a{organism.split(" ")[1]}_{gene}_alignment.svg')
    plt.show()
