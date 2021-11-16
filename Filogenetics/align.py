from Bio.Align.Applications import ClustalwCommandline
import sys

infile = sys.argv[1]
outfile = infile.split('.')[0] + '_aln'
print(infile)
print(outfile)
cmd = ClustalwCommandline('clustalw', infile=infile, gapext=0.5, align=True,
                          gapopen=10, matrix='matrix.txt',
                          pwmatrix='matrix.txt',
                          type='PROTEIN', outfile=outfile, quiet=True)

res = cmd()
for i in res:
    print(i)
