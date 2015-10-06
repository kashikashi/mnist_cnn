# -*- coding: utf-8 -*- 

import numpy as np
import sys

argvs = sys.argv
argc = len(argvs)

if not argc == 2:
    sys.stderr.write("python calc_score.py <feats dim>\n")
    quit()

ssize=int(argvs[1])
R_mat=np.zeros((ssize,ssize))

for line in sys.stdin:

    if line.find("[") >=0:
        label=line.strip().split()[0]
        sys.stdout.write(label + "\r")
        sys.stdout.flush()

    else:
        best_results=np.array(map(float,line.strip().split()[0:ssize])).argmax()
        label=int(line.strip().split()[ssize])

        R_mat[best_results,label]+=1

sys.stdout.write("\n")

precision=np.diag(R_mat).sum() / ( R_mat.sum())
print precision
