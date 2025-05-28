import os
import argparse
import numpy as np
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('method_path', type=str)
parser.add_argument('F', type=int)
parser.add_argument('--M', type=int, default=1)
parser.add_argument('--Nsubjs', type=int, default=100)
parser.add_argument('--crop_map', type=str, default='./utils/ix_common.txt')

args = parser.parse_args()

Nframes_per_rec = args.F*args.M

srcdir = args.method_path
dstdir = f'{srcdir}_{Nframes_per_rec:03d}'

if not os.path.exists(dstdir):
    os.mkdir(dstdir)

vertex_ix = None
if args.crop_map is not None:
    vertex_ix = np.loadtxt(args.crop_map).astype(int)
    
for n in range(args.Nsubjs):
    Nrecs = int(args.F/args.M)
    files = glob(f'{srcdir}/id{n:04d}_*txt')[:Nrecs]

    avg_face = np.mean([np.loadtxt(f) for f in files], axis=0)
    if vertex_ix is not None:
        avg_face = avg_face[vertex_ix, :]

    tpath = f'{dstdir}/id{n:04d}.txt'
    np.savetxt(tpath, avg_face)
