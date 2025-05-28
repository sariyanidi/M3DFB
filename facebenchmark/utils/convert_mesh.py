import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('srcmesh_path', type=str, help="Path of the source mesh. This is assumed to be a .txt file.")
parser.add_argument('dstmesh_path', type=str, help="Path of the destination mesh. The created mesh will be written as a .txt file.")
parser.add_argument('--srcmesh_type', type=str,  default="BFM-53490")
parser.add_argument('--dstmesh_type', type=str,  default="BFM-23470")
parser.add_argument('--verbosity_level', type=int,  default=1)

args = parser.parse_args()

if os.path.exists(args.dstmesh_path) and args.verbosity_level > 0:
    print(f'Skipping -- destination file already exists: {args.dstmesh_path}')
    sys.exit(1)

if not os.path.exists(args.srcmesh_path):
    print(f'Source mesh not found at {args.srcmesh_path}', file=sys.stderr)
    sys.exit(1)

supported_srcmesh_types = ['BFM-53490', 'BFM-53215']
if args.srcmesh_type not in supported_srcmesh_types: 
    print(f'Given source mesh type is {args.srcmesh_type}, but we currently support only the following source mesh types: {supported_srcmesh_types}', file=sys.stderr)
    sys.exit(1)

if args.dstmesh_type != 'BFM-23470':
    print(f'We currently support only BFM-23470 as dstmesh_type', file=sys.stderr)
    sys.exit(1)


src = np.loadtxt(args.srcmesh_path)
Nexpected = int(args.srcmesh_type.split('-')[1])

if src.shape[0] != Nexpected:
    print(f'Expected a mesh with {Nexpected} vertices but got a mesh with {src.shape[0]} vertices', file=sys.stderr)
    sys.exit(1)

if src.shape[1] != 3:
    print('Number of dimensions in input mesh is not 3', file=sys.stderr)
    sys.exit(1)

if args.srcmesh_type == 'BFM-53490' and args.dstmesh_type == 'BFM-23470':
    ix = np.loadtxt('idxs/ix_23470_relative_to_53490.txt').astype(int)
    dst = src[ix,:]
elif args.srcmesh_type == 'BFM-53215' and args.dstmesh_type == 'BFM-23470':
    src_full = np.zeros((53490, 3))
    
    ix53215 = np.loadtxt('idxs/ix_53215_relative_to_53490.txt').astype(int)
    src_full[ix53215,:] = src

    ix23470 = np.loadtxt('idxs/ix_23470_relative_to_53490.txt').astype(int)
    dst = src_full[ix23470,:]

np.savetxt(args.dstmesh_path, dst)
sys.exit(0)






