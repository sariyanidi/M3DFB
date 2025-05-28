import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('mesh_path', type=str)
parser.add_argument('lmks_path', type=str)

args = parser.parse_args()

li = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508] 

X = np.loadtxt(args.mesh_path)
lmks = X[li,:]
np.savetxt(args.lmks_path, lmks)
# plt.figure(figsize=(20, 20))
# plt.plot(X[li,0], -X[li,1])
# plt.plot(X[:,0], -X[:,1],'.')