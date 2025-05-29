#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import multiprocessing as mp

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('experiment_filepath', type=str, help="JSON file that contains experiment info")
parser.add_argument('data_dir', type=str, help="Path to directory that contains data")
parser.add_argument('--num_processes', type=int, default=1)

args = parser.parse_args()

avail_processes = min(mp.cpu_count(), args.num_processes)
if args.num_processes > avail_processes:
    print(f'Warning: reducing the number of processes to {avail_processes}')
    args.num_processes = avail_processes 

mod = __import__("facebenchmark")

with open(args.experiment_filepath) as f:
    experiment = json.load(f)

reporter_class = getattr(mod, experiment['reporter_type'])
reporter_opts = experiment['reporter_opts']

error_computers = []
for ec_json in experiment['error_computers']:
    with open(ec_json) as f:
        error_computers.append(json.load(f))

mms_info = {}
for mm in experiment['mms_info']:
    with open(experiment['mms_info'][mm]) as f:
        mms_info[mm] = json.load(f)

reporter = reporter_class(experiment['dataset'], error_computers, 
                          experiment['rec_methods'], mms_info, args.data_dir,
                          num_processes=args.num_processes,
                          use_cache=True,
                          opts=reporter_opts)
if __name__ == '__main__':
    reporter.produce()
