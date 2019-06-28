import argparse
import os
import sys

import h5py
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        'Converts expert trajectories from h5 to pt format.')
    parser.add_argument(
        '--h5-file',
        default='trajs_HalfCheetah-v2_00000.h5',
        help='input h5 file',
        type=str)
    parser.add_argument(
        '--pt-file',
        default=None,
        help='output pt file, by default replaces file extension with pt',
        type=str)
    args = parser.parse_args()

    if args.pt_file is None:
        args.pt_file = os.path.splitext(args.h5_file)[0] + '.pt'

    h5_files = os.listdir()
    h5_files = [f for f in h5_files if '.h5' in f]
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            dataset_size = f['states'].shape[0]  # full dataset size

            states = f['states'][:dataset_size, ...][...]
            actions = f['actions'][:dataset_size, ...][...]
            rewards = f['rewards'][:dataset_size, ...][...]
            done = f['done'][:dataset_size, ...][...]
            lens = f['lengths'][:dataset_size, ...][...]

            states = torch.from_numpy(states).float()
            actions = torch.from_numpy(actions).float()
            rewards = torch.from_numpy(rewards).float()
            done = torch.from_numpy(done).float()
            lens = torch.from_numpy(lens).long()

        data = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'done':done,
            'lengths': lens
        }

        pt_file = os.path.splitext(h5_file)[0] + '.pt'
        torch.save(data, pt_file)


if __name__ == '__main__':
    main()
