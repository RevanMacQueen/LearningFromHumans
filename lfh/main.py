'''
Main file for training agent with demonstrations
'''
import os
import argparse
import numpy as np
import time
import logging
from pathlib import Path

# from lfh.envs.atari import make_env
from utils.io import load_demonstrations


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="breakout_standard", help="ID of the experiment to run")
    parser.add_argument("--exp-name", type=str, default='settings',
                        help='directory from which to load experiment settings')
    parser.add_argument("--profile", action='store_true', help='whether to run the profiler')
    parser.add_argument("--demonstrations-dir", type=str, default='../Demonstrations/BreakoutDemonstrations/', help='directory containing demonstrations')

    return vars(parser.parse_args())




def main(args : dict):

    load_demonstrations(args["demonstrations_dir"])



if __name__ == "__main__":
    # Read configurations from file
    args = get_args()
    main(args)