'''
Generates script for running many experiemnts
'''
from lfh.utils.config import Configurations
import numpy as np
import argparse

### Experimental Parameters #
np.random.seed(569)
SEEDS = np.random.randint(0, 10000, size=3)
AGENTS = ["uniform_zpd", "unseq_DDQN"]
OFFSETS = [ -64, -32, -8, -4, 8, 32, 64]
RADII = [32, 64, 128]
MIX_RATIOS = [1/4]

###############################

def to_command(dic):
    command = 'python main.py'
    for key, value in dic.items():
        command += ' --{} {}'.format(key, value)
    return command + '\n'

def get_args(): 
    """
    This function will extract the arguments from the command line
    """
    parser = argparse.ArgumentParser(description='Initial Experiments for Tabular')
    parser.add_argument('--output_type',  default='compute_canada_format', type=str, choices=("bash_file", "compute_canada_format", "execute"), help="What should be the output of this file: bash_file: generates a bash file of the commands that you should run in linux for all the experiments | compute_canada_format: generates a file that can be used to run all the experiments on compute canada | execute: will run all experiments on your computer")
    parser.add_argument('--output_path', default='experiments', type=str,
            nargs='?', help="The path to save the output file of this script")
    return vars(parser.parse_args())


def main(args):
    bash_file_commands = []

    for seed in SEEDS:
        for agent in AGENTS:
            if agent == "uniform_zpd":
                for offset in OFFSETS:
                    for radius in RADII:
                        for mix_ratio in MIX_RATIOS:
                                run_args = {}
                                run_args["agent"] = agent
                                run_args["offset"] = offset
                                run_args["radius"] = radius
                                run_args["mix-ratio"] = mix_ratio
                                run_args["seed"] = seed

                                if args['output_type'] == 'bash_file':
                                    bash_file_commands.append(to_command(run_args))
                                elif args['output_type'] == 'compute_canada_format':
                                    bash_file_commands.append(to_command(run_args))
            else:
                for mix_ratio in MIX_RATIOS:
                    run_args = {}
                    run_args["mix-ratio"] = mix_ratio
                    run_args["agent"] = agent
                    run_args["seed"] = seed

                    if args['output_type'] == 'bash_file':
                        bash_file_commands.append(to_command(run_args))
                    elif args['output_type'] == 'compute_canada_format':
                        bash_file_commands.append(to_command(run_args))


    if args['output_type'] == 'bash_file':
        with open(args['output_path'] + '.bash', 'w') as output:
            for row in bash_file_commands:
                output.write(str(row))

    elif args['output_type'] == 'compute_canada_format':
        with open(args['output_path'] + '.txt', 'w') as output: # This .txt file can use a command list for GNU Parallel
            for row in bash_file_commands:
                output.write(str(row))
 

if __name__ == '__main__':
    ARGS = get_args()
    main(ARGS)
