import os
import json
import pickle
# import torch
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp


def dump_pickle(_object, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


def create_sub_dir(params):
    """Add different subdirectories here.

    New keys are introduced in the params[log], and can use later in code.
    Applies for whatever is the current log dir, for students or teachers, it
    will apply only to its own log dir. Thus, teacher agents actually have
    `learner_snapshots` as subdirs, though they don't put anything there.
    """
    for subdirectory in ["snapshots", "episodes", "learner_snapshots"]:
        _param_name = "dir_{0}".format(subdirectory)
        params["log"][_param_name] = os.path.join(params["log"]["dir"],
                                                  subdirectory)
        os.makedirs(params["log"][_param_name])


def create_output_dir(params):
    """From `Configurations` class to create output directory.
    """
    assert not os.path.exists(params["log"]["dir"]), \
        "Error, {} exists (unlucky random seed?)".format(params['log']['dir'])
    os.makedirs(params["log"]["dir"])
    create_sub_dir(params)


def read_json(dir_path, file_name):
    if ".txt" in file_name:
        file_name = file_name
    else:
        file_name = "{0}.json".format(file_name)
    with open(os.path.join(dir_path, file_name), 'r') as f:
        data = json.load(f)
    return data


def write_dict(dict_object, dir_path, file_name):
    with open(os.path.join(dir_path, "{0}.txt".format(file_name)), 'w') as f:
        json.dump(dict_object, f, sort_keys=True, indent=4)
        f.write('\n')


def save_trajectory(dir_episodes, episode, trajectory, flag="play"):
    """Does the memory-consuming process of saving the trajectory.

    Might consider adding a parameter for this. It pickle-dumps `trajectory`
    which must contain all the frames. Yeowch! There is already a parameter for
    the _colored_ output, which we can always set to false for now.

    Parameters
    ----------
    dir_episodes: parent directory to save episode (Path object)
    episode: integer, representing episode index (edit: lifespan)
    trajectory: a `dqn.replay.episode.Episode` object
    flag: either 'train' or 'test', I think, not 'play'
    """

    if not osp.isdir(dir_episodes):
        os.makedirs(dir_episodes)

    lifespan = "episode_{}_{}.pkl".format(flag, str(episode).zfill(7))
    with open(os.path.join(dir_episodes, lifespan), 'wb') as _trajectory_file:
        pickle.dump(trajectory, _trajectory_file)


def read_trajectory(dir_episodes, episode, flag="train"):
    lifespan = "episode_{}_{}.pkl".format(flag, str(episode).zfill(7))
    _episode_path = os.path.join(dir_episodes, lifespan)
    if os.path.exists(_episode_path):
        with open(_episode_path, 'rb') as f:
            _trajectory = pickle.load(f)
        return _trajectory
    else:
        return None


# def read_snapshot(model_dir, number_0_idx, gpu, gpu_id):
#     """Read snapshot from zero indexed number.
#     """
#     number = number_0_idx + 1
#     _snapshot_loc = os.path.join(model_dir, "snapshots",
#                                  "snapshot_{0}.pth.tar".format(str(number).zfill(4)))
#     assert os.path.exists(_snapshot_loc), _snapshot_loc
#     assert isinstance(gpu, bool)
#     if gpu:
#         _snapshot = torch.load(
#             _snapshot_loc,
#             map_location=lambda storage, loc: storage.cuda(gpu_id))
#     else:
#         _snapshot = torch.load(_snapshot_loc, map_location="cpu")
#     return _snapshot


def read_episode(model_dir, number_0_idx):
    """Read a LIFESPAN (not episode despite name) from zero indexed number.
    """
    number = number_0_idx + 1
    lifespan = "episode_train_{}.pkl".format(str(number).zfill(7))
    _episode_loc = os.path.join(model_dir, "episodes", lifespan)
    assert os.path.exists(_episode_loc), _episode_loc
    return read_pickle(filename=_episode_loc)

def load_results(dir):
    import pandas
    monitor_files = (
            glob(osp.join(dir, "*monitor.json")) +
            glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise NotImplementedError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                if not firstline:
                    continue
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df



def load_demonstrations(root, plot = False):
    '''
    Loads all demonstrations, and saves them into two directories:
     "return_ordered_demononstrations" and "player_ordered_demonstrations". 

    This code is gross but works for now
    '''

    episode_ctr = 0 # counter for giving episodes unqiue IDs when saving

    demonstrations = list()
    rewards = list()


    for demonstrator_id in os.listdir(root):
        demonstrator_path =  os.path.join(root, demonstrator_id )
        demonstrator_path = os.path.join(demonstrator_path, "demonstrations")

        for demonstration_id in os.listdir(demonstrator_path):
            demonstration_path = os.path.join(demonstrator_path,  demonstration_id)
            demonstration_path = os.path.join(demonstration_path,   "episodes")
            episode_num = 1

            for episode_dir in  os.listdir(demonstration_path): 
                episode_path = os.path.join(demonstration_path, episode_dir)

                self_rating = None
                rate_file = os.path.join(episode_path ,"rate.txt")
                with open(rate_file, "r")as f:
                    self_rating = int(f.read())

                for filename in os.listdir(episode_path):

                    if filename.endswith(".pkl"): # filter out rating files
                        traj_file = os.path.join(episode_path, filename)


                        with open(traj_file, 'rb') as f:
                            traj = pickle.load(f)# trajectory is an Episode object

                        # first save in return_ordered_demononstrations in a subdirectory for the return. 
                        traj_return =  traj.episode_total_reward 

                        demonstrations.append(traj)
                        rewards.append(traj_return)

                        episode_ctr += 1 #increment index


    return rewards, demonstrations 