'''
Main file for training agent with demonstrations
'''

import os
import argparse
import logging
import time
from tqdm import tqdm
from lfh.utils.config import Configurations
from lfh.utils.setup import cuda_config, set_all_seeds
from lfh.utils.io import write_dict, load_demonstrations
from lfh.utils.train import init_atari_model
from lfh.utils.logger import setup_logger
from lfh.environment.atari_wrappers import make_env
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from lfh.replay.experience import ExperienceReplay, ExperienceSource, ZPDExperienceReplay,UnsequencedExperienceReplay
from lfh.optimizer import Optimizer
from lfh.agent.dqn import DQNTrainAgent
from lfh.environment.setup import Environment
from lfh.policy import GreedyEpsilonPolicy
import cProfile
from pprint import pformat
import numpy as np
from lfh.processes import play, test
from pathlib import Path

def main_profiler(params):
    """
    A similar 'roundabout' procedure is done in training when calling the play
    and test methods: it is actually called through the cProfile profiler.
    """
    pth = os.path.join(params.params["log"]["dir"], 'main.prof')
    cProfile.runctx('main(params)', globals(), locals(), pth)


def get_true_rew(monitor_dir):
    true_results = monitor.load_results(monitor_dir)
    true_rews = true_results['r'].tolist()
    return true_rews


def main(params):
    # Remap the gpu devices if using gpu
    if cuda_config(gpu=params.params["gpu"]["enabled"],
                   gpu_id=params.params["gpu"]["id"]):
        params.params["gpu"]["id"] = 0


    # Include log directory names in the params
    setup_logger(dir_path=params.params["log"]["dir"],
                 filename="root",
                 level=params.params["log"]["log_level"])
    logger = logging.getLogger("root")
    logger.info("Output directory is located at {0}."
                .format(params.params["log"]["dir"]))

    # Set env to add stuff to `params['env']` -- we do NOT use this `_env` for
    # stepping; we create again in `dqn/processes.py` via `Environment` class.
    _env = make_env(
        env_name=params.params["env"]["name"],
        episode_life=params.params["env"]["episode_life"],
        skip_frame=params.params["env"]["frame_skip"],
        clip_rewards=params.params["env"]["clip_rewards"],
        frame_stack=params.params["env"]["frame_stack"])
       # logdir=params.params["log"]["dir"])

    params.params["env"]["num_actions"] = _env.action_space.n
    params.params["env"]["action_meanings"] = \
        _env.unwrapped.get_action_meanings()
    params.params["env"]["obs_space"] = _env.observation_space.shape
    params.params["env"]["total_lives"] = _env.unwrapped.ale.lives()

    # Output all configurations
    params = params.dump()
    set_all_seeds(seed=params["seed"], gpu=params["gpu"]["enabled"])

    # Initialize dqn Q-network
    model = init_atari_model(
        obs_space=params["env"]["obs_space"],
        num_actions=params["env"]["num_actions"],
        hidden_size=params["train"]["hidden_size"],
        gpu=params["gpu"]["enabled"],
        gpu_id=params["gpu"]["id"])
    model.init_weight()
    model.share_memory()

    if params["agent"] == "uniform_zpd":
        replay_memory = ZPDExperienceReplay(
            capacity=params["replay"]["size"],
            capacity_dem=1e6, # uncessesarily large. we will never actually reach this capacity  
            init_cap=params["replay"]["initial"],
            init_cap_dem=1, # this doesn't matter for now. Maybe in the future we can use this
            frame_stack=params["env"]["frame_stack"], 
            gamma=params["train"]["gamma"],
            tag="train",
            root= params["zpd"]["demonstrations_dir"], 
            offset=params["zpd"]["offset"], 
            radius=params["zpd"]["radius"], 
            mix_ratio=params["zpd"]["mix_ratio"],
            batch_size=params["train"]["batch_size"])
    elif params["agent"] == "DDQN":
        replay_memory = ExperienceReplay(
            capacity=params["replay"]["size"],
            init_cap=params["replay"]["initial"],
            frame_stack=params["env"]["frame_stack"],
            gamma=params["train"]["gamma"], 
            tag="train")     
    elif params["agent"] == "unseq_DDQN":
        replay_memory = UnsequencedExperienceReplay(
            capacity=params["replay"]["size"],
            capacity_dem=1e6, # uncessesarily large. we will never actually reach this capacity  
            init_cap=params["replay"]["initial"],
            init_cap_dem=1, # this doesn't matter for now. Maybe in the future we can use this
            frame_stack=params["env"]["frame_stack"], 
            gamma=params["train"]["gamma"],
            tag="train",
            root= params["zpd"]["demonstrations_dir"],
            mix_ratio=params["zpd"]["mix_ratio"])
    else:
        raise NotImplementedError

    # Initialized optimizer with decaying `lr_schedule` like OpenAI does.
    opt = Optimizer(net=model, opt_params=params["opt"],
                    max_num_steps=params["env"]["max_num_steps"],
                    train_freq_per_step=params["train"]["train_freq_per_step"])

    # Initialize dqn agent. Like I do, it uses a schedule.
    policy = GreedyEpsilonPolicy(
        params=params["epsilon"], num_actions=params["env"]["num_actions"],
        max_num_steps=params["env"]["max_num_steps"],
        train_freq_per_step=params["train"]["train_freq_per_step"])

    agent = DQNTrainAgent(net=model, gpu_params=params['gpu'],
                          log_params=params["log"],
                          opt=opt, train_params=params["train"],
                          replay=replay_memory, policy=policy, teacher=None,
                          avg_window=params["env"]["avg_window"])

    # Finally, training. See `global_settings` and other experiment files.
    snapshots_summary = {}
    info_summary = {}
    _play_rewards = []  # clipped rewards
    _play_speeds = []
    _test_rewards = []  # clipped rewards
    _test_speeds = []
    steps = 0
    _end = False
    num_true_episodes = 0
    time_start = time.time()
    avg_w = params["env"]["avg_window"]
    max_steps = params["env"]["max_num_steps"]

    # Initialize environment 
    train_env = Environment(env_params=params["env"], log_params=params["log"],
                            train=True, logger=logger, seed=params["seed"],)

    exp_source = ExperienceSource(env=train_env, agent=agent,
                                  episode_per_epi=params["log"]["episode_per_epi"])
    exp_source_iter = iter(exp_source)
    pbar = tqdm(total=max_steps)

    while not _end:
        for _ in range(params["train"]['train_freq_per_step']): # for number of training iter per step
            # stoppping condition
            if steps >= max_steps:
                _end = True
                break
            steps += 1
            pbar.update(1)
            exp = next(exp_source_iter)    
            rewards, mean_rewards, speed = exp_source.pop_latest()
            
            if rewards is not None: # only happens at end of an episode
                _play_rewards.append(rewards)
                if params["agent"] != "DDQN":
                    # update the average reward 
                    replay_memory.update_avg_reward(rewards) 
            # add transition to RB
            replay_memory.add_one_transition(exp) 

        # Exit early, ignore training, or finish training.
        if _end:
            write_dict(dict_object=snapshots_summary,
                       dir_path=params["log"]["dir_snapshots"],
                       file_name="snapshots_summary")
            break
        
        
        if len(replay_memory) < params["replay"]['initial']: # to make sure there are enough steps in the replay buffer
            continue

        # Weights should be synced among this and train/test processes.
        # Called at every multiple of four, so steps = {0, 4, 8, 12, ...}.
        agent.train(steps=steps)
        
    logger.info("Training complete!")
    np.save("returns", _play_rewards)

    
if __name__ == '__main__':
    mp = mp.get_context('spawn')
    # mp.set_start_method('spawn')

    # Read configurations from file
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="breakout_standard", help="ID of the experiment to run")
    parser.add_argument("--exp-name", type=str, default='settings',
                        help='directory from which to load experiment settings')
    parser.add_argument("--profile", action='store_true', help='whether to run the profiler')
    parser.add_argument("--demonstrations-dir", type=str, default='../Demonstrations/BreakoutDemonstrations/', help='directory containing demonstrations')

    parser.add_argument("--agent", type=str, default="uniform_zpd", help="what agent to run 1. uniform_zpd 2. DDQN 3. unseq_DDQN")
    parser.add_argument("--offset", type=int, default=1, help="offset for zpd window")
    parser.add_argument("--radius", type=int, default=2, help="radius of zpd window")
    parser.add_argument("--mix-ratio", type=float, default=1/4, help="ratio of demonstrations in the replay buffer")
    parser.add_argument("--seed", type=int, help="random seed")

    params = parser.parse_args()
    _params = Configurations(params, note="")

    # Include log directory names in the param

    main(_params)
