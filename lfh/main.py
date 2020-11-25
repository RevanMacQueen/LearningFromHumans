'''
Main file for training agent with demonstrations
'''

import os
import argparse
import logging
import time
from lfh.utils.config import Configurations
from lfh.utils.setup import cuda_config, set_all_seeds
from lfh.utils.io import write_dict, load_demonstrations
from lfh.utils.train import init_atari_model
from lfh.utils.logger import setup_logger
from lfh.environment.atari_wrappers import make_env
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from lfh.replay.experience import ExperienceReplay, ExperienceSource
from lfh.optimizer import Optimizer
from lfh.agent.dqn import DQNTrainAgent
from lfh.environment.setup import Environment
# from lfh.teacher.teacher_centers import MultiTeacherTeachingCenter
# from lfh.utils.debug import generate_debug_msg
from lfh.policy import GreedyEpsilonPolicy
import cProfile
from pprint import pformat
import numpy as np
from lfh.processes import play, test
# from lfh.utils.heuristics import perform_bad_exit_early
# from lfh.environment import monitor
from pathlib import Path
# from lfh.envs.atari import make_env

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="breakout_standard", help="ID of the experiment to run")
    parser.add_argument("--exp-name", type=str, default='settings',
                        help='directory from which to load experiment settings')
    parser.add_argument("--profile", action='store_true', help='whether to run the profiler')
    parser.add_argument("--demonstrations-dir", type=str, default='../Demonstrations/BreakoutDemonstrations/', help='directory containing demonstrations')

    return vars(parser.parse_args())



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

    #load_demonstrations(args["demonstrations_dir"])
        
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

    # Initialize TensorBoard writer for visualization purposes
    # writer = SummaryWriter(log_dir=params["log"]["dir"],
    #                        comment="-" + params["env"]['name'])

    # Initialized replay buffer. Not quite like OpenAI, seems to operate at
    # another 'granularity'; that of episodes, in addition to transitions?
    replay_memory = ExperienceReplay(
        #writer=writer,
        capacity=params["replay"]["size"],
        init_cap=params["replay"]["initial"],
        frame_stack=params["env"]["frame_stack"],
        gamma=params["train"]["gamma"], tag="train",
        debug_dir=os.path.join(params["log"]["dir"], "learner_replay"))

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
    snapshot_number = 1
    _play_rewards = []  # clipped rewards
    _play_speeds = []
    _test_rewards = []  # clipped rewards
    _test_speeds = []
    steps = 0
    _end = False
    num_true_episodes = 0
    time_start = time.time()
    avg_w = params["env"]["avg_window"]
    matched_list = None
    test_match = False

    # exp_queue = mp.Queue(maxsize=params["train"]['train_freq_per_step'] * 2)

    # Initialize environment 
    train_env = Environment(env_params=params["env"], log_params=params["log"],
                            train=True, logger=logger, seed=params["seed"],)

    exp_source = ExperienceSource(env=train_env, agent=agent,
                                  episode_per_epi=params["log"]["episode_per_epi"], max_episodes=10000)
    exp_source_iter = iter(exp_source)
    # exp_queue.put(None)

    while not _end:
        for _ in range(params["train"]['train_freq_per_step']): # for number of training iter per step
        
            steps += 1
            try:
                exp = next(exp_source_iter)    
            except:
                pass
    

            rewards, mean_rewards, speed = exp_source.pop_latest()
            
            if rewards is not None: # only happens at end of an episode
                _play_rewards.append(rewards)
           
            if exp is None:
                _end = True
                break
   
            replay_memory.add_one_transition(exp) # add transition to ERB

        # Update time
        # Exit early, ignore training, or finish training.
        # if perform_bad_exit_early(params, steps, _play_rewards):
        #     break
        if len(replay_memory) < params["replay"]['initial']: # to make sure there are enough steps in the replay buffer
            continue
        if _end:
          
            write_dict(dict_object=snapshots_summary,
                       dir_path=params["log"]["dir_snapshots"],
                       file_name="snapshots_summary")
            break

        # For teacher-only, if matched to something. Save snapshot. We can use
        # to check if saved snapshots 'match' the teacher snapshot.  The other
        # snapshot saving is for later with training a single teacher.

        # this might be where we add stuff for training with demonstrations

        # Weights should be synced among this and train/test processes.
        # Called at every multiple of four, so steps = {0,4,8,12,...}.
        agent.train(steps=steps)

        # Output stuff
        # if steps % params["log"]['snapshot_per_step'] < \
        #         params["train"]['train_freq_per_step'] and \
        #         steps > params['log']['snapshot_min_step']:
        #     rew_list = get_true_rew(params["log"]["dir"])
        #     current_true_rew = np.mean(rew_list[-avg_w:])
        #     current_clip_rew = np.mean(_play_rewards[-avg_w:])
        #     agent.save_model(snapshot_number)
        #     snapshots_summary[snapshot_number] = {
        #         "clip_rew_life": current_clip_rew,
        #         "true_rew_epis": current_true_rew,
        #         "num_finished_epis": len(rew_list),
        #         "steps": steps,
        #     }
        #     snapshot_number += 1
        #     write_dict(dict_object=snapshots_summary,
        #                dir_path=params["log"]["dir_snapshots"],
        #                file_name="snapshots_summary")
        
    logger.info("Training complete!")

    np.save("returns", _play_rewards)

    
if __name__ == '__main__':
    mp = mp.get_context('spawn')
    # mp.set_start_method('spawn')

    # Read configurations from file
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=str, help="ID of the experiment to run")
    parser.add_argument("--exp-name", type=str, default='settings',
                        help='directory from which to load experiment settings')
    parser.add_argument("--profile", action='store_true', help='whether to run the profiler')
    params = parser.parse_args()
    _params = Configurations(params, note="")

    # Include log directory names in the params
    if params.profile:
        main_profiler(_params)
    else:
        main(_params)
