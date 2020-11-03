import os
import argparse
import numpy as np
import time
import logging
from pathlib import Path

# from lfh.envs.atari import make_env
from lfh.envs.wrapper import Environment
# from lfh.replay.base import ExperienceReplay
# from lfh.replay.transition import Transition
from lfh.utils.config import Configurations
from definitions import ROOT_DIR

# Adapted from https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    next_obs = env.reset()
    env.reset_episode()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        # obs = next_obs
        # next_obs, r, done, info = env.step(a)
        env.step(a)
        r = env.env_rew

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.env.unwrapped.render()

        if not window_still_open: return False

        if env.env_done:
            env.save_trajectory()
            # replay.add_episode(env.trajectory)
            break

        if human_wants_restart: break

        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


if __name__ == "__main__":
    # Read configurations from file
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="breakout_standard", help="ID of the experiment to run")
    parser.add_argument("--exp-name", type=str, default='settings',
                        help='directory from which to load experiment settings')
    parser.add_argument("--profile", action='store_true', help='whether to run the profiler')
    params = parser.parse_args()

    logger = logging.getLogger("train_env")

    params.exp_name = Path(ROOT_DIR, "lfh", "experiments", params.exp_name)
    params = Configurations(params, "")
    # env = make_env(
    #             env_name=params.params["env"]["name"],
    #             episode_life=params.params["env"]["episode_life"],
    #             skip_frame=params.params["env"]["frame_skip"],
    #             clip_rewards=params.params["env"]["clip_rewards"],
    #             frame_stack=params.params["env"]["frame_stack"],
    #             )
    wrapper_env = Environment(params.params['env'], params.params['log'], logger=logger, seed=params.exp_config['seed'])
    env = wrapper_env.env
    params.params["env"]["num_actions"] = env.action_space.n
    params.params["env"]["action_meanings"] = \
        env.unwrapped.get_action_meanings()
    params.params["env"]["obs_space"] = env.observation_space.shape
    params.params["env"]["total_lives"] = env.unwrapped.ale.lives()

    # Initialized replay buffer. Not quite like OpenAI, seems to operate at
    # another 'granularity'; that of episodes, in addition to transitions?

    # replay_memory = ExperienceReplay(
    #     capacity=params.params["replay"]["size"],
    #     init_cap=params.params["replay"]["initial"],
    #     frame_stack=params.params["env"]["frame_stack"],
    #     gamma=params.params["train"]["gamma"], tag="train",
    #     debug_dir=os.path.join(params.params["log"]["dir"], "learner_replay"))

    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                        # can test what skip is still usable.


    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(ACTIONS))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

    while 1:
        window_still_open = rollout(wrapper_env)
        if not window_still_open: break