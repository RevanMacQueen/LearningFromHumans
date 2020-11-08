import os
import argparse
import numpy as np
import time
import logging
from pathlib import Path

# from lfh.envs.atari import make_env
from lfh.envs.wrapper import Environment
from lfh.replay.base import ExperienceReplay
from lfh.replay.transition import Transition
from lfh.utils.config import Configurations
from definitions import ROOT_DIR

# Adapted from https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

start_episode = True


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, start_episode

    # print("ere")
    # if start_episode:
    #     print("hereh")
    #     start_episode = False
    a = int(key)
    if key==0xff0d: human_wants_restart = True
    if a==112: human_sets_pause = not human_sets_pause

    if a == ord('a'):
        human_agent_action = 3
    elif a == ord('d'):
        human_agent_action = 2
    else:
        return

def key_release(key, mod):
    global human_agent_action
    #a = int( key - ord('0') )
    a = int(key)
    if a == ord('a'):
        human_agent_action = 0
    elif a == ord('d'):
        human_agent_action = 0
    else:
        return

def rollout(env, practise =False):
    global human_agent_action, human_wants_restart, human_sets_pause, start_episode
    human_wants_restart = False
    next_obs = env.reset()

    skip = 0
    total_reward = 0
    total_timesteps = 0

    num_lives = 5
    
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

        #if r != 0:
        #    print("reward %0.3f" % r)
        total_reward += r
        # window_still_open = env.env.unwrapped.render()
        window_still_open = env.env.window_open

        if not window_still_open: return False

        if env.env_done:
        # if not practise:
                #env.save_trajectory()
        
            #  env.reset()
            env.finish_episode(save=not practise)
            num_lives -= 1
            if num_lives == 0:
                break
                env.close()
            #
            #env.finish_episode(save=True)
            # replay.add_episode(env.trajectory)
            # 

        if human_wants_restart: break

        while human_sets_pause:
            # env.render()
            window_still_open = env.env.unwrapped.render()
            time.sleep(0.1)

        
        time.sleep(0.06)

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



    print(params.params['env'])
    wrapper_env = Environment(params.params['env'], params.params['log'],
                              logger=logger,
                              seed=params.exp_config['seed'],
                              render_every_frame=True)
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

    # while True:
    #     input("Hello :) Welcome to the ZPD from human demonstrations pilot study! [Press Enter]\n")
    #     input("You will be playing Atari Breakout to help teach a reinforcement learning based agent how to play [Press Enter]\n")
    #     input("You can play as many games as you would like. After each game you will be asked via the terminal whether you would like to play again [Press Enter]\n")

    #     print("The goal of the game is to break all the blocks at the top of the display, in order to get as many points as possible.")
    #     input("You control a paddle at the bottom of the screen, the controls are: [Press Enter]\n")
    #     print("\t 'a' to move left")
    #     print("\t 'd' to move right")

    #     print()
    #     input("You can additionally press 'p' to pause the game [Press Enter]\n")

    #     a = input("Enter'n' to advance to the game. Enter any other key to hear these instructions again. [Press key then press enter]\n")
    #     if a == 'n':
    #         break 

    # input("You will be allowed 2 practise rounds, which will not be saved. All other games will be saved and used for learning [Press Enter to begin practise round 1]\n")
    
    
    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                        # can test what skip is still usable.

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    rollout(wrapper_env, practise=True)

    input("Practise round 1 complete. [Press Enter to begin practise round 2]\n")


    while 1:
        wrapper_env.reset()

        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        rollout(wrapper_env, practise=True)

        contin = input("Play again? [Press 'y' and enter for yes, 'n' and enter for no")

        if contin == 'n':
            break

