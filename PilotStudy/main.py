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
human_sets_pause = True

start_episode = True


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, start_episode

    a = int(key)
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

def rollout(env):
    env.env.render()
    env.env.unwrapped.viewer.window.on_key_press = key_press
    env.env.unwrapped.viewer.window.on_key_release = key_release

    global human_agent_action, human_wants_restart, human_sets_pause, start_episode
    human_wants_restart = False
    next_obs = env.reset()

    skip = 0
    total_reward = 0
    total_timesteps = 0

    num_lives = 5
    
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        env.step(a)
        r = env.env_rew

        total_reward += r
        window_still_open = env.env.window_open

        if not window_still_open: return False

        if env.env_done:
            env.finish_episode(save = True)
            num_lives -= 1
            if num_lives == 0:
                break
                env.close()

        if human_wants_restart: break

        while human_sets_pause:
            window_still_open = env.env.unwrapped.render()
            time.sleep(0.1)

        
        time.sleep(0.08)

    env.increment_game_number()
    human_sets_pause = True
    #print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    return total_reward

if __name__ == "__main__":
    # Read configurations from file
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="breakout_standard", help="ID of the experiment to run")
    parser.add_argument("--exp-name", type=str, default='settings',
                        help='directory from which to load experiment settings')
    parser.add_argument("--profile", action='store_true', help='whether to run the profiler')



    while True:
        input("Hello! Welcome to the ZPD from human demonstrations pilot study! Thanks for taking the time to take part. [Press Enter]\n")
        input("You will be playing Atari Breakout to help teach a reinforcement learning agent how to play. [Press Enter]\n")
        input("You can play as many games as you would like. We ask you to play at least 5 games, but if you feel like playing more games it would be greatly appreciated! The more games you play the better our agent will learn. [Press Enter]\n")
        input("Since your playing will be teaching the agent, please play to the best of your ability. [Press Enter]\n")
        input("After each game you will be asked via the terminal to rate how well you played and whether you would like to play again. We will use the rating information to decide in which order to give demonstrations to our RL agent. [Press Enter]\n")
        input("You will control a paddle to hit a ball towards the bricks at the top of the game window.  [Press Enter]\n")
        input("The goal of the game is to break all the bricks at the top of the display, in order to get as many points as possible. [Press Enter]\n")
        input("If you're unfamiliar with this game, we recommend you watch this video to see the game in action: https://www.youtube.com/watch?v=V1eYniJ0Rnk [Press Enter]\n" )

        print("You control a paddle at the bottom of the screen, the controls are:\n")
        print("\t 'a' to move left")
        print("\t 'd' to move right\n")
        print("\t 'p' to pause the game"  )

       
        input("The game will start in pause mode, press 'p' to start the game once the window is open. [Press Enter]\n")
        input("The window may initally appear small, once the game is paused you can adjust the size of the window. [Press Enter]\n")
        input("Once a new game starts, you may need to click on the window to allow it to read your key presses. [Press Enter]\n")
        
    
        a = input("Enter 'n' to advance to the game. Enter any other key to hear these instructions again. [Press key then press enter]\n>>>")
        if a == 'n':
            break 
    
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

    replay_memory = ExperienceReplay(
        capacity=params.params["replay"]["size"],
        init_cap=params.params["replay"]["initial"],
        frame_stack=params.params["env"]["frame_stack"],
        gamma=params.params["train"]["gamma"], tag="train",
        debug_dir=os.path.join(params.params["log"]["dir"], "learner_replay"))

    
    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                        # can test what skip is still usable.

    games_completed = 0
    total_games = 5

    while 1:
        wrapper_env.reset()
        total_reward = rollout(wrapper_env)

        print("Game complete. Your score was: %d" % total_reward)

        entered = False
        while not entered:

            rate = input("How well do you think you played on that game? [Enter a number 1-5 (1 low, 5 high)] \n>>>")
            try:
                rate = int(rate)
                if rate >= 1 and rate <= 5:
                    break
                else:
                    print("Invalid rating, please enter a rating from 1-5\n")
            except:
                print("Not a number\n")
                        
        rate_file = Path(wrapper_env.log_params["dir_episodes"])/str(wrapper_env.game_number - 1) / "rate.txt"

        with open(rate_file, "w+") as f:
            f.write(str(rate))

        games_completed += 1

        if games_completed >= total_games:
            contin = input("Do you want to play again? [Press 'y' and enter for yes, 'n' and enter for no]\n>>> ")

            if contin == 'n':
                break
        else:
            input("%d/%d games completed. Please press enter, and then press 'p' on the Atari UI to start the next game.\n" % (games_completed, total_games))
