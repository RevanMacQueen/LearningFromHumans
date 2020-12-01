from lfh.environment.setup import Environment
from lfh.replay.experience import ExperienceSource

def test(agent, env_params, log_params, logger, seed, episodes):
    eval_env = Environment(env_params=env_params, log_params=log_params,
                           train=True, logger=logger, seed=seed)
    agent.set_eval()

    # TODO: I changed ExperienceSource to cap both max steps and max episodes. You might need to change this back.
    exp_source = ExperienceSource(env=eval_env, agent=agent,
                                  episode_per_epi=0, max_steps=1e6, max_episodes=episodes)

    exp_source_iter = iter(exp_source)

    ep = 0
    total_rewards = []
    print(f"Beginning eval from step {seed}")
    _end = False

    while not _end:
        exp = next(exp_source_iter)
        # rewards, mean_rewards, speed = exp_source.pop_latest()

        if exp_source.env.env_done:
            total_rewards.append(exp_source.env.epi_rew)

        if exp is None:
            _end = True
            break

    return total_rewards