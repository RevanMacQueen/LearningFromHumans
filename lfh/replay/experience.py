import os
import logging
import random
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

from lfh.utils.replay import merge_transitions_xp
from lfh.replay.episode import Episode
from lfh.replay.transition import Transition
from lfh.agent.agent import Agent
from lfh.environment.setup import Environment
from lfh.utils.io import load_demonstrations

class ExperienceReplay(object):
    def __init__(self, capacity, init_cap, frame_stack, gamma, tag,
                 debug_dir=None):
        """
        A regular experience replay buffer. The buffer is maintained by
        holding `Episode` object for each episode. When the size of the
        replay buffer is met, it sets the first episode to `None`.

        :param capacity: The total amount of transitions to hold in the buffer
        :param init_cap: The initial amount of transitions in the buffer in
        order to call `sample` function
        :param frame_stack: Number of frames to stack in the samples
        :param gamma: Discount rate
        :param tag: tag for logging purposes
        :param debug_dir: an output directory for storing debugging output
        """
        self._buffer = []
        self.capacity = capacity
        self._debug_dir = debug_dir
        self._tag = tag
        # Total number of frames covered by all active episodes in the memory
        self.total_active_trans = 0
        # Table storing the mapping of replay index to a tuple of
        # (episode_number, frame_number)
        self._frame_lookup = {}
        # Table storing the mapping of episode number to the index in `_buffer`
        self._episode_lookup = {}
        self._current_pos = 0
        self._init_cap = init_cap
        self._frame_stack = frame_stack
        self._gamma = gamma
        # The most recent episode
        self._current_episode = None
        self._first_active_idx = 0
        self.info = {}
        self._debug_output = {}
        # Number of frames in the last episode not included in sampling
        self.logger = logging.getLogger(tag + "_replay")
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        self.logger.debug("Replay buffer with capacity {0} has been "
                          "initialized".format(capacity))

    @property
    def buffer(self):
        return self._buffer

    @property
    def total_episodes(self):
        """All episodes that have EVER been added to the replay buffer.

        Including episodes that are evicted, which are assigned to `None` in
        this list, in the order they are evicted, so this list always starts
        with some number of `None` items (including potentially none, which
        happens before we over-ride any samples).
        """
        return len(self._buffer)

    @property
    def total_active_episodes(self):
        """
        :return only the active episodes that have ever been added to the
        replay buffer.
        """
        return self.total_episodes - self._first_active_idx

    def __len__(self):
        """
        :return: the true replay buffer size. The replay buffer contains
        episodes or transitions more than its capacity, but it only for explicit
        storage purposes. The `__len__` function outputs the true size,
        which is always bounded by the capacity.
        """
        return len(self._frame_lookup)

    def _convert_episode_idx(self, episode_num):
        assert episode_num in self._episode_lookup
        return self._buffer[self._episode_lookup[episode_num]]

    def _get_transition(self, i, num_steps, force=False):
        """Find a sample inside the replay buffer.

        :param i: the index of the transition in the replay buffer lookup
        table
        :param num_steps: Number of steps to look forward
        :param force: if `force=True`, we always return the last frame even
        if episode frames are not enough
        :return: A `Transition` object
        """
        assert 0 <= i < self.__len__()
        _episode_num, _frame_num = self._frame_lookup[i]
        assert _episode_num in self._episode_lookup
        assert self._episode_lookup[_episode_num] >= self._first_active_idx





        return self._convert_episode_idx(_episode_num).sample_transition(
            idx=_frame_num, frame_stack=self._frame_stack,
            num_steps=num_steps, gamma=self._gamma, force=force)

    def get_transitions(self, idx, num_steps, force=False):
        """
        Find multiple samples in the replay buffer and output a numpy array
        stacking all samples together.

        :param idx: An iterable that contains indexes of transitions
        :param num_steps: Number of steps to look forward
        :param force: if `force=True`, we always return the last frame even
        if episode frames are not enough
        :return: A `Transition` object, where each field is a `numpy` array
        stacked all requested samples
        """
        assert num_steps >= 1
        _transitions = []
        for i in idx:
            _transitions.append(self._get_transition(i, num_steps, force))
        return merge_transitions_xp(_transitions)


    def get_transitions_list(self, idx, num_steps, force=False):
        """
        Like get_transitions, but do not stack, just return list of transitions

        :param idx: An iterable that contains indexes of transitions
        :param num_steps: Number of steps to look forward
        :param force: if `force=True`, we always return the last frame even
        if episode frames are not enough
        :return: list of  `Transition` objects
        """
        assert num_steps >= 1
        _transitions = []
        for i in idx:
            _transitions.append(self._get_transition(i, num_steps, force))
        return _transitions

    def sample_one(self, num_steps, output_idx=False):
        if self.__len__() < self._init_cap:
            return None
        idx = self._sample_idx(1)
        if output_idx:
            return self._get_transition(i=idx[0], num_steps=num_steps), idx[0]
        else:
            return self._get_transition(i=idx[0], num_steps=num_steps)

    def sample(self, batch_size, num_steps, output_idx=False, no_stack = False):
        """Called from `agent.dqn.sample_transitions()`.

        Also from `SnapshotsTeacher.get_teacher_samples()` for teachers.

        Sample in the replay buffer. If not enough transitions (true size less
        than `init_cap`, return `None`, but `atari.py` guards against this.

        :param batch_size: the size of the batch to be sampled
        :param num_steps: A list of number of steps to look forward
        :param output_idx: A boolean controlling if index also outputs
        along with the samples
        :return: A numpy array of `Transition` object, with index if
        `output_idx` is True
        """
        if self.__len__() < self._init_cap:
            return None
        idx = self._sample_idx(batch_size)

        if no_stack:
            if output_idx:
                return self.get_transitions_list(idx=idx, num_steps=num_steps), idx
            else:
                return self.get_transitions_list(idx=idx, num_steps=num_steps)
        else:
            if output_idx:
                return self.get_transitions(idx=idx, num_steps=num_steps), idx
            else:
                return self.get_transitions(idx=idx, num_steps=num_steps)

    def _sample_idx(self, bs):
        assert self.__len__() >= self._init_cap
        if "p" in self.info:
            idx = np.random.choice(self.__len__(), bs, replace=False, p=self.info["p"])
        else:
            # https://github.com/CannyLab/dqn/issues/28
            #idx = np.random.choice(self.__len__(), bs, replace=False)
            idx = random.sample( range(self.__len__()) , bs)
        return idx

    def add_episode(self, episode):
        """Add a LIFESPAN to replay buffer and evict older ones if necessary.

        :param episode: An `Episode` object
        """
        assert isinstance(episode, Episode)
        self._register_one_episode(next_episode_idx=self.total_episodes, episode=episode)
        self._add_one_episode(episode)
        self._evict_episodes()

    def _register_frame(self, episode_number, frame_number):
        """
        Register a frame in the `_frame_lookup` table for quick selection
        later on. Once a frame is registered, it moves the counter
        `_current_pos` by 1. The frame number idx will increase faster than
        the episode number, since there are many frames per episode.

        :param episode_number: current episode number
        :param frame_number: current frame number
        """
        self._frame_lookup[self._current_pos] = (episode_number, frame_number)
        self._current_pos = (self._current_pos + 1) % self.capacity

    def _register_one_episode(self, next_episode_idx, episode):
        """Register frames for a lifespan.

        :param next_episode_idx: Index in `_buffer` for the current episode
        :param episode: An `Episode` object
        """
        assert isinstance(episode, Episode)
        self._episode_lookup[episode.episode_num] = next_episode_idx
        
        for i in range(episode.length):
            self._register_frame(episode.episode_num, i)


    def _add_one_episode(self, episode):
        """Add episode to the replay buffer and write debug message.

        Called internally and the _shared_ XP replay subclass. Finally
        `self._buffer` contains the memory of the lifespan.

        :param episode: An `Episode` object
        """
        assert isinstance(episode, Episode)
        self._buffer.append(episode)
        self.total_active_trans += episode.length
        # self.logger.debug("Life {0} (1-idx) of length {1} added into replay, "
        #                   "with {2}/{3} active lifespans, {4} total frames and {5} "
        #                   "registered frames.".format(
        #     episode.episode_num, episode.length,
        #     self.total_active_episodes, self.total_episodes,
        #     self.total_active_trans, self.__len__()))

    def _evict_episodes(self):
        """Evict early episodes if capacity is met.

        The capacity is a soft upper bound. We can exceed it, and the while
        condition means evict episodes so long as the outcome _after_ eviction
        means we have at least the capacity. By design, we therefore do not go
        _under_ the capacity after we've filled up the buffer initially.  I
        think this is by design: the `frame_lookup` depends on there being
        something at all indices, and if we were to go below capacity, we would
        have some 'dead/broken' indices (again, after filling buffer).

        'Evicting' here means setting the corresponding buffer item to None.
        """
        while self.total_active_trans - self.capacity >= self._buffer[self._first_active_idx].length:
            _old_episode_num = self._buffer[self._first_active_idx].episode_num
            _old_episode_length = self._buffer[self._first_active_idx].length
            self.total_active_trans -= _old_episode_length
            self._buffer[self._first_active_idx] = None
            self._first_active_idx += 1
            # self.logger.debug(
            #     "Life {0} of length {1} was evicted from replay memory, "
            #     "with {2}/{3} active lifespans, {4} total frames and {5} "
            #     "registered frames.".format(
            #         _old_episode_num, _old_episode_length,
            #         self.total_active_episodes, self.total_episodes,
            #         self.total_active_trans, self.__len__()))

    def evict_all(self):
        """
        Evict all episodes in the buffer
        """
        self._buffer.clear()
        self._first_active_idx = 0
        self.total_active_trans = 0
    
        self._frame_lookup = {}
        self._episode_lookup = {}
        self._current_pos = 0
        self._current_episode = None


    def _init_episode(self, init_obs):
        """
        Maintain the replay buffer by adding transitions. The buffer is
        stored in terms of episodes so that before adding new transitions,
        it is necessary to initialize a new `Episode` to hold experience.

        :param init_obs: Initial observations of the episodes
        """
        assert self._current_episode is None
        assert init_obs.shape[0] == self._frame_stack
        self._current_episode = Episode(
            episode_num=self.total_episodes + 1,  # One-Index!
            init_obs=init_obs)
        self.add_episode(self._current_episode)

    def add_one_transition(self, transition):
        """
        Maintain the replay buffer by adding one transition to the current
        `Episode`. Called during (at least) normal D-DQN training. The states
        are 8-bit integers, for efficient image storage.

        :param transition: A `Transition` object
        """
        assert isinstance(transition, Transition)
        assert transition.next_state.shape[0] == 1
        if self._current_episode is None:
            # ------------------------------------------------------------------
            # From Allen: for the first transition in the episode, the states
            # consists of both current states and next_states. All other
            # transition only needs to contain the new frame from `next_state`.
            # ------------------------------------------------------------------
            assert transition.state.shape[0] == self._frame_stack
            self._init_episode(transition.state)
            transition.state = None
        assert transition.state is None
        self._register_frame(self._current_episode.episode_num,
                             self._current_episode.length)
        self._current_episode.add_transition(transition) # NOTE:  tranistion added to episode? 
        self.total_active_trans += 1
        if transition.done:
            # self.logger.debug("Life {0} finished w/{1} transition "
            #                   "frames. RBuffer contains {2}/{3} active episodes, "
            #                   "{4} total frames and {5} registered frames.".format(
            #     self._current_episode.episode_num,
            #     self._current_episode.length,
            #     self.total_active_episodes, self.total_episodes,
            #     self.total_active_trans, self.__len__()))
            self._current_episode = None

    def save(self, fname: Path) -> None:
        def is_picklable(obj):
            try:
                pickle.dumps(obj)

            except pickle.PicklingError:
                return False
            return True

        with open(fname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)



class Demonstrations(object):
    def __init__(self, returns, demonstrations, offset, radius):
        """
        Helper class for ZPDExperienceReplay. Holds demonstrations and implements 
        f_select function
        """

        self.returns, self.demonstrations = zip(*sorted(zip(returns, demonstrations),  key=lambda x: x[0]))
        self.returns  = np.array(self.returns)
        self.offset = offset
        self.radius = radius

        # maximum reward amongst all demonstrations
        self.max_reward  = self.returns[-1]
        assert self.radius >= 0


    def select(self, r):
        """
        Returns index of demonstration with reward closest to r, plus offset
        
        :param r the average reward of the agent over past 100 episodes
        """

        ind_match = np.argmin(np.abs(self.returns - r))
        ind_center = ind_match + self.offset
        # ensure we are indexing a non-negative index
        ind_center = max(0, ind_center) 
        
        start_ind = ind_center - self.radius
        if start_ind < 0:
            shift_amt = abs(start_ind)
            start_ind = 0 
            end_ind = ind_center + self.radius + shift_amt
        else:
            end_ind = ind_center + self.radius

        # +1 is because python slices are end-exclusive
        return self.demonstrations[start_ind:end_ind+1]


    def get_all_demonstartions(self):
        """
        Returns all demonstrations
        """
        return self.demonstrations 


class ZPDExperienceReplay(object):
    """
    Class that holds the two replay buffers, one for human demonstrations, one for the agents own experience. 
    Samples accoring to ZPD curriculum
    Will behave like ExperienceReplay in dqn.py and main.py (where it is called)
    """

    def __init__(self, capacity, capacity_dem,  init_cap, init_cap_dem, frame_stack, gamma, tag, root, offset, radius, mix_ratio, batch_size):
        # replay for agent 
        self.exp_replay = ExperienceReplay(capacity, init_cap, frame_stack, gamma, tag) 

        # replay for demonstrations 
        self.dem_replay = ExperienceReplay(capacity_dem, init_cap_dem, frame_stack, gamma, tag) 

        # load in demonstrations
        returns, demonstrations = load_demonstrations(root)

        # make sure the lower bound for number of trajectories in a window is sufficiently large
        assert min([d.length for d in demonstrations])*(radius*2+1) >= mix_ratio*batch_size
        assert radius >= 0

        # set up demonstrations object
        self.demonstrations = Demonstrations(returns, demonstrations, offset, radius)

        # the ratio of demonstrations transitions to experience transitions in mini-batches
        self.mix_ratio = mix_ratio

        # average reward of agent (over some number of episodes)
        self.rs = []
        self.avg_r = 0

        # coefficient for linear anneal once learner has surpassed demonstrations
        self.anneal_coeff = 0.95


    def __len__(self):
        return len(self.exp_replay)


    def sample(self, batch_size, num_steps):

        # if learner has surpassed all demonstrations, anneal the number of demonstrations in 
        # a mini-batch towards 0
        if self.avg_r > self.demonstrations.max_reward:
            dem_batch_size = int(batch_size * self.mix_ratio * self.anneal_coeff) 
            self.anneal_coeff *= self.anneal_coeff
        else:
            dem_batch_size = int(batch_size * self.mix_ratio)

        # remainder of mini-batch will be learners experience 
        exp_batch_size = batch_size - dem_batch_size    

        # populate self.dem_replay
        demonstrations = self.demonstrations.select(self.avg_r)

        # make sure we have enough transitions in the buffer
        assert sum([trajectory.length for trajectory in demonstrations])  >= dem_batch_size

        for trajectory in demonstrations:
            self.dem_replay.add_episode(trajectory)
        assert False

        # sample from demonstrations
        dem_samples = self.dem_replay.sample(batch_size=dem_batch_size, num_steps=num_steps, no_stack=True)

        # evict all from dem_replay
        self.dem_replay.evict_all() 
        assert len(self.dem_replay._buffer) == 0

        exp_samples = self.exp_replay.sample(batch_size=exp_batch_size, num_steps=num_steps, no_stack=True)

        return merge_transitions_xp(dem_samples+exp_samples)
        
        
    def add_one_transition(self, transition):
        self.exp_replay.add_one_transition(transition)


    def update_avg_reward(self, new_r):
        if len(self.rs) >= 100:
            self.rs.pop(0)
        self.rs.append(new_r)
        self.avg_r = np.average(self.rs)


class UnsequencedExperienceReplay(object):
    """
    Class that holds the two replay buffers, one for human demonstrations, one for the agents own experience. 
    human demonstrations are sampled without any sequence
    Will behave like ExperienceReplay in dqn.py and main.py (where it is called)
    """

    def __init__(self, capacity, capacity_dem, init_cap, init_cap_dem, frame_stack, gamma, tag, root, mix_ratio):
        # replay for agent 
        self.exp_replay = ExperienceReplay(capacity, init_cap, frame_stack, gamma, tag) 

        # replay for demonstrations 
        self.dem_replay = ExperienceReplay(capacity_dem, init_cap_dem, frame_stack, gamma, tag) 

        # load in demonstrations
        rewards, demonstrations = load_demonstrations(root)
        self.max_reward = sorted(rewards)[-1]
        for trajectory in demonstrations:
            self.dem_replay.add_episode(trajectory)

        # the ratio of demonstrations transitions to experience transitions in mini-batches
        self.mix_ratio = mix_ratio

         # average reward of agent (over some number of episodes)
        self.rs = []
        self.avg_r = 0

        # coefficient for linear anneal once learner has surpassed demonstrations
        self.anneal_coeff = 0.95


    def __len__(self):
        return len(self.exp_replay)

    def sample(self, batch_size, num_steps):

        # if learner has surpassed all demonstrations, anneal the number of demonstrations in 
        # a mini-batch towards 0
        if self.avg_r >  self.max_reward:
            dem_batch_size = int(batch_size * self.mix_ratio * self.anneal_coeff) 
            self.anneal_coeff *= self.anneal_coeff
        else:
            dem_batch_size = int(batch_size * self.mix_ratio)

        # remainder of mini-batch will be learners experience 
        exp_batch_size = batch_size - dem_batch_size    
        
        # sample from demonstrations
        dem_samples = self.dem_replay.sample(batch_size=dem_batch_size, num_steps=num_steps, no_stack=True)
        exp_samples = self.exp_replay.sample(batch_size=exp_batch_size, num_steps=num_steps, no_stack=True)

        return merge_transitions_xp(dem_samples+exp_samples)
        
        
    def add_one_transition(self, transition):
        self.exp_replay.add_one_transition(transition)


    def update_avg_reward(self, new_r):
        if len(self.rs) >= 100:
            self.rs.pop(0)
        self.rs.append(new_r)
        self.avg_r = np.average(self.rs)



class ExperienceSource:
    def __init__(self, env, agent, episode_per_epi):
        """
        `ExperienceSource` is an iterable that integrates environment and agent.
        In train/test processes, we call the `next` for stepping purposes.

        :param env: an `Environment` object, custom class wrapping around gym
            env, adds stuff for saving; see `dqn/environment/setup.py`. Exposes
            a similar `step` interface which calls the true gym env's step.
        :param agent: an `Agent` object
        :param episode_per_epi: Number of episodes to write to disk.
        """
        assert isinstance(agent, Agent)
        assert isinstance(env, Environment)
        self.env = env
        self.agent = agent
        self.episode_per_epi = episode_per_epi
        self.mean_reward = None
        self.latest_speed = None
        self.latest_reward = None

        

    def __iter__(self):
        """THIS is what calls `finish_episode` with `save_tag`.
        """
        while True:
            _obs = self.env.env_obs
            _steps = self.env.env_steps
            action = self.agent(_obs, self.env.total_steps)
            self.env.step(action)  # ENVIRONMENT STEPPING!!!
            _next_obs = np.expand_dims(self.env.env_obs[-1], 0)
            if _steps == 0:
                # new episodes, need `state`
                _obs = _obs
            else:
                # --------------------------------------------------------------
                # Existing episodes, set `state` to None.  XP replay code
                # w/`Transition`s don't check `state` except for when it's the
                # first in episode: see `add_one_transition()` above.
                # --------------------------------------------------------------
                _obs = None
            transition = Transition(state=_obs, next_state=_next_obs,
                                    action=action, reward=self.env.env_rew,
                                    done=self.env.env_done)

            yield transition
            # ------------------------------------------------------------------
            # If env (either train or test) has an episode which just finished,
            # call this to formally finish and potentially save the trajectory.
            # The `episode_per_epi` determines save frequency. Can increase it
            # to decrease memory requirements. But this should only determine
            # saving trajs into pickle files; regardless of what happens the
            # replay _buffer_ should get the frames.
            # ------------------------------------------------------------------
            # Recall yield keyword: we stop at transition above, then next time
            # this is called, we start here to check for if we lost a life.
            # ------------------------------------------------------------------
            if self.env.env_done:
             
                if self.episode_per_epi:
                    save_tag = False #self.env.get_num_lives() % self.episode_per_epi == 0 #Revan: I changed this to never save
                else:
                    save_tag = False
                self.latest_reward = self.env.epi_rew
                self.env.finish_episode(
                    save=save_tag, gif=False,
                    epsilon=self.agent.get_policy_epsilon(self.env.total_steps))
                self.latest_speed = self.env.speed
                self.mean_reward = self.env.mean_rew

               
    def pop_latest(self):
        r = self.latest_reward
        mr = self.mean_reward
        s = self.latest_speed
        if r is not None:
            self.latest_reward = None
            self.mean_reward = None
            self.latest_speed = None
        return r, mr, s
