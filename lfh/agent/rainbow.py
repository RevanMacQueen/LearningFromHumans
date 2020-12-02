import torch
import os
import copy
import numpy as np
from lfh.utils.variables import create_var, create_batch

from lfh.agent.dqn import DQNTrainAgent
from lfh.utils.config import Configurations
from lfh.models import RainbowDQN as DQN

def replace_with_rbw_cfg(params: Configurations):
    """
    Replaces the params.params parameters with the rbw_config parameters.
    :return:
    """
    assert params.rbw_config is not None
    params.params["env"]["max_num_steps"] = params.rbw_config["T_max"]
    params.params["replay"]["initial"] = params.rbw_config["learn_start"]
    params.params["replay"]["size"] = params.rbw_config["memory_capacity"]
    params.params["train"]["train_freq_per_step"] = params.rbw_config["replay_frequency"]
    params.params["train"]["num_steps"] = params.rbw_config["multi_step"]
    params.params["train"]["hidden_size"] = params.rbw_config["hidden_size"]
    params.params["opt"]["name"] = "Adam"
    params.params["opt"]["params"]["lr"] = params.rbw_config["learning_rate"]
    params.params["opt"]["params"]["eps"] = params.rbw_config["adam_eps"]
    params.params["env"]["frame_stack"] = params.rbw_config["history_length"]
    params.params["train"]["gamma"] = params.rbw_config["discount"]
    params.params["train"]["target_sync_per_step"] = params.rbw_config["target_update"]
    params.params["log"]["snapshot_per_step"] = params.rbw_config["snapshot_per_step"]
    params.params["log"]["snapshot_min_step"] = params.rbw_config["snapshot_min_step"]

    # we remove epsilon here.
    params.params["epsilon"]["start"] = 0.
    params.params["epsilon"]["mid"] = 0.
    params.params["epsilon"]["end"] = 0.

    return params.params


class RainbowAgent(DQNTrainAgent):
    def __init__(self, args, net, action_space, opt, train_params, replay,
                 gpu_params, log_params, policy=None, teacher=None,
                 avg_window=100, tag=""):
        self.action_space = action_space
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=gpu_params['id'])  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip

        self.online_net = net
        if hasattr(args, 'model'):  # Load pretrained model if provided
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        super(RainbowAgent, self).__init__(net=self.online_net, gpu_params=gpu_params,
                                           log_params=log_params, opt=opt, train_params=train_params,
                                           replay=replay, policy=policy, teacher=teacher, avg_window=avg_window,
                                           tag=tag)
        self._target = None
        # self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.target_net = copy.deepcopy(net).to(device=gpu_params['id'])
        self.sync_target(1)
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def __call__(self, states, steps=None):
        obs = self._states_preprocessor(np.expand_dims(states, 0))
        return self._policy(self._qs_postprocessor(self.evaluate_qs(obs)), steps=steps)

    def loss(self, steps):
        # (1) Sample from buffer.
        transitions, learner_bs = self.sample_transitions(steps=steps)
        apply_extra_losses = (learner_bs != self._train_params['batch_size'])

        # (2) Create minibatch via PyTorch tensors.
        states, next_states, actions, rewards, dones = create_batch(
            transitions=transitions,
            gpu=self._gpu_params["enabled"],
            gpu_id=self._gpu_params["id"],
            gpu_async=self._gpu_params["async"],
            requires_grad=False)

        weights = self.get_weights(states, actions, transitions.weight)

        nonterminals = 1 - dones

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = rewards.unsqueeze(1) + nonterminals.unsqueeze(-1) * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms).float()
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        self._add_new_loss(loss, "bellman_loss")
        self._add_new_loss(loss[:learner_bs], "bellman_from_student")
        if apply_extra_losses:
            self._add_new_loss(loss[learner_bs:], "bellman_from_teacher")

        return loss, weights

    def opt_step(self, steps, loss, weights):

        opt = self._opt.get_opt(steps)
        opt.zero_grad()
        loss.mean().backward()

        if self._opt.clipping != 0:
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), self._opt.clipping)
        opt.step()

    def sync_target(self, steps):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    def evaluate_qs(self, state):
        with torch.no_grad():
            return (self.online_net(state) * self.support).sum(2)

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state) * self.support).sum(2).max(1)[0].item()

    def set_train(self):
        self.online_net.train()

    def set_eval(self):
        self.online_net.eval()
