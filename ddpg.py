import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import Actor, Critic
from memory import SequentialMemory, StateHistory
from utils import *

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):

        self.nb_states = nb_states * args.window_length
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1,
            'hidden2':args.hidden2,
            'init_w':args.init_w
        }

        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg).to(device)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg).to(device)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        self.criterion = nn.MSELoss()

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.history = StateHistory(args.window_length)
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])
            target_q_batch = to_tensor(reward_batch) + \
                self.discount * to_tensor(terminal_batch) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
        critic_loss = self.criterion(q_batch, target_q_batch)

        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        actor_loss = -self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))])
        actor_loss = actor_loss.mean()

        actor_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.is_training = False
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.is_training = True
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def select_action(self, s_t, advice=None, use_advice=False, decay_epsilon=True):
        self.history.append(s_t)
        if self.is_training and advice is not None and (use_advice or np.random.rand() < self.epsilon):
            action = advice
        else:
            phi_t = self.history.get_full_state()
            with torch.no_grad():
                action = to_numpy(self.actor(to_tensor(np.array([phi_t])))).squeeze(0)

        if decay_epsilon and not use_advice:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.history.reset(obs)

    def load_weights(self, output):
        prGreen("Loading from {}".format(output))
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output), map_location=device)
        )

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
