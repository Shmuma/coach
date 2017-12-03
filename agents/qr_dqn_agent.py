#
# Copyright (c) 2017 Intel Corporation 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from agents.value_optimization_agent import *

import os
import matplotlib.pylab as plt

import numpy as np


# Quantile Regression Deep Q Network - https://arxiv.org/pdf/1710.10044v1.pdf
class QuantileRegressionDQNAgent(ValueOptimizationAgent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        ValueOptimizationAgent.__init__(self, env, tuning_parameters, replicated_device, thread_id)
        self.quantile_probabilities = np.ones(self.tp.agent.atoms) / float(self.tp.agent.atoms)
        self.done_state = None
        self.done_act = None
        self.normal_states = []
        self.normal_acts = []

    # prediction's format is (batch,actions,atoms)
    def get_q_values(self, quantile_values):
        return np.dot(quantile_values, self.quantile_probabilities)

    def _gather_states(self, states, actions, overs, normal_count=5):
        if self.done_state is not None and len(self.normal_states) == normal_count:
            return
        for s, a, o in zip(states, actions, overs):
            if o and self.done_state is None:
                self.done_state = s
                self.done_act = a
                continue
            if len(self.normal_states) < normal_count:
                self.normal_states.append(s)
                self.normal_acts.append(a)

    def _plot_one(self, x, y, f_name, sort=False):
        plt.clf()
        q_val = self.get_q_values(y)
        if sort:
            y = list(sorted(y))
        plt.plot(x, y)
        plt.title("Inv CDF, q_val=%.4f" % q_val)
        plt.savefig(f_name)


    def _plot_states(self, steps=10000, out_dir='img_qr'):
        if self.total_steps_counter % steps != 0:
            return
        if self.done_state is None or len(self.normal_states) < 5:
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        done_quantiles = self.main_network.target_network.predict(np.expand_dims(self.done_state, 0))
        done_quantiles = done_quantiles[0, self.done_act]
        norm_quantiles = self.main_network.target_network.predict(np.array(self.normal_states))
        norm_quantiles = norm_quantiles[range(len(self.normal_states)), self.normal_acts]

        cum_probs = np.array(range(self.tp.agent.atoms+1))/float(self.tp.agent.atoms)
        cum_probs = cum_probs[1:]

        fname = os.path.join(out_dir, "raw_done_%06d.png" % self.total_steps_counter)
        self._plot_one(cum_probs, done_quantiles, fname, sort=False)
        fname = os.path.join(out_dir, "sorted_done_%06d.png" % self.total_steps_counter)
        self._plot_one(cum_probs, done_quantiles, fname, sort=True)

        for idx, quant in enumerate(norm_quantiles):
            fname = os.path.join(out_dir, "raw_%d_%06d.png" % (idx, self.total_steps_counter))
            self._plot_one(cum_probs, quant, fname, sort=False)
            fname = os.path.join(out_dir, "sorted_%d_%06d.png" % (idx, self.total_steps_counter))
            self._plot_one(cum_probs, quant, fname, sort=True)

        return

    def learn_from_batch(self, batch):
        current_states, next_states, actions, rewards, game_overs, _ = self.extract_batch(batch)
        self._gather_states(current_states, actions, game_overs)
        self._plot_states()

        # get the quantiles of the next states and current states
        next_state_quantiles = self.main_network.target_network.predict(next_states)
        # current_quantiles = self.main_network.online_network.predict(current_states)

        # get the optimal actions to take for the next states
        target_actions = np.argmax(self.get_q_values(next_state_quantiles), axis=1)

        # calculate the Bellman update
        batch_idx = list(range(self.tp.batch_size))
        rewards = np.expand_dims(rewards, -1)
        game_overs = np.expand_dims(game_overs, -1)
        TD_targets = rewards + (1.0 - game_overs) * self.tp.agent.discount \
                               * next_state_quantiles[batch_idx, target_actions]

        # get the locations of the selected actions within the batch for indexing purposes
        actions_locations = [[b, a] for b, a in zip(batch_idx, actions)]

        # calculate the cumulative quantile probabilities and reorder them to fit the sorted quantiles order
        cumulative_probabilities = np.array(range(self.tp.agent.atoms+1))/float(self.tp.agent.atoms)  # tau_i
        quantile_midpoints = 0.5*(cumulative_probabilities[1:] + cumulative_probabilities[:-1])  # tau^hat_i
        quantile_midpoints = np.tile(quantile_midpoints, (self.tp.batch_size, 1))
        # sorted_quantiles = np.argsort(current_quantiles[batch_idx, actions])
        # for idx in range(self.tp.batch_size):
        #     quantile_midpoints[idx, :] = quantile_midpoints[idx, sorted_quantiles[idx]]

        # train
        result = self.main_network.train_and_sync_networks([current_states, actions_locations, quantile_midpoints], TD_targets)
        total_loss = result[0]

        return total_loss

