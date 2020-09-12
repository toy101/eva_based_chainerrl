from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library

standard_library.install_aliases()  # NOQA

from logging import getLogger
import numpy as np

import chainer
from chainer import cuda, Variable
from chainerrl.agents import dqn
from chainerrl.misc.batch_states import batch_states
from chainerrl.action_value import DiscreteActionValue
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
import copy

from value_buffer import ValueBuffer, WeightedValueBuffer


def batch_trajectory(trajectory, xp, phi, gamma, batch_states=batch_states):
    batch_tr = {
        'state': batch_states(
            [elem['state'] for elem in trajectory], xp, phi),
        'action': np.asarray([elem['action'] for elem in trajectory], dtype=np.int32),
        'reward': np.asarray([elem['reward'] for elem in trajectory], dtype=np.float32),
        'is_state_terminal': np.asarray(
            [elem['is_state_terminal'] for elem in trajectory], dtype=np.float32),
        'embedding': [elem['embedding'] for elem in trajectory]
    }

    return batch_tr

class EVA(dqn.DQN):
    """EVAアルゴリズム(DQNに適用)"""
    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=40000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000, clip_delta=True,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='sum', episodic_update=False,
                 episodic_update_len=100,
                 logger=getLogger(__name__),
                 batch_states=batch_states,
                 insert_period = 20,
                 lamba = 0.5, xp = np,
                 vbuf_use_weight = False):


        super().__init__(q_function, optimizer, replay_buffer, gamma,
                         explorer, gpu, replay_start_size=replay_start_size,
                         minibatch_size=minibatch_size, update_interval=update_interval,
                         target_update_interval=target_update_interval, clip_delta=clip_delta,
                         phi=phi,
                         target_update_method=target_update_method,
                         soft_update_tau=soft_update_tau,
                         n_times_update=n_times_update, average_q_decay=average_q_decay,
                         average_loss_decay=average_loss_decay,
                         batch_accumulator=batch_accumulator, episodic_update=episodic_update,
                         episodic_update_len=episodic_update_len,
                         logger=logger,
                         batch_states=batch_states)



        # 必要なパラメータ記述
        self.last_embedding = None
        self.replay_buffer = replay_buffer
        self.insert_period = insert_period
        self.num_action = self.model.num_action
        self.trajectory_max_len = self.replay_buffer.follow_range

        if vbuf_use_weight:
            self.value_buffer = WeightedValueBuffer(capacity=2000,
                                        n_action=self.model.num_action,
                                        xp = xp)
        else:
            self.value_buffer = ValueBuffer(capacity=2000,
                                        n_action=self.model.num_action,
                                        xp = xp)
        self.lamda = lamba
        self.eval_t = 0

    """
    def sync_target_network(self):
        
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)
    """

    def act_and_train(self, obs, reward):

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                #84*84を1*84*84にしている
                #この形で渡さないといけない
                #そのためにobsを[]に入れて次元を増やしている
                self.batch_states([obs], self.xp, self.phi))

            # embeddingの所得
            embedding = self.model.embedding

            #Q値を合わせる
            if len(self.value_buffer) > 0:
                q_np = self.value_buffer.compute_q(embedding=embedding)
                q_theta = action_value.q_values.array
                q = Variable(self.lamda*q_theta + (1 - self.lamda)*q_np)
                q = DiscreteActionValue(q)
                q = float(q.max.array)
                #q_mixed = (1-self.lamda)*q_theta + self.lamda*q_np
                #q = float(q_mixed.max())
                #greedy_action = cuda.to_cpu(q_mixed.argmax())

            else:
                q = float(action_value.max.array)

        greedy_action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            assert self.last_embedding is not None
            # Add a transition to the replay buffer
            self.replay_buffer.add(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                embedding=self.last_embedding,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self._backup_if_necessary(self.t, embedding)

        self.last_state = obs
        self.last_action = action
        self.last_embedding = embedding

        self.replay_updater.update_if_necessary(self.t)

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """
        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        assert self.last_state is not None
        assert self.last_action is not None
        assert self.last_embedding is not None

        # Add a transition to the replay buffer
        self.replay_buffer.add(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            embedding=self.last_embedding,
            next_state=state,
            next_action=None,
            is_state_terminal=done)

        self._backup_if_necessary(self.t, self.last_embedding)

        self.last_embedding = None
        self.stop_episode()

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))

            embedding = self.model.embedding

            if len(self.value_buffer) > 0:
                q_np = self.value_buffer.compute_q(embedding=embedding)
                q_theta = action_value.q_values.array
                q = Variable((1 - self.lamda) * q_theta + self.lamda * q_np)
                q = DiscreteActionValue(q)
                q = float(q.max.array)

            else:
                q = float(action_value.max.array)

        action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        self.eval_t += 1
        self._backup_if_necessary(self.eval_t, embedding)

        return action

    def stop_episode(self):
        self.last_state = None
        self.last_action = None
        self.last_embedding = None
        self.eval_t = 0

        if isinstance(self.model, Recurrent):
            self.model.reset_state()

        self.replay_buffer.stop_current_episode()

    def _trajectory_centric_planning(self, trajectories):

        #遷移によってエピソード数が違うかもしれないので、形を整える
        batch_states = self.xp.empty((0, 4, 84, 84), dtype='float32')
        for trajectory in trajectories:
            shape_tr = self.xp.empty((self.trajectory_max_len, 4, 84, 84), dtype='float32')
            shape_tr[:len(trajectory['state'])] = trajectory['state']
            batch_states = self.xp.vstack((batch_states, shape_tr))

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            q_theta = self.target_model(batch_states)
            q_theta_all = cuda.to_cpu(q_theta.q_values.array).reshape((len(trajectories),
                                                                       self.trajectory_max_len, self.num_action))

        for q_theta_tr, trajectory in zip(q_theta_all, trajectories):

#            batch_state = trajectory['state']
            batch_action = trajectory['action']
            batch_reward = trajectory['reward']
            batch_embedding = trajectory["embedding"]

            q_np_tr = q_theta_tr[:len(batch_action)]
            v_np = np.max(q_np_tr[-1])
            for t in range(len(batch_action)-2, -1, -1):
            # range(一番後ろはとばして, -1になる前=0番目まで, -1ずつ減らす(逆順))
                #実際にとった行動の場合、Q値を算出
                q_np_tr[t][batch_action[t]] = batch_reward[t] + self.gamma * v_np
                # ここでemmbeddingとQnpベクトル追加
                self.value_buffer.store(batch_embedding[t], q_np_tr[t])
                v_np = np.max(q_np_tr[t])



            """
                # V(s) := maxQ(s', a)
                param_q = self.model(
                    self.batch_states([transition[-1]["state"]],
                                      self.xp, self.phi)
                )
                param_q = cuda.to_cpu(param_q.q_values.array)
                v_np = float(param_q.max())
    

                #行動の数分回す
                for a in range(self.num_action):
                    if a == t["action"]:
                        # Qnp(s, a) = r + gamma*Vnp(s)
                        q_np = t["reward"] + self.gamma * v_np
                    else:
                        # Qnp(s, a) = Q(s, a)
                        q_np = param_q[0][a]
                    q_np_vector.append(q_np.astype("float32"))
            """


    def _backup_if_necessary(self, t, embedding):
        if t % self.insert_period == 0 and len(self.replay_buffer) >= self.replay_buffer.capacity:
            trajectories = self.replay_buffer.lookup(embedding=embedding)
            batch_trajectories = [batch_trajectory(trajectory, self.xp, self.phi, self.gamma,
                                                   batch_states=batch_states) for trajectory in trajectories]
            self._trajectory_centric_planning(batch_trajectories)





