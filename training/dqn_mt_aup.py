import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from safelife.helper_utils import load_kwargs
from safelife.render_graphics import render_board

from . import checkpointing
from .utils import round_up, LinearSchedule

from .cb_vae import train_encoder, load_state_encoder, encode_state
import ipdb as pdb
from copy import deepcopy

USE_CUDA = torch.cuda.is_available()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.buffer = np.zeros(capacity, dtype=object)

    def push(self, *data):
        self.buffer[self.idx % self.capacity] = data
        self.idx += 1

    def sample(self, batch_size):
        sub_buffer = self.buffer[:self.idx]
        data = np.random.choice(sub_buffer, batch_size, replace=False)
        return zip(*data)

    def __len__(self):
        return min(self.idx, self.capacity)


class DQN_MT_AUP(object):
    summary_writer = None
    logdir = None

    num_steps = 0
    num_episodes = 0

    gamma = 0.97
    training_batch_size = 64
    optimize_freq = 16
    learning_rate_aup = 3e-4

    replay_initial = 40000
    replay_size = 100000
    target_update_freq = 10000

    checkpoint_freq = 100000
    num_checkpoints = 3
    report_freq = 256
    test_freq = 100000

    #compute_device = torch.device('cuda:0' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    epsilon = 0.0  # for exploration

    def __init__(
            self,
            training_model_aup,
            target_model_aup,
            reward_model,
            modR,
            env_type,
            **kwargs):

        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.training_model_aup = training_model_aup#.to(self.compute_device)
        self.target_model_aup = target_model_aup#.to(self.compute_device)
        
        self.optimizer_aup = optim.Adam(
                self.training_model_aup.parameters(), lr=self.learning_rate_aup)
        self.replay_buffer_aup = ReplayBuffer(self.replay_size)

        checkpointing.load_checkpoint(self.logdir, self)

        self.exp = env_type
        self.buffer_size = 50e3
        self.lamb_schedule = LinearSchedule(1.985e6, initial_p=0.015, final_p=0.015)
        self.reward_model = reward_model
        self.modR = modR
        

    def epsilon_old(self):
        # hardcode this for now
        t1 = 1e5
        t2 = 1e6
        y1 = 1.0
        y2 = 0.1
        t = (self.num_steps - t1) / (t2 - t1)
        return y1 + (y2-y1) * np.clip(t, 0, 1)

    def get_random_rewards(self, states):
        #
        states = torch.stack(states)
        if self.is_random_projection:
            #states = states.to(self.compute_device)
            rp = self.random_fns[0].transpose(2, 3)
            r = torch.einsum('abcd, abde -> abce', states, rp)
            rewards = r.view(r.size(0), -1).mean(1)
        else:
            self.state_encoder.eval()
            states_z = encode_state(self.state_encoder, states, self.compute_device)
            rewards = torch.mm(states_z, self.random_fns.T)
        return rewards

    def preprocess_state(self, env, reset=False, return_original=False):
        if reset:
            _ = env.reset()
        obs = render_board(env.game.board, env.game.goals, env.game.orientation)
        obs = np.asarray(obs)
        obsp = torch.from_numpy(np.matmul(obs[:, :, :3], [0.299, 0.587, 0.114]))
        obsp = obsp.unsqueeze(0) # [1, batch, H, W]
        if obsp.size(-1) == 210: # test env
            obsp = F.avg_pool2d(obsp, 2, 2)/255.
        else: # big env
            obsp = F.avg_pool2d(obsp, 5, 4)/255.
        if return_original:
            ret = (obsp.float(), obs)
        else:
            ret = obsp.float()
        return ret
    
    def run_test_envs(self):
        # Just run one episode of each test environment.
        # Assumes that the environments themselves handle logging.
        model = self.training_model_aup
        for env in self.testing_envs:
            state = env.reset()
            done = False
            while not done:
                state = torch.tensor([state], device=self.compute_device, dtype=torch.float32)
                qvals = model(state).detach().cpu().numpy().ravel()
                state, reward, done, info = env.step(np.argmax(qvals))

    def collect_data(self):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in self.training_envs
        ]
        # if self.training_aux:
        #     rstates = [
        #         e.last_rstate if hasattr(e, 'last_rstate') else self.preprocess_state(e, reset=True)
        #         for e in self.training_envs
        #     ]
        #     #
        #     rreward = self.get_random_rewards(rstates)
        #     rreward = rreward.squeeze(-1).tolist()
        # else:
        #     rstates = None

        tensor_states = torch.tensor(states, dtype=torch.float32)#device=self.compute_device, dtype=torch.float32)
        
        # get aup actions and values if needed
        # if not self.training_aux:
        #     qvals_aup = self.training_model_aup(tensor_states).detach().cpu().numpy()
        #     actions_aup = np.argmax(qvals_aup, axis=-1)
        # qval aux dimension modr,env,9
        qvals_aup, qvals_aux = self.training_model_aup(tensor_states)#.detach().cpu().numpy()
        actions_aup = torch.argmax(qvals_aup, axis=-1)
        aux_rewards = self.reward_model(tensor_states)
        num_states, num_actions = qvals_aup.shape
        #
        
        random_actions = np.random.randint(num_actions, size=num_states)
        use_random = np.random.random(num_states) < self.epsilon
        actions = actions_aup
        actions = np.choose(use_random, [actions, random_actions])

        self.penalty = []
        self.penalty = []
        action_actor = None
        for i, (env, state, action, aux_reward) in enumerate(zip(self.training_envs, states, actions, aux_rewards)):
            action_actor = action
            #copy env to get a counterfactual no-op next state
            #
            # noop_env = deepcopy(env.env)
            # next_state_noop, reward_noop, done_noop, info_noop = noop_env.step(0)
            # del noop_env
            next_state, reward, done, info = env.step(action_actor)
            
            
            # noop_value = qvals_aux[:, 0]
            # max_value = qvals_aux[:, action_actor]
            # penalty = np.abs(max_value - noop_value)
            # lamb = self.lamb_schedule.value(self.num_steps)
            # reward = reward - lamb * penalty[i]
            # self.penalty.append(penalty)
            
            if done:
                next_state = env.reset()
                self.num_episodes += 1
            env.last_state = next_state
            replay_buffer = self.replay_buffer_aup
            replay_buffer.push(state, action, reward, next_state, done, aux_reward)

        self.num_steps += len(states)
        self.epsilon = self.epsilon_old()

    def optimize(self, report=False):
        #
        replay_buffer = self.replay_buffer_aup
        model = self.training_model_aup
        target_model = self.target_model_aup
        optimizer = self.optimizer_aup
        # if len(replay_buffer) < self.replay_initial:
        #     return

        state, action, reward, next_state, done, aux_reward = \
            replay_buffer.sample(self.training_batch_size)

        # state = torch.tensor(state, device=self.compute_device, dtype=torch.float32)
        # next_state = torch.tensor(next_state, device=self.compute_device, dtype=torch.float32)
        # action = torch.tensor(action, device=self.compute_device, dtype=torch.int64)
        # reward = torch.tensor(reward, device=self.compute_device, dtype=torch.float32)
        # done = torch.tensor(done, device=self.compute_device, dtype=torch.float32)

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        aux_reward = torch.tensor(aux_reward, dtype=torch.float32)
        
        q_values, aux_q_values = model(state)
        next_q_values, aux_next_q_values = target_model(next_state)
        next_q_values, aux_next_q_values = next_q_values.detach(), aux_next_q_values.detach()
        
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value, next_action = next_q_values.max(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        #
        aux_q_value = [aux_q_values[i].gather(1, action.unsqueeze(1)).squeeze(1) for i in range(self.modR)]
        aux_next_q_value = [aux_next_q_values[i].max(1)[0] for i in range(self.modR)]
        aux_next_action = [aux_next_q_values[i].max(1)[1] for i in range(self.modR)]
        aux_expected_q_value = [aux_reward[i] + self.gamma * aux_next_q_value[i] * (1 - done) for i in range(self.modR)]
        #print(aux_q_value,aux_next_q_value,aux_next_action,aux_expected_q_value)
        
        main_loss = torch.mean((q_value - expected_q_value)**2)
        aux_losses = [torch.mean((aux_q_value[i] - aux_expected_q_value[i])**2) for i in range(self.modR)]
        #if not self.training_aux:
        #    print (loss.shape)
        
        loss = main_loss+0.01*sum(aux_losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer = self.summary_writer
        n = self.num_steps
        if report and self.summary_writer is not None:
            writer.add_scalar("loss", loss.item(), n)
            writer.add_scalar("epsilon", self.epsilon, n)
            writer.add_scalar("qvals/model_mean", q_values.mean().item(), n)
            writer.add_scalar("qvals/model_max", q_values.max(1)[0].mean().item(), n)
            writer.add_scalar("qvals/target_mean", next_q_values.mean().item(), n)
            writer.add_scalar("qvals/target_max", next_q_value.mean().item(), n)
            writer.add_scalar("aux_qvals/model_mean", aux_q_values[0].mean().item(), n)
            writer.add_scalar("aux_qvals/model_max", aux_q_values[0].max(1)[0].mean().item(), n)
            writer.add_scalar("aux_qvals/target_mean", aux_next_q_values[0].mean().item(), n)
            writer.add_scalar("aux_qvals/target_max", aux_next_q_value[0].mean().item(), n)

            writer.flush()

    def train(self, steps):
        needs_report = True

        for _ in range(int(steps / len(self.training_envs))):
            num_steps = self.num_steps
            next_opt = round_up(num_steps, self.optimize_freq)
            next_update = round_up(num_steps, self.target_update_freq)
            next_checkpoint = round_up(num_steps, self.checkpoint_freq)
            next_report = round_up(num_steps, self.report_freq)
            next_test = round_up(num_steps, self.test_freq)

            self.collect_data()

            num_steps = self.num_steps

            replay_buffer = self.replay_buffer_aup
            if len(replay_buffer) < self.replay_initial:
                continue

            if num_steps >= next_report:
                needs_report = True

            if num_steps >= next_opt:
                self.optimize(needs_report)
                needs_report = False

            if num_steps >= next_update:
                target_model = self.target_model_aup
                model = self.training_model_aup
                target_model.load_state_dict(model.state_dict())

            if num_steps >= next_checkpoint:
                checkpointing.save_checkpoint(self.logdir, self, ['training_model_aup','target_model_aup','optimizer_aup'])

            # if num_steps >= next_test:
            #     self.run_test_envs()


