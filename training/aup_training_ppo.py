import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from safelife.helper_utils import load_kwargs
from safelife.render_graphics import render_board

from .utils import named_output, round_up, LinearSchedule
from . import checkpointing
from .cb_vae import train_encoder, load_state_encoder, encode_state

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class PPO_AUP(object):
    summary_writer = None
    logdir = None

    num_steps = 0
    num_episodes = 0

    steps_per_env = 20
    num_minibatches = 4
    epochs_per_batch = 3

    gamma = 0.97
    lmda = 0.95
    learning_rate_aup = 3e-4
    entropy_reg_aup = 0.01 

    entropy_clip = 1.0  # don't start regularization until it drops below this
    vf_coef = 0.5
    max_gradient_norm = 5.0
    eps_policy = 0.2  # PPO clipping for policy loss
    eps_value = 0.2  # PPO clipping for value loss
    rescale_policy_eps = False
    min_eps_rescale = 1e-3  # Only relevant if rescale_policy_eps = True
    reward_clip = 0.0
    policy_rectifier = 'relu'  # or 'elu' or ...more to come

    checkpoint_freq = 100000
    num_checkpoints = 3
    report_freq = 960
    test_freq = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None
    epsilon = 0.0  # for exploration


    def __init__(self,
        model_aup,
        model_aux,
        env_type,
        z_dim,
        aup_train_steps,
        **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None
        
        self.aup_train_steps = aup_train_steps
        for m in model_aux.model_aux:
            m.to(self.compute_device)
        self.model_aux = model_aux.model_aux
        self.optimizer_aux = model_aux.optimizer_aux
        self.model_aup = model_aup.to(self.compute_device)
        self.optimizer_aup = optim.Adam(self.model_aup.parameters(), lr=self.learning_rate_aup)
        checkpointing.load_checkpoint(self.logdir, self, aup=True)
        self.exp = env_type

        """ AUP-specific parameters """
        self.z_dim = z_dim
        self.use_scale = False
        if env_type == 'append-still':
            n_steps = 5e6
        else:
            n_steps = 4e6

        self.lamb_schedule = LinearSchedule(n_steps, initial_p=1e-3, final_p=1e-1)
   
    def tensor(self, data, dtype):
        data = np.asanyarray(data)
        return torch.as_tensor(data, device=self.compute_device, dtype=dtype)

    @named_output('states actions rewards done policies values')
    def take_one_step_aup(self, envs):
        states = [
            e.last_obs if hasattr(e, 'last_obs') else e.reset()
            for e in envs
        ]
        tensor_states = self.tensor(states, dtype=torch.float32)
        values_q, policies = self.model_aup(tensor_states)
        values = values_q.mean(1)

        values_q_aux = []
        policies_aux = []
        for model in self.model_aux:
            value_aux, policy_aux = model(tensor_states)
            values_q_aux.append(value_aux)
            policies_aux.append(policy_aux)
        
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()
        
        actions = []
        rewards = []
        dones = []

        for i, (policy, env) in enumerate(zip(policies, envs)):
            action = np.random.choice(len(policy), p=policy) # fine 
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            env.last_obs = obs
            actions.append(action)

            # calculate AUP penalty
            penalty = 0.
            for v, p in zip(values_q_aux, policies_aux):
                inaction_value = v[i, 0]
                action_value = v[i, action]
                penalty += (action_value - inaction_value).abs()
            self.scale = inaction_value
            if self.use_scale:
                scale = inaction_value
            else:
                scale = 1.
            lamb = self.lamb_schedule.value(self.num_steps)
            self.lamb = lamb
            penalty = penalty / len(self.model_aux)
            self.penalty = penalty
            reward = reward - lamb * (penalty / scale)
            reward = reward.cpu().tolist()
            rewards.append(reward)
            dones.append(done)
       
        return states, actions, rewards, dones, policies, values


    def run_test_envs(self):
        """
        Run each testing environment until completion.

        It's assumed that the testing environments take care of their own
        logging via wrappers.
        """
        test_envs = self.testing_envs or []
        while test_envs:
            data = self.take_one_step_aup(test_envs)
            test_envs = [
                env for env, done in zip(test_envs, data.done) if not done
            ]

    @named_output('states actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env, flat=True):
        """
        Run each environment a number of steps and calculate advantages.

        Parameters
        ----------
        steps_per_env : int
            Number of steps to take per environment.
        flat : bool
            If True, each output tensor will have shape
            ``(steps_per_env * num_env, ...)``.
            Otherwise, shape will be ``(steps_per_env, num_env, ...)``.
        """
        steps = [self.take_one_step_aup(self.training_envs) for _ in range(steps_per_env)] # [steps]
        final_states = [e.last_obs for e in self.training_envs]
        tensor_states = self.tensor(final_states, dtype=torch.float32)
        final_vals = self.model_aup(tensor_states)[0]
        final_vals = final_vals.mean(1).detach().cpu().numpy()  # adjust for learning Q function
        values = np.array([s.values for s in steps] + [final_vals])

        rewards = np.array([s.rewards for s in steps])
        done = np.array([s.done for s in steps])
        reward_mask = ~done

        # Calculate the discounted rewards
        gamma = self.gamma
        lmda = self.lmda
        returns = rewards.copy()
        # shape of returns is different
        returns[-1] += gamma * final_vals * reward_mask[-1]
        advantages = rewards + gamma * reward_mask * values[1:] - values[:-1]
        for i in range(steps_per_env - 2, -1, -1):
            returns[i] += gamma * reward_mask[i] * returns[i+1]
            advantages[i] += lmda * reward_mask[i] * advantages[i+1]

        # Calculate the probability of taking each selected action
        policies = np.array([s.policies for s in steps])
        actions = np.array([s.actions for s in steps])
        probs = np.take_along_axis(
            policies, actions[..., np.newaxis], axis=-1)[..., 0]

        def t(x, dtype=torch.float32):
            if flat:
                x = np.asanyarray(x)
                x = x.reshape(-1, *x.shape[2:])
            return self.tensor(x, dtype=dtype)

        self.num_steps += actions.size
        self.num_episodes += np.sum(done)
        return (
                t([s.states for s in steps]), t(actions, torch.int64),
                t(probs), t(returns), t(advantages), t(values[:-1]),
                )

    def calculate_loss(
            self, states, actions, old_policy, old_values, returns, advantages):
        """
        All parameters ought to be tensors on the appropriate compute device.
        ne_step = s
        """
        values, policy = self.model_aup(states)
        values = values.mean(1)  # adjust for learning Q function

        a_policy = torch.gather(policy, -1, actions[..., np.newaxis])[..., 0]

        prob_diff = advantages.sign() * (1 - a_policy / old_policy)
        policy_loss = advantages.abs() * torch.clamp(prob_diff, min=-self.eps_policy)
        policy_loss = policy_loss.mean()

        v_clip = old_values + torch.clamp(
            values - old_values, min=-self.eps_value, max=+self.eps_value)
        value_loss = torch.max((v_clip - returns)**2, (values - returns)**2)
        value_loss = value_loss.mean()

        entropy = torch.sum(-policy * torch.log(policy + 1e-12), dim=-1)
        entropy_loss = torch.clamp(entropy.mean(), max=self.entropy_clip)
        entropy_loss *= -self.entropy_reg_aup

        return entropy, policy_loss + value_loss * self.vf_coef + entropy_loss

    def train_batch(self, batch):
        idx = np.arange(len(batch.states))

        for _ in range(self.epochs_per_batch):
            np.random.shuffle(idx)
            for k in idx.reshape(self.num_minibatches, -1):
                entropy, loss = self.calculate_loss(
                    batch.states[k], batch.actions[k], batch.action_prob[k],
                    batch.values[k], batch.returns[k], batch.advantages[k])
                self.optimizer_aup.zero_grad()
                loss.backward()
                self.optimizer_aup.step()

    def train(self):
        print ('starting training')
        max_steps = self.aup_train_steps
        
        while self.num_steps < max_steps:
            next_checkpoint = round_up(self.num_steps, self.checkpoint_freq)
            next_report = round_up(self.num_steps, self.report_freq)
            next_test = round_up(self.num_steps, self.test_freq)
            batch = self.gen_training_batch(self.steps_per_env)
            self.train_batch(batch)

            n = self.num_steps

            if n >= next_report and self.summary_writer is not None:
                writer = self.summary_writer
                entropy, loss = self.calculate_loss(
                    batch.states, batch.actions, batch.action_prob,
                    batch.values, batch.returns, batch.advantages)
                loss = loss.item()
                entropy = entropy.mean().item()
                values = batch.values.mean().item()
                advantages = batch.advantages.mean().item()
                scale = self.scale.mean().item()
                lamb = self.lamb
                penalty = self.penalty.item()
                logger.info(
                    "n=%i: loss=%0.3g, entropy=%0.3f, val=%0.3g, adv=%0.3g",
                    n, loss, entropy, values, advantages)
                writer.add_scalar("training/loss", loss, n)
                writer.add_scalar("training/entropy", entropy, n)
                writer.add_scalar("training/values", values, n)
                writer.add_scalar("training/advantages", advantages, n)
                writer.add_scalar("training/scale_value", scale, n)
                writer.add_scalar("training/lambda", lamb, n)
                writer.add_scalar("training/aup_penalty", penalty, n)
                writer.flush()

            if n >= next_checkpoint:
                checkpointing.save_checkpoint(self.logdir, self, [
                    'model_aux', 'optimizer_aux', 'model_aup', 'optimizer_aup',
                ], 'aup_')

            if n >= next_test:
                if self.testing_envs is not None:
                    self.run_test_envs()
        
        # Closing up
        if self.training_envs is not None:
            for env in self.training_envs:
                env.close()
        if self.testing_envs is not None:
            for env in self.testing_envs:
                env.close()
