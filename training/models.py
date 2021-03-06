import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import ipdb as pdb
from torch.nn.modules.linear import Linear

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def safelife_cnn(input_shape):
    """
    Defines a CNN with good default values for safelife.

    This works best for inputs of size 25x25.

    Parameters
    ----------
    input_shape : tuple of ints
        Height, width, and number of channels for the board.

    Returns
    -------
    cnn : torch.nn.Sequential
    output_shape : tuple of ints
        Channels, width, and height.

    Returns both the CNN module and the final output shape.
    """
    h, w, c = input_shape
    """
    cnn = nn.Sequential(
        nn.Conv2d(c, 32, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1),
        nn.ReLU()
    )
    h = (h-5)//2 + 1
    h = (h-5)//2 + 1
    h = (h-5)//2 + 1
    h = (h-5)//2 + 1
    h = (h-3)//1 + 1
    h = (h-1)//2  # ?
    w = (w-5)//2 + 1
    w = (w-5)//2 + 1
    w = (w-5)//2 + 1
    w = (w-5)//2 + 1
    w = (w-3)//1 + 1
    w = (w-1)//2  # ?
    return cnn, (256, h, w)
    """
    cnn = nn.Sequential(
	nn.Conv2d(c, 32, kernel_size=5, stride=2),
	nn.ReLU(),
	nn.Conv2d(32, 64, kernel_size=3, stride=2),
	nn.ReLU(),
	nn.Conv2d(64, 64, kernel_size=3, stride=1),
	nn.ReLU()
	)
    h = (h-4+1)//2
    h = (h-2+1)//2
    h = (h-2)
    w = (w-4+1)//2
    w = (w-2+1)//2
    w = (w-2)

    return cnn, (64, w, h)
   

def signed_sqrt(x):
    s = torch.sign(x)
    return s * torch.sqrt(torch.abs(x))


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, factorized=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.factorized = factorized

        init_scale = in_features**-0.5
        self.weight_mu = nn.Parameter(
            2 * init_scale * (torch.rand(out_features, in_features)-0.5))
        self.weight_sigma = nn.Parameter(
            2 * init_scale * (torch.rand(out_features, in_features)-0.5))
        if self.use_bias:
            self.bias_mu = nn.Parameter(
                2 * init_scale * (torch.rand(out_features)-0.5))
            self.bias_sigma = nn.Parameter(
                2 * init_scale * (torch.rand(out_features)-0.5))

    def forward(self, x):
        b = None
        device = self.weight_mu.device
        if self.factorized:
            eps1 = signed_sqrt(torch.randn(self.in_features, device=device))
            eps2 = signed_sqrt(torch.randn(self.out_features, device=device))
            w = self.weight_mu + self.weight_sigma * eps1 * eps2[:,np.newaxis]

            if self.use_bias:
                # As with the original paper, use the signed sqrt even though
                # we're not taking a product of noise params.
                eps3 = signed_sqrt(torch.randn(self.out_features, device=device))
                b = self.bias_mu + self.bias_sigma * eps3

        else:
            eps1 = torch.randn(self.out_features, self.in_features, device=device)
            w = self.weight_mu + self.weight_sigma * eps1

            if self.use_bias:
                eps3 = torch.randn(self.out_features, device=device)
                b = self.bias_mu + self.bias_sigma * eps3

        return F.linear(x, w, b)


class SafeLifeQNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape, use_noisy_layers=True):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        Linear = NoisyLinear if use_noisy_layers else nn.Linear

        self.advantages = nn.Sequential(
            Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.value_func = nn.Sequential(
            Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        #pdb.set_trace()
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        advantages = self.advantages(x)
        value = self.value_func(x)
        qval = value + advantages - advantages.mean()
        return qval


class SafeLifePolicyNetwork(nn.Module):
    def __init__(self, input_shape, rfn=False):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        self.dense = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
        )
        self.logits = nn.Linear(512, num_actions)
        self.value_func = nn.Linear(512, num_actions)
        self.random_fn = None
    
    def register_reward_function(self, dim, projection, device):
        if projection:
            rfn = torch.ones(1, 90, 90).uniform_(0, 1).to(device)
            rfn = rfn.unsqueeze(0).repeat(16, 1, 1, 1)
        else:
            rfn = torch.ones(dim).to(device)
            if dim > 1:
                rfn = rfn.uniform_(0, 1).cuda()
            rfn = rfn.unsqueeze(0)
        self.random_fn = rfn

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs)
        x = x.flatten(start_dim=1)
        x = self.dense(x)
        value = self.value_func(x) # [batch, <1 or num_actions>]
        # value = value[...,0] # [batch]
            
        policy = F.softmax(self.logits(x), dim=-1)
        return value, policy

class SafeLifeMTQNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape, modR, use_noisy_layers=True):
        super().__init__()
        self.modR = modR
        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        Linear = NoisyLinear if use_noisy_layers else nn.Linear

        self.advantages = nn.Sequential(
            Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.value_func = nn.Sequential(
            Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.aux_advantages = [nn.Sequential(Linear(num_features, 256),nn.ReLU(),nn.Linear(256, num_actions)) for _ in range(modR)]
        self.aux_value_func = [nn.Sequential(Linear(num_features, 256),nn.ReLU(),nn.Linear(256, 1)) for _ in range(modR)]

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        # obs shape torch.Size([16, 25, 25, 10])
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        advantages = self.advantages(x)
        aux_advantages = [self.aux_advantages[i](x) for i in range(self.modR)]
        value = self.value_func(x)
        aux_value = [self.aux_value_func[i](x) for i in range(self.modR)]
        qval = value + advantages - advantages.mean()
        aux_qval = [aux_value[i] + aux_advantages[i] - aux_advantages[i].mean() for i in range(self.modR)]
        # output penalty term here? try one variation with that 
        # select greedy action lambda* Sum (Qs - Qphi)/modR
        return qval, torch.stack(aux_qval)

class SafeLifePEQNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape, modR, use_noisy_layers=True):
        super().__init__()
        self.modR = modR
        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        self.num_features = np.product(cnn_out_shape)
        self.num_actions = 9

        Linear = NoisyLinear if use_noisy_layers else nn.Linear

        self.advantages = nn.Sequential(
            Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

        self.value_func = nn.Sequential(
            Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.aux_advantages = [nn.Sequential(Linear(self.num_features, 256),nn.ReLU(),nn.Linear(256, self.num_actions)) for _ in range(self.modR)]
        self.aux_value_func = [nn.Sequential(Linear(self.num_features, 256),nn.ReLU(),nn.Linear(256, 1)) for _ in range(self.modR)]

    def forward(self, obs, epsilon, out_penalty=False):
        # Switch observation to (c, w, h) instead of (h, w, c)
        # obs shape torch.Size([16, 25, 25, 10])
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        advantages = self.advantages(x)
        aux_advantages = [self.aux_advantages[i](x) for i in range(self.modR)]
        value = self.value_func(x)
        aux_value = [self.aux_value_func[i](x) for i in range(self.modR)]
        qval = value + advantages - advantages.mean()
        aux_qval = [aux_value[i] + aux_advantages[i] - aux_advantages[i].mean() for i in range(self.modR)]
        # output penalty term here? try one variation with that 
        # select epsilon greedy action, calculate penalty term
        chosen_actions = torch.argmax(qval, axis=-1)
        use_random = torch.rand(qval.shape[0]) < epsilon
        random_actions = torch.randint(0,self.num_actions,(qval.shape[0],))
        actions = np.choose(use_random, [chosen_actions, random_actions])
        #aux_qval= torch.stack(aux_qval)
        if out_penalty:
            aux_q_value = torch.stack([aux_qval[i].gather(1, actions.unsqueeze(1)).squeeze(1) for i in range(self.modR)])
            noop_aux_q_value = torch.stack([aux_qval[i][:,0] for i in range(self.modR)])
            penalty_term = torch.mean(torch.abs(noop_aux_q_value-aux_q_value),dim=0)
            return qval, penalty_term, actions 
        else:
            return qval, actions                       

class RandomNN(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape, modR):
        super().__init__()
        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9
        self.fc1 = nn.Linear(576,modR)
        

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        # obs shape torch.Size([16, 25, 25, 10])
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        x = self.fc1(x)
        return torch.tanh(x)        