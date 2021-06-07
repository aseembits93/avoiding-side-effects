import torch
import torch.nn as nn
import torch.nn.functional as F 

import random
import numpy as np
from statistics import mean

# from Environment import CreateBreakout
from .Network import QNet_LSTM

# settings
# Train_max_step         = 4000000
Train_max_step         = 1000000
learning_rate          = 1e-4
# gamma                  = 0.99
gamma                  = 0.97
buffer_capacity        = 500000
# batch_size             = 32
batch_size             = 64
replay_start_size      = 50000
# replay_start_size      = 1000
final_exploration_step = 1000000
update_interval        = 10000 # target net
# update_frequency       = 4  # the number of actions selected by the agent between successive SGD updates
update_frequency       = 16  # the number of actions selected by the agent between successive SGD updates
save_interval          = 1000
# model_path             = './Models/Breakout_DRQN.model'
# history_path           = './Train_Historys/Breakout_DRQN'
model_path             = '/scratch/crowleyd/avoiding-side-effects/training_results/breakout_drqn/models/Breakout_DRQN.model'
history_path           = '/scratch/crowleyd/avoiding-side-effects/training_results/breakout_drqn/train_histories/Breakout_DRQN'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.Buffer = []
        self.position = 0

    def push(self, transition):
        """
        push transition data to Beffer

        input:
          transition -- list of [s, a, r, t]
        """
        if len(self.Buffer) < self.capacity:
            self.Buffer.append(None)
        self.Buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random_idx = random.randint(0, len(self.Buffer) - batch_size)
        mini_batch = self.Buffer[random_idx : random_idx + batch_size]

        s_batch, a_batch, r_batch, t_batch = [], [], [], []
        for transition in mini_batch:
            s, a, r, t = transition

            s_batch.append(s)
            a_batch.append([a])
            r_batch.append([r])
            t_batch.append([t])
        
        return s_batch, a_batch, r_batch, t_batch

    def size(self):
        return len(self.Buffer)

def init_hidden():
    h, c = torch.zeros([1, 256], dtype=torch.float).to(device), torch.zeros([1, 256], dtype=torch.float).to(device) 
    return h, c

def train(optimizer, behaviourNet, targetNet, s_batch, a_batch, r_batch, done_batch, aup_model):
    s_batch = torch.stack(s_batch)
    a_batch = torch.FloatTensor(a_batch[:-1])
    r_batch = torch.FloatTensor(r_batch[:-1])
    done_batch = torch.FloatTensor(done_batch)
    
    hb, cb = init_hidden()
    ht, ct = init_hidden()
    Q_batch = []
    target_Q_batch = []
    for state, done in zip(s_batch, done_batch):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        Q, (hb, cb) = behaviourNet.forward(state_tensor.to(device), (hb, cb))
        target_Q, (ht, ct) =  targetNet.forward(state.unsqueeze(0).to(device), (ht, ct))

        Q_batch.append(Q)
        target_Q_batch.append(target_Q)

        if done.item() == 0:
            hb, cb = init_hidden()
            ht, ct = init_hidden()
    
    Q_batch = torch.cat(Q_batch[:-1]).to(device)
    next_Q_batch = torch.cat(target_Q_batch[1:])

    Q_a = Q_batch.gather(1, a_batch.long().to(device))

    max_next_Q = next_Q_batch.max(1, keepdims=True)[0].to(device)
    TD_target = r_batch.to(device) + gamma * max_next_Q * done_batch[:-1].to(device)

    loss = F.smooth_l1_loss(Q_a, TD_target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# def main():
def breakout_drqn(env, aup_model):
    aup_model.to(device)
    # env = CreateBreakout(stack=False)
    obs_shape = env.observation_space.shape
    buffer = ReplayBuffer(buffer_capacity)
    behaviourNet = QNet_LSTM(obs_shape)
    behaviourNet.to(device)
    #behaviourNet.load_state_dict(torch.load(model_path))
    targetNet = QNet_LSTM(obs_shape)
    targetNet.to(device)
    targetNet.load_state_dict(behaviourNet.state_dict())
    optimizer = torch.optim.Adam(behaviourNet.parameters(), learning_rate)
    
    score_history = []
    train_history = []
    #train_history = np.load(history_path+'.npy').tolist()

    step = 0
    score = 0

    state = env.reset()
    h, c = init_hidden()

    print("Train start")
    from .utils import LinearSchedule
    lamb_schedule = LinearSchedule(1.985e6, initial_p=0.015, final_p=0.015)
    while step < Train_max_step:
        if step % 1000 == 0:
            print(f"Training step: {step}")
        epsilon = max(0.1, 1.0 - (0.9/final_exploration_step) * step)

        action_value, (next_h, next_c) = behaviourNet.forward(torch.FloatTensor([state]).to(device), (h, c))

        # epsilon greedy
        coin = random.random()
        if coin < epsilon:
            action = random.randrange(4)
        else:
            action = action_value.argmax().item()
        
        next_state, reward, done, info = env.step(action)

        # Include aup penalty.
        lamb = lamb_schedule.value(step)
        Q_aup = aup_model(torch.FloatTensor(state).unsqueeze(0).to(device))
        # Q_aup_a = Q_aup.gather(1, torch.tensor(action))
        Q_aup_a = Q_aup[:, torch.tensor(action)]
        aup_penalty = (Q_aup_a - Q_aup[:, 0]).abs() # subtract the Q-values for action 0: inaction.
        reward -= lamb * aup_penalty

        buffer.push((torch.FloatTensor(state), action, reward, 1-done))

        score += reward
        step += 1

        if done:
            next_state = env.reset()
            next_h, next_c = init_hidden()
            score_history.append(score.item())
            score = 0
            if len(score_history)> 100:
                del score_history[0]
        
        state = next_state
        h = next_h.detach()
        c = next_c.detach()

        if step%update_frequency==0 and buffer.size() > replay_start_size:
            s_batch, a_batch, r_batch, done_batch = buffer.sample(batch_size)
            train(optimizer, behaviourNet, targetNet, s_batch, a_batch, r_batch, done_batch, aup_model)

        if step % update_interval==0 and buffer.size() > replay_start_size:
            targetNet.load_state_dict(behaviourNet.state_dict())

        if step % save_interval == 0 and len(score_history) > 0:
            train_history.append(mean(score_history))
            torch.save(behaviourNet.state_dict(), model_path)
            np.save(history_path, np.array(train_history))
            print("step : {}, Average score of last 100 episode : {:.1f}".format(step, mean(score_history)))
    
    torch.save(behaviourNet.state_dict(), model_path)
    np.save(history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))

if __name__ == "__main__":
    main()