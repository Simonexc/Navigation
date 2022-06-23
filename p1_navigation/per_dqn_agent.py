from dqn_agent import BATCH_SIZE, Agent, BUFFER_SIZE, GAMMA, UPDATE_EVERY, TAU, device
from SumTree import SumTree
import random
import numpy as np
import torch.nn.functional as F
import torch


E = 0.01


class PERDQNAgent(Agent):
    def __init__(self, state_size, action_size, seed, lr=5e-4, a=0.6, b=0.4, b_inc=0.001):
        super().__init__(state_size, action_size, seed, lr)
        self.memory = PERBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, a, b, b_inc)

    def step(self, *experience):
        # Save experience in replay memory
        torch_experience = self.memory.convert_experiences_to_torch([experience])
        Q_expected, Q_targets = self.calculate_Q_values(torch_experience, GAMMA)
        error = (Q_expected - Q_targets).detach().cpu().numpy().reshape(-1)[0]
        self.memory.add(error, experience)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.memory.n_entries > BATCH_SIZE:
                experiences, idxs, is_weights = self.memory.sample()
                self.learn((experiences, idxs, is_weights), GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        experiences, idxs, is_weights = experiences
        Q_expected, Q_targets = self.calculate_Q_values(experiences, GAMMA)
        # update priority
        errors = (Q_expected - Q_targets).detach().cpu().numpy().reshape(-1)
        for i in range(BATCH_SIZE):
            self.memory.update(idxs[i], errors[i])

        # Compute loss
        loss = (is_weights * F.mse_loss(Q_expected, Q_targets)).mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(TAU)


class PERBuffer:
    """
    Prioritized Experience Replay Buffer
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, a=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = SumTree(buffer_size)
        self.batch_size = batch_size
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        random.seed(seed)

    def _get_priority(self, error):
        return (np.abs(error) + E) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.memory.add(p, sample)

    def convert_experiences_to_torch(self, experiences):
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones

    def sample(self):
        batch = []
        idxs = []
        segment = self.memory.total() / BATCH_SIZE
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(BATCH_SIZE):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.memory.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.memory.total()
        is_weight = np.power(self.memory.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()  # normalize

        batch = self.convert_experiences_to_torch(batch)
        is_weight = torch.from_numpy(is_weight).float().to(device)

        return batch, idxs, is_weight  # importance sampling weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.memory.update(idx, p)
