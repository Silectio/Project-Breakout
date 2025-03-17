import torch
import numpy as np
import random

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4, device='cpu'):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def store(self, transition, priority=1.0):
        """
        transition: (state, action, reward, next_state, done)
          - Ici, state, next_state peuvent être des np.array
          - action, reward, done peuvent être des scalars
        """
        # Conversion en tenseurs PyTorch (même logique que SumTreeReplayMemory)
        state, action, reward, next_state, done = transition
        
        state_t      = torch.tensor(state,      device=self.device, dtype=torch.float32)
        action_t     = torch.tensor(action,     device=self.device, dtype=torch.long)
        reward_t     = torch.tensor(reward,     device=self.device, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        done_t       = torch.tensor(done,       device=self.device, dtype=torch.float32)
        
        transition_gpu = (state_t, action_t, reward_t, next_state_t, done_t)

        # On élève la priorité à la puissance alpha
        prio = (priority ** self.alpha)

        # Insertion dans la mémoire
        if len(self.memory) < self.capacity:
            self.memory.append(transition_gpu)
        else:
            self.memory[self.pos] = transition_gpu

        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        # On ne sample que parmi les "vrais" éléments (si memory non pleine)
        actual_size = len(self.memory)
        if actual_size == 0:
            return None, None, None

        # On récupère toutes les priorités existantes
        prios = self.priorities[:actual_size]
        probs = prios / prios.sum()

        # Choix d'indices selon ces probabilités
        indices = np.random.choice(actual_size, batch_size, p=probs)
        
        # Récupère les transitions correspondantes
        batch = [self.memory[i] for i in indices]

        # MàJ beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Calcul des weights
        # w_i = (1/(P(i)))^beta  normalisé par le max
        weights = (batch_size * probs[indices]) ** (-self.beta)
        if weights.size > 0:
            weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = (prio ** self.alpha)

    def __len__(self):
        return len(self.memory)
