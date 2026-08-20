"""Compact PPO for the discrete-acceleration planning task."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        for layer in (self.actor, self.critic):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        features = self.body(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, obs, deterministic=False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class RunningNorm:
    """Welford mean/variance over the observation vector.

    The policy input concatenates raw kinematics with belief features whose
    natural units differ by two orders of magnitude, and the trunk is Tanh with
    no input scaling. Without this, a small-magnitude feature is effectively
    absent from the network regardless of how informative it is. Statistics are
    frozen during an iteration and refreshed from the collected batch, so the
    behaviour policy and the update see identical inputs.

    Constant features (a mode where a belief slot is structurally zero) have
    zero variance and stay at zero rather than being amplified.
    """

    def __init__(self, dim, clip=10.0, epsilon=1e-4):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = epsilon
        self.clip = clip

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None]
        n = batch.shape[0]
        if n == 0:
            return
        mean, var = batch.mean(axis=0), batch.var(axis=0)
        delta = mean - self.mean
        total = self.count + n
        m2 = (self.var * self.count + var * n
              + delta**2 * self.count * n / total)
        self.mean = self.mean + delta * n / total
        self.var = m2 / total
        self.count = total

    def __call__(self, obs):
        # A structurally constant feature keeps variance 0; guard the divisor
        # only where there is genuine spread so those slots stay exactly zero.
        scale = np.sqrt(np.where(self.var > 1e-12, self.var, 1.0))
        out = (np.asarray(obs, dtype=np.float64) - self.mean) / scale
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.actions, self.logprobs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, obs, action, logprob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.obs)


class PPO:
    def __init__(self, obs_dim, n_actions, device, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, batch_size=256, entropy_coef=0.01, value_coef=0.5,
                 normalize_obs=True):
        self.device = device
        self.policy = ActorCritic(obs_dim, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.batch_size = epochs, batch_size
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.obs_norm = RunningNorm(obs_dim) if normalize_obs else None

    def normalize(self, obs):
        return obs if self.obs_norm is None else self.obs_norm(obs)

    @torch.no_grad()
    def select(self, obs, deterministic=False):
        tensor = torch.as_tensor(self.normalize(obs), dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        action, logprob, value = self.policy.act(tensor, deterministic=deterministic)
        return int(action), float(logprob), float(value)

    def _advantages(self, rewards, values, dones, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            running = delta + self.gamma * self.lam * non_terminal * running
            advantages[t] = running
            next_value = values[t]
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def update(self, buffer, last_value):
        advantages, returns = self._advantages(
            buffer.rewards, buffer.values, buffer.dones, last_value
        )
        raw_obs = np.asarray(buffer.obs)
        obs = torch.as_tensor(self.normalize(raw_obs), dtype=torch.float32,
                              device=self.device)
        actions = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(np.asarray(buffer.logprobs), dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        indices = np.arange(len(buffer))
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < 2:
                    continue
                idx = torch.as_tensor(batch, device=self.device)
                logprobs, entropy, values = self.policy.evaluate(obs[idx], actions[idx])
                ratio = torch.exp(logprobs - old_logprobs[idx])
                surrogate = ratio * advantages_t[idx]
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages_t[idx]
                policy_loss = -torch.min(surrogate, clipped).mean()
                value_loss = ((values - returns_t[idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                n_updates += 1

        for key in stats:
            stats[key] /= max(n_updates, 1)
        if self.obs_norm is not None:
            self.obs_norm.update(raw_obs)
        return stats

    def state_dict(self):
        state = {"policy": self.policy.state_dict()}
        if self.obs_norm is not None:
            state["obs_norm"] = self.obs_norm.state_dict()
        return state

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        if self.obs_norm is not None and "obs_norm" in state:
            self.obs_norm.load_state_dict(state["obs_norm"])
