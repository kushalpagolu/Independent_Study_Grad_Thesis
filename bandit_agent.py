# ============================
# bandit_agent.py  (Fixed)
# ============================
"""LinUCB with ε‑greedy exploration and tie‑break randomisation.
   Suitable for <30 samples per session.
"""
import numpy as np
import os, pickle

class LinUCBAgent:
    def __init__(self, n_actions: int = 9, dim: int = 13, alpha: float = 1.5,
                 epsilon: float = 0.3, min_epsilon: float = 0.05, decay: float = 0.995):
        self.n_actions    = n_actions
        self.dim          = dim
        self.alpha        = alpha

        # ε‑greedy params
        self.epsilon      = epsilon
        self.min_epsilon  = min_epsilon
        self.decay        = decay
        self.step_count   = 0

        # LinUCB stats
        self.A = [np.eye(dim) for _ in range(n_actions)]  # (d,d)
        self.b = [np.zeros((dim, 1)) for _ in range(n_actions)]

    # ------------------------------------------------------------------
    def predict(self, context: np.ndarray, deterministic: bool = False):
        """Return full action‑vector (discrete + 4 continuous values)."""
        # Use the full context vector (13 elements from the LSTM) to choose a discrete action.
        # This is consistent with the agent's internal dimension `dim=13`.
        disc = self._choose_action(context)
        
        # The LSTM output vector is structured as [9 discrete scores, 4 continuous values].
        # The final action vector sent to the environment requires the chosen discrete action
        # followed by the 4 continuous values.
        # These continuous values are the LAST 4 elements of the context vector.
        # We slice from the index equal to the number of discrete actions to get them.
        continuous_values = context[self.n_actions:]
        
        # Combine the chosen discrete action with the continuous values from the LSTM.
        action_vec = np.concatenate([[disc], continuous_values])
        return action_vec, None

    def _choose_action(self, ctx: np.ndarray) -> int:
        # ε‑greedy exploration
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Compute UCB scores & random tie‑break
            x = ctx.reshape(-1, 1)
            scores = []
            for A, b in zip(self.A, self.b):
                theta = np.linalg.inv(A) @ b
                mean  = float(theta.T @ x)
                var   = float(x.T @ np.linalg.inv(A) @ x)
                scores.append(mean + self.alpha * np.sqrt(var))
            max_score = max(scores)
            best = [i for i, s in enumerate(scores) if np.isclose(s, max_score)]
            action = int(np.random.choice(best))
        return action

    # ------------------------------------------------------------------
    def update(self, context: np.ndarray, discrete_action: int, reward: float):
        x = context.reshape(-1, 1)
        a = int(discrete_action)
        self.A[a] += x @ x.T
        self.b[a] += reward * x

        # decay ε each step
        self.step_count += 1
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

    # ------------------------------------------------------------------
    def get_theta_matrix(self) -> np.ndarray:
        return np.stack([(np.linalg.inv(A) @ b).flatten() for A, b in zip(self.A, self.b)])

    # ------------------------------------------------------------------
    # Persistence helpers
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(dict(A=self.A, b=self.b, eps=self.epsilon), f)

    @classmethod
    def load(cls, path: str, dim: int, n_actions: int):
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)
        agent = cls(n_actions=n_actions, dim=dim)
        agent.A, agent.b, agent.epsilon = data['A'], data['b'], data['eps']
        return agent
