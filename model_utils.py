import os
import logging
from bandit_agent import LinUCBAgent

# Bandit model will be persisted as <MODEL_FILENAME>.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(BASE_DIR, "models", "drone_rl_eeg_bandit")

def load_or_create_model(env=None):
    """Return an online LinUCB contextual‑bandit agent.

    `env` is unused but kept so existing callers don’t break.
    """
    logger = logging.getLogger(__name__)
    dim       = 5   # length of LSTM output vector
    n_actions = 9   # discrete commands 0‑8
    try:
        if os.path.exists(f"{MODEL_FILENAME}.pkl"):
            model = LinUCBAgent.load(MODEL_FILENAME, dim=dim, n_actions=n_actions)
            logger.info("Loaded existing LinUCB bandit model")
        else:
            model = LinUCBAgent(n_actions=n_actions, dim=dim, alpha=1.5)
            logger.info("Created new LinUCB bandit model")
    except Exception as e:
        logger.error(f"Error initializing bandit model: {e}")
        raise
    return model

def log_model_parameters(model):
    logger = logging.getLogger(__name__)
    if model is not None:
        logger.info("Logging model parameters:")
        for name, param in model.policy.named_parameters():
            logger.info(f"Parameter {name}: {param.data}")

def save_model_parameters(model, filename="model_parameters.txt"):
    if model is not None:
        with open(filename, "w") as f:
            for name, param in model.policy.named_parameters():
                f.write(f"Parameter {name}: {param.data}\n")
        logging.info(f"Model parameters saved to {filename}")
