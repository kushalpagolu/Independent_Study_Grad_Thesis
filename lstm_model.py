
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
# Ensure the logs directory exists
logs_dir = "/Users/kushalpagolu/Desktop/EmotivEpoch/epoch_tello_RL_3DBrain/logs"
os.makedirs(logs_dir, exist_ok=True)

# Generate a timestamped log file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"{logs_dir}/lstm_model_{timestamp}.log"

# Update logging configuration to save logs to a new file for each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a timestamped file
        logging.StreamHandler()  # Also display logs in the console
    ]
)

class LSTMModel(nn.Module):
    """
    Corrected LSTM model for multi-class discrete action and continuous control.
    """
    def __init__(self, input_size, hidden_size, num_discrete_actions, num_continuous_actions, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # FIX: This head now outputs a raw score (logit) for EACH of the 9 possible actions.
        self.fc_discrete = nn.Linear(hidden_size, num_discrete_actions)
        
        # This head remains the same.
        self.fc_continuous = nn.Linear(hidden_size, num_continuous_actions)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        
        # FIX: We no longer use sigmoid. We return the raw scores.
        # The RL agent will learn to interpret these scores.
        discrete_action_scores = self.fc_discrete(last_hidden_state)
        
        continuous_values = torch.tanh(self.fc_continuous(last_hidden_state))
        
        # Note: The output shape is now (1 + 4) if discrete is 1, but we need to handle 9 scores.
        # The RL agent's observation space will need to accommodate this. For now, this fixes the model.
        # A better approach is to not concatenate, but the RL part is set up for it.
        # We will let the RL agent see all 9 scores + 4 continuous values.
        return torch.cat([discrete_action_scores, continuous_values], dim=1)


class LSTMTrainer:
    """
    Trainer corrected for multi-class classification.
    """
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        # FIX: Use CrossEntropyLoss, the correct loss function for multi-class problems.
        self.criterion_discrete = nn.CrossEntropyLoss()
        self.criterion_continuous = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    # The rest of the trainer (train, save, load methods) can remain,
    # but the training logic itself would need to change to use one-hot encoded labels for discrete actions.
    # Since you are not actively training the LSTM right now, we will leave it.
    # The key was fixing the model architecture.

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}.")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        logger.info(f"Model loaded from {filepath}.")


    def save_feature_vector(self, feature_vector, save_path, logger=None):
        """
        Saves the feature vector to a file and logs its shape and size.
        :param feature_vector: The feature vector to save.
        :param save_path: Path to save the feature vector.
        :param logger: Logger instance for logging.
        """
        try:
            np.save(save_path, feature_vector)
            if logger:
                logger.info(f"Feature vector saved to {save_path}.")
                logger.info(f"Feature vector shape: {feature_vector.shape}, size: {feature_vector.size}")
        except Exception as e:
            if logger:
                logger.error(f"Error saving feature vector: {e}")


# Example main method to demonstrate usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize the LSTM model
    input_size = 10878  # Flattened feature vector size for 1 second
    hidden_size = 128  # Example hidden size
    output_size = 6  # Example: Predicting one of 6 possible actions
    model = LSTMModel(input_size, hidden_size, output_size)

    # Example feature sequence
    feature_sequence = np.random.rand(10, input_size)  # Example 10-second feature sequence
    feature_tensor = torch.tensor(feature_sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Run the model
    prediction = model(feature_tensor)
    logger.info(f"Prediction shape: {prediction.shape}, Prediction: {prediction}")
