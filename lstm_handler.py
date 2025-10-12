import os
import torch
from lstm_model import LSTMModel, LSTMTrainer
import logging
import numpy as np

logger = logging.getLogger(__name__)

class LSTMHandler:
    def __init__(self):
        # Adjusted to match 10-second EEG sequences (flattened feature vector size: 10878)
        input_size = 10878  # Flattened feature vector size for 1 second
        hidden_size = 128  # Hidden layer size
        num_discrete_actions = 9  # Example: Predicting one of 9 possible actions discrete_action (integer in [0, 8] for action type), cont1, cont2, cont3, cont4
        num_continuous_actions = 4 # For lr, fb, ud, yaw

        num_layers = 2

        self.model = LSTMModel(input_size, hidden_size, num_discrete_actions, num_continuous_actions, num_layers, dropout=0.3)
        self.trainer = LSTMTrainer(self.model)

        # Load pre-trained model if available
        model_path = os.path.join(os.getcwd(), "models", "lstm_model.pth")
        if os.path.exists(model_path):
            self.trainer.load_model(model_path)
            logger.info("Loaded pre-trained LSTM model.")
        else:
            logger.warning("No pre-trained LSTM model found. Using untrained model.")


    def predict(self, feature_sequence):
        """
        Predicts an action based on a 10-second sequence of EEG features.

        This method takes the feature sequence, formats it for the LSTM model,
        and returns the raw model output. The raw output is what the RL agent
        will use as its observation.

        :param feature_sequence: A NumPy array of shape (10, 10878), representing
                                 10 time-steps of flattened EEG features.
        :return: A 1D NumPy array containing the raw output from the model,
                 or None if an error occurs.
        """
        try:
            # 1. Validate the input shape.
            expected_shape = (10, 10878)
            if feature_sequence.shape != expected_shape:
                raise ValueError(f"Expected input shape {expected_shape}, but got {feature_sequence.shape}")

            # 2. Convert to a PyTorch tensor and add the batch dimension.
            # The shape changes from (10, 10878) to (1, 10, 10878)
            # which is (batch_size, sequence_length, input_size).
            feature_tensor = torch.tensor(feature_sequence, dtype=torch.float32).unsqueeze(0)

            # 3. Get the raw prediction from the model.
            # We use .detach() to remove it from the computation graph,
            # .numpy() to convert to a NumPy array, and .flatten() to make it 1D.
            raw_output = self.model(feature_tensor).detach().numpy().flatten()
            
            return raw_output

        except (RuntimeError, ValueError) as e:
            logger.error(f"Error during LSTM prediction: {e}")
            return None

    def named_parameters(self):
        """
        Expose the named parameters of the underlying LSTM model.
        :return: Named parameters of the model.
        """
        return self.model.named_parameters()
