import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BehaviorCloningLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=1):
        """
        An LSTM-based network for behavior cloning. The network processes an entire sequence of states
        and outputs a sequence of predicted actions.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            hidden_dim (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
        """
        super(BehaviorCloningLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x is expected to be of shape (batch_size, sequence_length, state_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, sequence_length, hidden_dim)
        predictions = self.fc(lstm_out)  # predictions: (batch_size, sequence_length, action_dim)
        return predictions


def load_expert_data(filepath):
    """
    Loads expert trajectory data from an NPZ file.

    The NPZ file is expected to contain two arrays:
    - 'states': A NumPy array of shape (num_trajectories, sequence_length, state_dim)
    - 'actions': A NumPy array of shape (num_trajectories, sequence_length, action_dim)

    Args:
        filepath (str): Path to the NPZ file.

    Returns:
        states (np.ndarray), actions (np.ndarray)
    """
    data = np.load(filepath)
    states = data['states']
    actions = data['actions']
    return states, actions


if __name__ == '__main__':
    # Define hyperparameters.
    state_dim = 10     # Number of state features. Adjust based on your problem.
    action_dim = 4     # Number of control outputs.
    hidden_dim = 64    # Hidden dimension for LSTM.
    num_layers = 1     # Number of LSTM layers.
    batch_size = 32    # Adjust batch size based on your dataset.
    num_epochs = 50
    learning_rate = 1e-3

    # Load expert data.
    # Ensure that the data is arranged as sequences: (num_trajectories, sequence_length, state_dim)
    states, actions = load_expert_data('expert_data.npz')

    # Convert the NumPy arrays to PyTorch tensors.
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)

    # Create a dataset and DataLoader for batching.
    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the LSTM-based model.
    model = BehaviorCloningLSTM(state_dim, action_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    # Set up the loss criterion and optimizer.
    criterion = nn.MSELoss()  # Use MSE for continuous actions; use an alternative loss for categorical actions.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting LSTM behavior cloning training ...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()  # Clear gradients from the previous step.

            # Forward pass: the model returns a sequence of predicted actions.
            predictions = model(batch_states)
            loss = criterion(predictions, batch_actions)

            # Backward pass and optimization.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # Save the trained model.
    torch.save(model.state_dict(), 'behavior_cloning_lstm_model.pth')
    print("Training complete. Model saved as 'behavior_cloning_lstm_model.pth'")
