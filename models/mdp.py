import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.optim as optim


class ValueFunctionApproximator(nn.Module):
    def __init__(self, state_dim):
        super(ValueFunctionApproximator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)

def continuous_fitted_value_iteration(dynamics_model, reward_fn, state_dim, action_dim, 
                                      initial_dataset, num_iterations=100, learning_rate=0.001):
    """
    Continuous Fitted Value Iteration (cFVI) implementation.

    Args:
        dynamics_model: Function representing the system dynamics \dot{x} = f(x, u).
        reward_fn: Function representing the reward A(x, u) = g(x) - h(u).
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        initial_dataset: Initial dataset containing sampled states.
        num_iterations: Number of value iteration steps.
        learning_rate: Learning rate for training the value function.

    Returns:
        value_function: Trained value function approximator.
    """
    value_function = ValueFunctionApproximator(state_dim)
    optimizer = optim.Adam(value_function.parameters(), lr=learning_rate)

    def optimal_policy(state, value_function):
        """Compute optimal policy given the current value function."""
        with torch.no_grad():
            grad_v = torch.autograd.grad(outputs=value_function(state), inputs=state, create_graph=True)[0]
            control_matrix = dynamics_model.control_matrix(state)
            optimal_action = torch.matmul(control_matrix.T, grad_v.T)
            return optimal_action

    for iteration in range(num_iterations):
        value_targets = []
        for state in initial_dataset:
            state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True)
            action = optimal_policy(state_tensor, value_function)

            # Compute reward and next state
            reward = reward_fn(state_tensor, action)
            next_state = dynamics_model(state_tensor, action)

            # Compute target value
            with torch.no_grad():
                target_value = reward + value_function(next_state)
            value_targets.append((state_tensor, target_value))

        # Update value function with supervised learning
        optimizer.zero_grad()
        loss = 0
        for state, target in value_targets:
            predicted_value = value_function(state)
            loss += nn.MSELoss()(predicted_value, target)
        loss.backward()
        optimizer.step()

        print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item()}")

    return value_function

# Example Usage (requires a dynamics model and reward function to be defined):
# Define your dynamics_model and reward_fn here, along with initial_dataset.
# value_function = continuous_fitted_value_iteration(dynamics_model, reward_fn, state_dim, action_dim, initial_dataset)



class MDP:
    def __init__(self, state_bounds, action_bounds, state_resolution, action_resolution, discount_factor, horizon, transition_function, reward_function):
        """
        Initialize the deterministic MDP.

        Args:
            state_bounds (list of tuples): [(s1_min, s1_max), (s2_min, s2_max), ...] bounds for each state dimension.
            action_bounds (list of tuples): [(a1_min, a1_max), (a2_min, a2_max), ...] bounds for each action dimension.
            state_resolution (list): Number of discrete states per dimension.
            action_resolution (list): Number of discrete actions per dimension.
            discount_factor (float): Discount factor (gamma).
            horizon (int): Planning horizon.
            transition_function (function): Deterministic transition function, f(state, action).
            reward_function (function): Reward function based on state, r(state).
        """
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.state_resolution = state_resolution
        self.action_resolution = action_resolution
        self.discount_factor = discount_factor
        self.horizon = horizon
        self.transition_function = transition_function
        self.reward_function = reward_function

        self.discretized_states = self._discretize_space(state_bounds, state_resolution)
        self.discretized_actions = self._discretize_space(action_bounds, action_resolution)

    def _discretize_space(self, bounds, resolution):
        """
        Discretize a continuous space into a grid.

        Args:
            bounds (list of tuples): Bounds for each dimension.
            resolution (list): Number of discrete points per dimension.

        Returns:
            np.array: Discretized points in the space.
        """
        grids = [np.linspace(b[0], b[1], res) for b, res in zip(bounds, resolution)]
        return np.array(np.meshgrid(*grids)).T.reshape(-1, len(bounds))

    # def value_iteration(self, )

    def value_iteration(self, tolerance=1e-3):
        """
        Perform value iteration to compute the optimal policy.

        Args:
            tolerance (float): Convergence threshold.
        
        Returns:
            dict: Optimal value function and policy.
        """
        value_function = {tuple(s): 0 for s in self.discretized_states}
        policy = {tuple(s): None for s in self.discretized_states}

        for _ in range(self.horizon):
            new_value_function = value_function.copy()
            for state in self.discretized_states:
                state = tuple(state)
                action_values = []
                for action in self.discretized_actions:
                    action = tuple(action)
                    next_state = self.transition_function(state, action)
                    reward = self.reward_function(state)
                    expected_value = reward + self.discount_factor * value_function.get(tuple(next_state), 0)
                    action_values.append((expected_value, action))
                best_value, best_action = max(action_values, key=lambda x: x[0])
                new_value_function[state] = best_value
                policy[state] = best_action

            if max(abs(new_value_function[s] - value_function[s]) for s in value_function) < tolerance:
                break
            value_function = new_value_function

        return value_function, policy
