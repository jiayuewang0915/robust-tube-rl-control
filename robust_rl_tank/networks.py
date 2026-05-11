import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic_model(nn.Module):
    """Q-network for the nominal model agent."""

    def __init__(self, input_state_size, input_action_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size + input_action_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, error, action):
        # The critic estimates Q(state, tracking_error, action). State and error
        # are concatenated first, then the action is injected at the second layer.
        x = F.relu(self.linear1(torch.cat([state, error], dim=1)))
        x = F.relu(self.linear2(torch.cat([x, action], dim=1)))
        return self.linear3(x)


class Actor_model(nn.Module):
    """Deterministic policy network for the nominal model agent."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, error):
        # Tanh constrains the actor to [-1, 1]; scaling to pump/input units is
        # handled outside the network.
        x = F.relu(self.linear1(torch.cat([state, error], dim=1)))
        x = F.relu(self.linear2(x))
        return torch.tanh(self.linear3(x))


class Critic_error(nn.Module):
    """Q-network for the error-compensation agent."""

    def __init__(self, input_state_size, input_state_error, input_action_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_state_size + input_state_error, hidden_size)
        self.linear2 = nn.Linear(hidden_size + input_action_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, state, error, state_error, action_error):
        # The error critic conditions on the nominal state, tracking error, and
        # model-process mismatch before evaluating a compensation action.
        x = F.relu(self.linear1(torch.cat([state, error, state_error], dim=1)))
        x = F.relu(self.linear2(torch.cat([x, action_error], dim=1)))
        x = F.relu(self.linear3(x))
        return self.linear4(x)


class Actor_error_multi(nn.Module):
    """Deterministic policy network for the error-compensation agent."""

    def __init__(self, input_state_size, input_state_error, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_state_size + input_state_error, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, error, state_error):
        # The extra state_error input lets this actor learn corrective behavior
        # that the nominal model agent cannot infer from model state alone.
        x = F.relu(self.linear1(torch.cat([state, error, state_error], dim=1)))
        x = F.relu(self.linear2(x))
        return torch.tanh(self.linear3(x))
