from pathlib import Path

import torch
import numpy as np

from robust_rl_tank.networks import Actor_error_multi, Actor_model
from robust_rl_tank.scaling import combine_scaled_actions, scale_error, scale_state, unscale_action


def load_actor(path):
    """Load an actor checkpoint saved either as a module or as a state dict.

    Default checkpoints are state dictionaries. The module fallback is kept so
    externally trained models from older PyTorch workflows can still be inspected.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if hasattr(checkpoint, "forward"):
        return checkpoint

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {path}")

    input_size = state_dict["linear1.weight"].shape[1]
    hidden_size = state_dict["linear1.weight"].shape[0]
    output_size = state_dict["linear3.weight"].shape[0]

    # The nominal actor consumes [state, tracking_error]. The error actor consumes
    # [state, tracking_error, model_process_error], so the first-layer width
    # identifies which architecture should be reconstructed.
    if input_size == 2:
        actor = Actor_model(input_size, hidden_size, output_size)
    elif input_size == 3:
        actor = Actor_error_multi(2, 1, hidden_size, output_size)
    else:
        raise ValueError(f"Cannot infer actor type from first layer input size {input_size}: {path}")

    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def run_setpoint_tracking(
    actor_model,
    actor_error,
    setpoints=(3.0, 0.5, 2.0, 3.3, 1.0, 3.5),
    steps_per_setpoint=100,
    initial_start=np.array([1.0]),
    Ts=0.1,
    gravity_constant=9.81,
    cp=1.30,
    un_value_cp=0.0,
    radius=2.0,
    un_value_d=0.2,
):
    """Run the paper-style setpoint-tracking evaluation with fixed uncertainty."""
    states = [setpoint for setpoint in setpoints for _ in range(steps_per_setpoint)]
    state_real = np.asarray(initial_start, dtype=float)
    state_model = np.asarray(initial_start, dtype=float)
    state_test = np.asarray(initial_start, dtype=float)
    state_er = state_real - state_model
    error = np.array([states[0]]) - state_real

    states_real = []
    states_test = []
    actions_model = []
    actions_error = []
    total_actions = []

    for state in states:
        desired_state = np.array([state])
        # Match the normalization used during training before querying the actors.
        scaled_state = torch.from_numpy(scale_state(state_model)).float().unsqueeze(0)
        scaled_error = torch.from_numpy(scale_error(error)).float().unsqueeze(0)
        scaled_error_state = torch.from_numpy(scale_error(state_er)).float().unsqueeze(0)

        action_scaled = actor_model.forward(scaled_state, scaled_error).detach().numpy()[0, 0]
        action_er_scaled = actor_error.forward(scaled_state, scaled_error, scaled_error_state).detach().numpy()[0, 0]

        action = unscale_action(action_scaled)
        action_er = unscale_action(action_er_scaled)
        action_sum = combine_scaled_actions(action_scaled, action_er_scaled)

        # Evaluate three trajectories side by side:
        # robust RL real process, standard DDPG baseline, and nominal model.
        state_real = state_real + Ts * (action_sum - (cp + un_value_cp) * np.sqrt(2 * gravity_constant * state_real)) / (
            np.pi * (2 * (radius + un_value_d) * state_real - state_real**2)
        )
        state_test = state_test + Ts * (action - (cp + un_value_cp) * np.sqrt(2 * gravity_constant * state_test)) / (
            np.pi * (2 * (radius + un_value_d) * state_test - state_test**2)
        )
        state_model = state_model + Ts * (action - cp * np.sqrt(2 * gravity_constant * state_model)) / (
            np.pi * (2 * radius * state_model - state_model**2)
        )
        state_er = state_real - state_model
        error = desired_state - state_model

        states_real.append(float(state_real[0]))
        states_test.append(float(state_test[0]))
        actions_model.append(float(action))
        actions_error.append(float(action_er))
        total_actions.append(float(action_sum))

    return {
        "setpoints": np.asarray(states, dtype=float),
        "robust_rl": np.asarray(states_real, dtype=float),
        "standard_ddpg": np.asarray(states_test, dtype=float),
        "actions_model": np.asarray(actions_model, dtype=float),
        "actions_error": np.asarray(actions_error, dtype=float),
        "actions_total": np.asarray(total_actions, dtype=float),
    }


def evaluate_from_checkpoints(model_dir="models", model_name="default_model_agent.pth", error_name="default_error_agent.pth"):
    model_dir = Path(model_dir)
    actor_model = load_actor(model_dir / model_name)
    actor_error = load_actor(model_dir / error_name)
    return run_setpoint_tracking(actor_model, actor_error)
