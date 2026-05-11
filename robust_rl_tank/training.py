from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from robust_rl_tank.agents import OUActionNoise
from robust_rl_tank.scaling import combine_scaled_actions, scale_error, scale_state, unscale_action


@dataclass
class TrainingHistory:
    """Episode-level metrics saved for plotting and reproducibility."""

    cum_rewards_model: list
    cum_rewards_error: list
    cum_rewards_real: list
    policy_loss_error: list
    policy_loss_model: list
    critic_loss_error: list
    critic_loss_model: list
    last_states_model: list
    last_states_error: list
    last_states_real: list
    last_errors: list


def train_multi_agent(env, agent_model, agent_error, batch_size, episodes, save_dir="models", plot=False):
    """Train the nominal model agent and the error-compensation agent together."""
    history = TrainingHistory([], [], [], [], [], [], [], [], [], [], [])
    actions_model = []
    actions_error = []
    actions_total = []
    policy_loss_model = critic_loss_model = policy_loss_error = critic_loss_error = 0.0

    ou_noise = OUActionNoise(mu=np.zeros(env.action_space.shape[0]), Ts=0.1, sigma=0.001, theta=0.15, dt=1e-4)
    ou_noise.reset()

    progress = trange(episodes, desc="Training", leave=True)
    for episode in progress:
        state_model, error, state_er, state_real, done = env.reset()
        cum_reward_model = 0.0
        cum_reward_error = 0.0
        cum_reward_real = 0.0
        t = 0

        while not done:
            # Actors are trained on normalized inputs. The model agent sees nominal
            # state and setpoint error; the error agent additionally sees the
            # mismatch between nominal and real process states.
            scaled_error_state = scale_error(state_er)
            scaled_error = scale_error(error)
            scaled_state = scale_state(state_model)

            # Small Gaussian exploration noise is added in the actor output space.
            action_scaled = agent_model.get_action(scaled_state, scaled_error) + np.random.normal(loc=0.0, scale=0.01)
            action_er_scaled = agent_error.get_action(scaled_state, scaled_error, scaled_error_state) + np.random.normal(loc=0.0, scale=0.01)
            action = unscale_action(action_scaled, env.action_space.low[0], env.action_space.high[0])
            action_er = unscale_action(action_er_scaled, env.action_space.low[0], env.action_space.high[0])
            # The plant receives the combined robust action; the nominal model
            # receives only the model-agent action to expose process mismatch.
            action_sum = combine_scaled_actions(action_scaled, action_er_scaled, env.action_space.low[0], env.action_space.high[0])

            next_state_model, next_error, next_state_er, next_state_real, reward_state, reward_error, reward_real, done = env.step_uncertainty(
                action, action_sum
            )

            scaled_next_state = scale_state(next_state_model)
            scaled_next_error = scale_error(next_error)
            scaled_next_error_state = scale_error(next_state_er)

            # Store separate transition formats because the two critics estimate
            # different objectives and observe different state variables.
            agent_model.memory.append((scaled_state, scaled_error, action_scaled, reward_state, scaled_next_state, scaled_next_error, not done))
            agent_error.memory.append(
                (
                    scaled_state,
                    scaled_error,
                    scaled_error_state,
                    action_er_scaled,
                    reward_error,
                    scaled_next_state,
                    scaled_next_error,
                    scaled_next_error_state,
                    not done,
                )
            )

            cum_reward_model += float(reward_state[0])
            cum_reward_error += float(reward_error[0])
            cum_reward_real += float(reward_real[0])

            if len(agent_model.memory) > batch_size:
                # Both agents update once enough shared experience has accumulated.
                policy_loss_model, critic_loss_model = agent_model.update(batch_size)
                policy_loss_error, critic_loss_error = agent_error.update(batch_size)

            state_model = next_state_model
            error = next_error
            state_er = next_state_er
            state_real = next_state_real
            t += 1

        history.last_states_model.append(float(state_model[0]))
        history.last_errors.append(float(error[0]))
        history.last_states_error.append(float(state_er[0]))
        history.last_states_real.append(float(state_real[0]))
        actions_model.append(action)
        actions_error.append(action_er)
        actions_total.append(action_sum)
        history.cum_rewards_model.append(cum_reward_model)
        history.cum_rewards_error.append(cum_reward_error)
        history.cum_rewards_real.append(cum_reward_real)
        history.policy_loss_model.append(float(policy_loss_model))
        history.critic_loss_model.append(float(critic_loss_model))
        history.policy_loss_error.append(float(policy_loss_error))
        history.critic_loss_error.append(float(critic_loss_error))

        progress.set_description(
            f"episode: {episode} terminated after {t} steps, "
            f"model state {history.last_states_model[-1]:.3f}, error {history.last_errors[-1]:.3f}, "
            f"state_error {history.last_states_error[-1]:.3f}"
        )

    agent_model.save(save_dir)
    agent_error.save(save_dir)

    if plot:
        plot_training_history(history, actions_model, actions_error, actions_total)
    return history


def plot_training_history(history, actions_model=None, actions_error=None, actions_total=None):
    """Create the three diagnostic plots used to inspect training behavior."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 3))
    ax[0].plot(history.cum_rewards_model)
    ax[0].plot(history.cum_rewards_error)
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[0].legend(["Model Agent", "Error Agent"])
    ax[0].grid()

    ax[1].plot(history.last_states_error)
    ax[1].plot(history.last_errors)
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Last State")
    ax[1].legend(["Error Agent", "Error to Setpoint"])
    ax[1].grid()

    if actions_model is not None:
        ax[2].plot(actions_model)
        ax[2].plot(actions_error)
        ax[2].plot(actions_total)
        ax[2].set_xlabel("Episode")
        ax[2].set_ylabel("Action")
        ax[2].legend(["Model Agent", "Error Agent", "Total"])
        ax[2].grid()
    return fig
