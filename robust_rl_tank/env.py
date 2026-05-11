from typing import Any

import numpy as np


class Box:
    """Small subset of gymnasium.spaces.Box used by the training code.

    The original experiment used Gym/Gymnasium-style spaces. This local class
    keeps the public fields that the RL code needs (`low`, `high`, `shape`, and
    `sample`) without making the project depend on a particular Gym version.
    """

    def __init__(self, low, high, dtype=np.float32):
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.dtype = dtype
        self.shape = self.low.shape
        self._rng = np.random.default_rng()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(self.dtype)


class SphericalTank:
    """Spherical tank level-control environment with parametric uncertainty.

    The environment evolves three related trajectories:
    - `cur_location`: the nominal model state used by the model agent.
    - `cur_state_real`: the uncertain physical process controlled by both agents.
    - `cur_state_test`: a baseline uncertain process driven only by the model agent
      during evaluation.

    State variables are one-dimensional arrays because the tank level is scalar,
    while preserving the tensor shape expected by the DDPG implementation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cp=1.3,
        radius=2.0,
        start_state=None,
        desired_state=np.array([1.0]),
        Ts=0.1,
        delt_d=(0.0, 0.0),
        delt_cp=(0.0, 0.0),
        render_mode=None,
        max_episode_len=120,
        seed=None,
        test_model=False,
    ) -> None:
        self.action_space = Box(low=np.array([0.0]), high=np.array([15.0]), dtype=np.float32)
        self.action_error_space = Box(low=np.array([-5.0]), high=np.array([5.0]), dtype=np.float32)
        self.observation_space = Box(low=np.array([-4.0]), high=np.array([4.0]), dtype=np.float32)
        self.cur_location_space = Box(low=np.array([0.0]), high=np.array([4.0]), dtype=np.float32)
        self.sample_location_space = Box(low=np.array([1.5]), high=np.array([2.5]), dtype=np.float32)
        self.observation_space.seed(seed=seed)

        self.start_state = start_state
        self.desired_state = np.asarray(desired_state, dtype=float)
        self.max_episode_len = max_episode_len
        self.render_mode = render_mode
        self.test_mode = test_model
        self.gravity_constant = 9.81
        self.radius = radius
        self.cp = cp
        self.Ts = Ts
        self.noise_d_mu, self.noise_d_sigma = delt_d
        self.noise_cp_mu, self.noise_cp_sigma = delt_cp
        # Curriculum-like setpoint sampling from the original experiments.
        self.sample_list = [[0.1, 0.5], [0.5, 2.0], [2.0, 3.5], [3.5, 3.9]]
        self.reset_time = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.observation_space.seed(seed=seed)
            self.sample_location_space.seed(seed=seed)
        if self.test_mode:
            # Evaluation uses fixed uncertainty values so that runs are comparable.
            self.un_value_d = self.noise_d_sigma
            self.un_value_cp = self.noise_cp_sigma
        else:
            # Training samples setpoints across different operating regions. Later
            # episodes emphasize higher levels, where nonlinear effects are stronger.
            if self.reset_time <= 150:
                num = np.random.choice([0, 1, 2, 3], p=[0.3, 0.2, 0.2, 0.3], size=1)
            else:
                num = np.random.choice([0, 1, 2, 3], p=[0.1, 0.1, 0.5, 0.3], size=1)
            sample_range = self.sample_list[num.item()]
            self.desired_state = np.random.uniform(low=sample_range[0], high=sample_range[1], size=1)
            self.un_value_d = 0.0
            self.un_value_cp = 0.0

        if self.start_state is None:
            self.cur_location = self.sample_location_space.sample().astype(float)
        else:
            self.cur_location = np.asarray(self.start_state, dtype=float)

        self.cur_state_real = self.cur_location.copy()
        self.cur_state_test = self.cur_location.copy()
        # `error_cur_state` is the model-process mismatch observed by the error
        # agent. `cur_state` is the nominal tracking error to the desired level.
        self.error_cur_state = self.cur_state_real - self.cur_location
        self.cur_state = self.desired_state - self.cur_location
        self.timestep = 0
        self.reset_time += 1
        return self.cur_location, self.cur_state, self.error_cur_state, self.cur_state_real, False

    def step_uncertainty(self, action: Any, action_sum: Any):
        """Advance one training step with newly sampled physical uncertainty."""
        self.un_value_d = np.clip(
            np.random.normal(loc=self.noise_d_mu, scale=self.noise_d_sigma),
            self.desired_state / 2.0 - self.radius,
            1.0,
        )
        self.un_value_cp = np.random.normal(loc=self.noise_cp_mu, scale=self.noise_cp_sigma)
        return self._step_dynamics(action, action_sum, terminate_on_violation=True)

    def step_uncertianty(self, action: Any, action_sum: Any):
        """Backward-compatible alias for the original misspelled method name."""
        return self.step_uncertainty(action, action_sum)

    def test(self, action: Any, action_sum: Any):
        """Advance one evaluation step for robust and baseline controllers."""
        self.cur_location = np.asarray(self.cur_location).reshape(self.cur_location_space.shape)
        # Robust controller: real uncertain process receives the combined action.
        self.cur_state_real = self._next_level(self.cur_state_real, action_sum, self.radius + self.un_value_d, self.cp + self.un_value_cp)
        # Baseline controller: uncertain process receives only the model-agent action.
        self.cur_state_test = self._next_level(self.cur_state_test, action, self.radius + self.un_value_d, self.cp + self.un_value_cp)
        # Nominal simulator remains uncertainty-free and supplies the model error.
        self.cur_location = self._next_level(self.cur_location, action, self.radius, self.cp)
        self.error_cur_state = self.cur_state_real - self.cur_location
        self.cur_state = self.desired_state - self.cur_location
        done = self.done_test()
        reward_state, reward_error, reward_real, reward_test = self.reward_fun(action)
        self.timestep += 1
        return (
            self.cur_location,
            self.error_cur_state,
            self.cur_state,
            self.cur_state_real,
            self.cur_state_test,
            reward_state,
            reward_error,
            reward_real,
            reward_test,
            done,
        )

    def _step_dynamics(self, action, action_sum, terminate_on_violation):
        self.cur_location = np.asarray(self.cur_location).reshape(self.cur_location_space.shape)
        # During training, the real process follows the combined action while the
        # nominal model follows only the model-agent action.
        self.cur_state_real = self._next_level(self.cur_state_real, action_sum, self.radius + self.un_value_d, self.cp + self.un_value_cp)
        self.cur_location = self._next_level(self.cur_location, action, self.radius, self.cp)
        self.error_cur_state = self.cur_state_real - self.cur_location
        self.cur_state = self.desired_state - self.cur_location
        done = self.done() if terminate_on_violation else self.done_test()
        reward_state, reward_error, reward_real, _ = self.reward_fun(action)
        self.timestep += 1
        return self.cur_location, self.cur_state, self.error_cur_state, self.cur_state_real, reward_state, reward_error, reward_real, done

    def _next_level(self, state, action, radius, cp):
        """Forward-Euler discretization of the spherical tank dynamics."""
        state = np.asarray(state, dtype=float)
        denominator = np.pi * (2.0 * radius * state - state**2)
        return state + self.Ts * (action - cp * np.sqrt(2.0 * self.gravity_constant * state)) / denominator

    def done(self):
        return self.timestep >= self.max_episode_len or self.terminated_model()

    def done_test(self):
        return self.timestep >= self.max_episode_len

    def reward_fun(self, action):
        # Rewards are negative absolute errors, matching the paper's tracking and
        # compensation objectives: high reward means smaller magnitude error.
        reward_state = -np.abs(self.cur_state)
        reward_error = -np.abs(self.error_cur_state)
        reward_real = -np.abs(self.desired_state - self.cur_state_real)
        reward_test = -np.abs(self.desired_state - self.cur_state_test)
        return reward_state, reward_error, reward_real, reward_test

    def state_space_violation(self):
        penalty = 0.0
        for i in range(len(self.cur_location)):
            penalty += (self.cur_location[i] - self.cur_location_space.high[i]) * float(self.cur_location[i] > self.cur_location_space.high[i])
            penalty += (self.cur_location_space.low[i] - self.cur_location[i]) * float(self.cur_location[i] < self.cur_location_space.low[i])
        return penalty

    def terminated_model(self):
        return (
            self.cur_location[0] > 4.0
            or self.cur_location[0] < 0.0
            or self.cur_state_real[0] > 2.0 * (self.radius + self.un_value_d)
            or self.cur_state_real[0] < 0.0
        )

    def terminated_test(self):
        return self.terminated_model() or self.cur_state_test[0] > 2.0 * (self.radius + self.un_value_d) or self.cur_state_test[0] < 0.0


SphericalTank_MultiAgent = SphericalTank
