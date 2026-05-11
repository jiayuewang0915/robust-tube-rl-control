from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robust_rl_tank import SphericalTank
from robust_rl_tank.scaling import scale_error, scale_state


def test_environment_reset_and_step():
    env = SphericalTank(start_state=np.array([1.0]), desired_state=np.array([2.0]), delt_d=(0.0, 0.15), delt_cp=(0.0, 0.15))
    state_model, error, state_er, state_real, done = env.reset(seed=1)
    assert state_model.shape == (1,)
    assert error.shape == (1,)
    assert state_er.shape == (1,)
    assert state_real.shape == (1,)
    assert done is False

    output = env.step_uncertainty(action=5.0, action_sum=5.0)
    assert len(output) == 8


def test_scaling_matches_expected_ranges():
    assert np.allclose(scale_state(np.array([0.0, 4.0])), np.array([-1.0, 1.0]))
    assert np.allclose(scale_error(np.array([-4.0, 4.0])), np.array([-1.0, 1.0]))


def test_pretrained_evaluation_smoke():
    if not Path("models/default_model_agent.pth").exists() or not Path("models/default_error_agent.pth").exists():
        pytest.skip("Pretrained paper checkpoints are not available.")

    subprocess.run([sys.executable, "scripts/evaluate.py", "--no-plot", "--output-dir", "results/test_smoke"], check=True)
    output = Path("results/test_smoke/setpoint_tracking.csv")
    assert output.exists()
    assert len(output.read_text().splitlines()) == 601
