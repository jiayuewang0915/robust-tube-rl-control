import argparse
import os
import sys
import csv
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_rl_tank import SphericalTank
from robust_rl_tank.agents import DDPGAgentError, DDPGAgentModel
from robust_rl_tank.training import train_multi_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train the two-agent DDPG controller for the spherical tank.")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-actor", default="actor-model-new.pth")
    parser.add_argument("--error-actor", default="actor-error-new.pth")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = SphericalTank(
        start_state=None,
        desired_state=np.array([3.8]),
        Ts=0.1,
        delt_d=(0.0, 0.15),
        delt_cp=(0.0, 0.15),
        max_episode_len=120,
        seed=args.seed,
        test_model=False,
    )
    agent_model = DDPGAgentModel(
        env,
        hidden_size=args.hidden_size,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        file_name=(args.model_actor, "critic-model-new.pth"),
    )
    agent_error = DDPGAgentError(
        env,
        hidden_size=args.hidden_size,
        actor_learning_rate=1e-4,
        file_name=(args.error_actor, "critic-error-new.pth"),
    )

    history = train_multi_agent(env, agent_model, agent_error, args.batch_size, args.episodes, save_dir=args.model_dir, plot=args.plot)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    history_path = data_dir / "record_training_data_new.csv"
    fieldnames = list(history.__dict__.keys())
    rows = zip(*(history.__dict__[field] for field in fieldnames))
    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    print(f"Saved checkpoints to {args.model_dir} and training history to {data_dir}")


if __name__ == "__main__":
    main()
