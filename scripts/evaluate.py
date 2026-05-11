import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Allow running `python scripts/evaluate.py` directly from the repository root
# without requiring the package to be installed first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_rl_tank.evaluation import evaluate_from_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained robust RL tank controllers.")
    parser.add_argument("--model-dir", default="models", help="Directory containing actor checkpoints.")
    parser.add_argument("--model-actor", default="default_model_agent.pth", help="Model-agent actor checkpoint.")
    parser.add_argument("--error-actor", default="default_error_agent.pth", help="Error-agent actor checkpoint.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV and figures.")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving PNG figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_from_checkpoints(args.model_dir, args.model_actor, args.error_actor)
    trajectory_path = output_dir / "setpoint_tracking.csv"
    # Save the raw trajectory so figures can be regenerated or inspected later.
    with trajectory_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setpoint", "robust_rl", "standard_ddpg", "action_model", "action_error", "action_total"])
        writer.writerows(
            zip(
                results["setpoints"],
                results["robust_rl"],
                results["standard_ddpg"],
                results["actions_model"],
                results["actions_error"],
                results["actions_total"],
            )
        )

    if not args.no_plot:
        # Matplotlib is imported lazily so `--no-plot` evaluation only needs
        # PyTorch and NumPy.
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(results["setpoints"], color="gray", linestyle="--", linewidth=2.0)
        ax.plot(results["standard_ddpg"], color="darkgreen", linewidth=2.0)
        ax.plot(results["robust_rl"], color="blue", linewidth=2.0)
        ax.set_ylim(0, 4)
        ax.set_xlabel("Time [0.1s]")
        ax.set_ylabel("Fluid Level [m]")
        ax.legend(["Set Points", "Standard DDPG", "Robust RL"], loc="lower right")
        ax.grid()
        fig.tight_layout()
        fig.savefig(output_dir / "setpoint_tracking.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(results["actions_model"], color="blue", linewidth=2.0)
        ax.plot(results["actions_error"], color="darkgreen", linewidth=2.0)
        ax.plot(results["actions_total"], color="gray", linewidth=2.0)
        ax.set_xlabel("Time [0.1s]")
        ax.set_ylabel("Input Value")
        ax.legend(["Model Agent", "Error Agent", "Total"], loc="lower right")
        ax.grid()
        fig.tight_layout()
        fig.savefig(output_dir / "control_actions.png", dpi=200)
        plt.close(fig)

    print(f"Wrote evaluation results to {trajectory_path}")


if __name__ == "__main__":
    main()
