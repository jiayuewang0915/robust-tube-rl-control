import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def mean_bounds(values):
    values = np.asarray(values)
    return values.mean(axis=0), values.max(axis=0), values.min(axis=0)


def batch_columns(df, batch_size, columns):
    first, second = [], []
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        first.append(df.loc[start : end - 1, columns[0]].values)
        second.append(df.loc[start : end - 1, columns[1]].values)
    return first, second


def main():
    parser = argparse.ArgumentParser(description="Plot the training return curves from the saved CSV.")
    parser.add_argument("--csv", default="data/record_training_data.csv")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--output", default="results/training_returns.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    model_returns, error_returns = batch_columns(df, args.batch_size, ["cum_rewards_model", "cum_rewards_error"])
    model_mean, model_upper, model_lower = mean_bounds(model_returns)
    error_mean, error_upper, error_lower = mean_bounds(error_returns)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(model_mean, color="teal")
    ax.plot(error_mean, color="brown")
    ax.fill_between(range(len(model_mean)), model_lower, model_upper, color="teal", alpha=0.2)
    ax.fill_between(range(len(error_mean)), error_lower, error_upper, color="brown", alpha=0.2)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Return")
    ax.legend(["Model Agent", "Error Agent"], loc="lower right")
    ax.grid()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
