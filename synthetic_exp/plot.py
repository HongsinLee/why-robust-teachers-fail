import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot synthetic experiment results.")

    parser.add_argument(
        "--result_dir",
        type=str,
        default="synthetic_exp/results",
        help="Root directory containing synthetic result folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synthetic_exp/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "at", "bad", "good"],
        help="Which result to plot. Default plots AT, bad teacher, and good teacher.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/metrics.csv",
        help="Glob pattern used to find result CSV files.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=5,
        help="Rolling average window for smoothing curves. Use 1 for raw curves.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display figures after saving.",
    )

    return parser.parse_args()


def get_plot_specs(mode):
    """Return the list of result groups to plot."""
    all_specs = {
        "at": {
            "input_subdir": "at",
            "output_name": "split_synt_at.pdf",
            "display_name": "AT",
        },
        "bad": {
            "input_subdir": "ad/bad",
            "output_name": "split_synt_bad.pdf",
            "display_name": "Bad Teacher",
        },
        "good": {
            "input_subdir": "ad/good",
            "output_name": "split_synt_good.pdf",
            "display_name": "Good Teacher",
        },
    }

    if mode == "all":
        return [all_specs["at"], all_specs["bad"], all_specs["good"]]

    return [all_specs[mode]]


def load_runs(input_dir, pattern):
    """Load result CSV files from one experiment group."""
    csv_paths = glob.glob(str(Path(input_dir) / pattern), recursive=True)

    if not csv_paths:
        print(f"Warning: no result CSV files found under '{input_dir}'.")
        return []

    runs = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)

            required_columns = {
                "Step",
                "Robust_Train_Acc",
                "Robust_Test_Acc",
                "P_Unlearnable",
            }
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns: {sorted(missing_columns)}")

            p_un = float(df["P_Unlearnable"].iloc[0])
            label = f"{round(100 * p_un)}%"

            runs.append(
                {
                    "p_un": p_un,
                    "label": label,
                    "metrics": df,
                    "path": csv_path,
                }
            )

        except Exception as e:
            print(f"Skipping {csv_path}: {e}")

    runs.sort(key=lambda run: run["p_un"])
    return runs


def smooth_curve(series, window):
    """Apply rolling-average smoothing."""
    return series.rolling(window=window, min_periods=1).mean()


def set_plot_style():
    """Use a paper-style theme when available, with a Matplotlib fallback."""
    for style in ["seaborn-v0_8-paper", "seaborn-paper", "default"]:
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def plot_runs(runs, output_path, smoothing_window, show=True):
    """Plot train and test robust accuracy curves."""
    if not runs:
        return

    set_plot_style()

    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    for run in runs:
        df = run["metrics"]
        label = run["label"]

        steps = df["Step"]
        train_acc = smooth_curve(df["Robust_Train_Acc"], smoothing_window)
        test_acc = smooth_curve(df["Robust_Test_Acc"], smoothing_window)

        axes[0].plot(steps, train_acc, label=label)
        axes[1].plot(steps, test_acc)

    axes[0].set_ylabel("Train Robust Acc (%)", fontsize=10)
    axes[0].tick_params(axis="both", which="major", labelsize=9)
    axes[0].yaxis.set_label_coords(-0.07, 0.5)
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    axes[1].set_xlabel("Training Steps", fontsize=10)
    axes[1].set_ylabel("Test Robust Acc (%)", fontsize=10)
    axes[1].tick_params(axis="both", which="major", labelsize=9)
    axes[1].yaxis.set_label_coords(-0.07, 0.5)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    max_step = max(run["metrics"]["Step"].max() for run in runs)
    if max_step <= 0:
        max_step = 1
    axes[1].set_xlim(0, max_step)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=len(runs),
        fontsize=9,
        title="Unlearnable Samples (%)",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved figure to {output_path}")

    if show:
        plt.show()

    plt.close()


def main():
    args = parse_args()

    specs = get_plot_specs(args.mode)

    for spec in specs:
        input_dir = Path(args.result_dir) / spec["input_subdir"]
        output_path = Path(args.output_dir) / spec["output_name"]

        print(f"[Plot] {spec['display_name']} from {input_dir}")

        runs = load_runs(
            input_dir=input_dir,
            pattern=args.pattern,
        )

        plot_runs(
            runs=runs,
            output_path=output_path,
            smoothing_window=args.smoothing_window,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
