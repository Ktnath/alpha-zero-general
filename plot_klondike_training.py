import csv
import os
import matplotlib.pyplot as plt


def main():
    log_file = os.path.join(os.getcwd(), "training_log.csv")
    if not os.path.isfile(log_file):
        raise FileNotFoundError(f"{log_file} not found")

    iterations = []
    win_rates = []
    avg_values = []

    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row["iteration"]))
            win_rates.append(float(row["win_rate"]))
            avg_values.append(float(row.get("avg_value", 0)))

    fig, ax1 = plt.subplots()

    ax1.plot(iterations, win_rates, marker="o", label="Win rate (%)", color="tab:blue")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Win rate (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(iterations, avg_values, marker="x", label="Average value", color="tab:red")
    ax2.set_ylabel("Average value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    plt.title("Klondike Training Progress")
    plt.savefig("training_plot.png")
    print("Saved plot to training_plot.png")


if __name__ == "__main__":
    main()
