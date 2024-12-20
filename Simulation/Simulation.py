import json
import subprocess
import os
import yaml
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Prompt user for the year and number of runs
year = input("Enter the year you would like to predict for: ")
num_runs = int(input("Enter the number of simulation runs: "))
num_rounds = 24
# Prompt for mode
mode = input("Enter simulation mode ('in-season' or 'out-of-season'): ").strip().lower()
if mode not in ["in-season", "out-of-season"]:
    print("Invalid mode. Defaulting to 'out-of-season'.")
    mode = "out-of-season"

# accuracy results per run and overall accuracy per run
accuracies_per_run = []
overall_accuracies = []

# execture the prediction model multiple times
for i in range(num_runs):
    print(f"\n--- Run {i + 1} ---\n")

    result = subprocess.run(
        ['python', '../NeuralNetwork/Prediction_Model.py', year, mode]
    )

    print(result.stdout)

    if i == num_runs - 1:
        print("\n--- Final Run Output ---\n")
        print(result.stdout)

    with open(f"results_summary_{year}.yaml", "r") as file:
        results = yaml.safe_load(file)

    round_accuracies = [results["round_accuracies"].get(round_num, 0) for round_num in range(1, num_rounds + 1)]
    accuracies_per_run.append(round_accuracies)

    overall_accuracy = sum(round_accuracies) / num_rounds if num_rounds > 0 else 0
    overall_accuracies.append(overall_accuracy)


print("\nOverall accuracy per run (4 decimal places):")
for run_index, accuracy in enumerate(overall_accuracies, start=1):
    print(f"Run {run_index}: {accuracy:.4f}")

average_overall_accuracy = sum(overall_accuracies) / num_runs if num_runs > 0 else 0
print(f"\nAverage overall accuracy after {num_runs} runs: {average_overall_accuracy:.4f}")

ladder_file_path = f"league_standings_{year}.json"
with open(ladder_file_path, "r") as file:
    ladder_data = json.load(file)

teams = [team["team"] for team in ladder_data["teams"]]
points = [team["points"] for team in ladder_data["teams"]]
elos = [team["elo"] for team in ladder_data["teams"]]

# Ladder graph
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.bar(teams, points, label="Points", alpha=0.8, color="skyblue")
ax1.set_ylabel("Points")
ax1.set_xlabel("Teams")
ax1.set_xticks(range(len(teams)))
ax1.set_xticklabels(teams, rotation=45, ha="right")
ax1.legend(loc="upper left")
ax1.set_ylim(0, 80)

# Elo Ratings line plot
ax2 = ax1.twinx()
ax2.plot(teams, elos, color='red', label="Elo Rating", linewidth=2)
ax2.set_ylabel("Elo Rating")
ax2.legend(loc="upper right")
ax2.set_ylim(700, 2000)
peaks, _ = find_peaks(elos)
troughs, _ = find_peaks([-e for e in elos])
for peak in peaks:
    ax2.annotate(
        f"{teams[peak]}\n{int(elos[peak])}",
        xy=(peak, elos[peak]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color="green",
        fontsize=8,
        weight="bold"
    )
for trough in troughs:
    ax2.annotate(
        f"{teams[trough]}\n{int(elos[trough])}",
        xy=(trough, elos[trough]),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        color="red",
        fontsize=8,
        weight="bold"
    )

plt.title(f"Final Ladder Standings for {year}")
plt.tight_layout()
plt.savefig(f"final_ladder_standings_{year}.png")
plt.show()
plt.close()