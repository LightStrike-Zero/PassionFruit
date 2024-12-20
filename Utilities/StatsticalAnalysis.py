import json
import numpy as np
import scipy.stats as stats

def load_team_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {team['team']: team['position'] for team in data['teams']}

ladder_data = load_team_data('2024_Ladder.json')
league_standings_data = load_team_data('../NeuralNetwork/league_standings_2024.json')

if ladder_data.keys() != league_standings_data.keys():
    print("Error: The two files have different teams.")
    exit()

teams = list(ladder_data.keys())
ladder_positions = np.array([ladder_data[team] for team in teams])
standings_positions = np.array([league_standings_data[team] for team in teams])

spearman_corr, _ = stats.spearmanr(ladder_positions, standings_positions)

average_absolute_error = np.mean(np.abs(ladder_positions - standings_positions))

rmse = np.sqrt(np.mean((ladder_positions - standings_positions) ** 2))


def rank_biased_overlap(l1, l2, p=0.9):
    overlap, rbo, weight = 0.0, 0.0, 1.0
    seen_1, seen_2 = set(), set()

    for i, (item_1, item_2) in enumerate(zip(l1, l2), start=1):
        seen_1.add(item_1)
        seen_2.add(item_2)
        current_overlap = len(seen_1 & seen_2)
        overlap = current_overlap / i
        rbo += overlap * weight
        weight *= p

    return (1 - p) * rbo


ladder_ranks = [team for team, pos in sorted(ladder_data.items(), key=lambda x: x[1])]
standings_ranks = [team for team, pos in sorted(league_standings_data.items(), key=lambda x: x[1])]
rbo_score = rank_biased_overlap(ladder_ranks, standings_ranks)

print(f"Spearman's Rank Correlation Coefficient: {spearman_corr:.4f}")
print(f"Average Absolute Error: {average_absolute_error:.4f}")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"Rank-Biased Overlap (RBO): {rbo_score:.4f}")
