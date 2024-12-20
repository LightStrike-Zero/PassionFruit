import math
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from NeuralNetwork.MLP import preprocess_data, load_model, load_scaler, selected_features, load_yaml
from Utilities.AcquireData import acquire_data
import pandas as pd
import yaml
import numpy as np
import os
import json

current_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
target_round = 24  # this is only used if the year was in progress, for example round 16 being the current round we stop there
mode = sys.argv[2] if len(sys.argv) > 2 else "out-of-season"


# track points, ELO for each team
team_stats = {}
match_results = []

# check if data file exists or acquire it if not
def ensure_data_availability(year):
    data_file = f'../match_stats_{year}/clean_match_data_{year}.yaml'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Acquiring data for year {year}...")
        acquire_data(year)
    else:
        print(f"Data file {data_file} found. Proceeding with prediction...")

# update team stats for each team at the start of each round
def initialize_team_stats(team_name, elo_rating):
    if team_name not in team_stats:
        team_stats[team_name] = {
            "points": 0,
            "elo": elo_rating
        }

# Update team points based on the match result
def update_team_points(team_name, points):
    team_stats[team_name]["points"] += points

# Load base Elo ratings from the previous season
def load_base_elo_ratings(year):
    with open(f'../match_stats_{year}/elo_ratings_{year}.yaml', 'r') as file:
        return yaml.safe_load(file)

# Load saved Elo state
def load_saved_state(elo_file):
    if os.path.exists(elo_file):
        with open(elo_file, 'r') as f:
            running_elo_ratings = yaml.safe_load(f)
        return running_elo_ratings
    else:
        return None

# Save Elo ratings
def save_state(elo_file, running_elo_ratings, round_num=None):
    if round_num is not None:
        base, ext = os.path.splitext(elo_file)
        elo_file = f"{base}_round_{round_num}{ext}"
    sorted_elo_ratings = {team: float(rating) for team, rating in sorted(running_elo_ratings.items(), key=lambda item: item[1], reverse=True)}
    with open(elo_file, 'w') as f:
        yaml.dump(sorted_elo_ratings, f, default_flow_style=False, sort_keys=False)

# Save MLP
def save_model(model, filename='mlp_model_adjusted.pth'):
    torch.save(model.state_dict(), filename)

# update Elo ratings
def update_elo(rating1, rating2, result, k):
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    new_rating1 = rating1 + k * (result - expected1)
    new_rating2 = rating2 + k * ((1 - result) - (1 - expected1))
    return new_rating1, new_rating2


# calculate rolling average stats - used in-season only
def get_historic_stats(df, team_name, current_round, selected_features, n=3):
    team_df = df[(df['Team'] == team_name) & (df['Round'] < current_round)].sort_values('Round', ascending=False)
    team_df = team_df.head(n)
    if not team_df.empty:
        avg_stats = team_df[selected_features].mean()
        return avg_stats.values.reshape(1, -1)
    else:
        return None

# previous season's average stats - out-of-season only
def load_average_stats(year):
    with open(f"../match_stats_{year}/average_stats_{year}_last_3_rounds.json", 'r') as f:
        return json.load(f)


# display match prediction results, not using now
def display_prediction(match_id, team1_name, team2_name, elo1_prob, elo2_prob, team1_prob, team2_prob,
                       combined_prob_team1, combined_prob_team2, predicted_winner, actual_winner):
    is_correct = predicted_winner == actual_winner
    result_text = f"Match {match_id}: {team1_name} vs {team2_name}\n" \
                  f"  Elo Prediction -> {team1_name}: {elo1_prob:.2f}, {team2_name}: {elo2_prob:.2f}\n" \
                  f"  MLP Prediction -> {team1_name}: {team1_prob:.2f}, {team2_name}: {team2_prob:.2f}\n" \
                  f"  Combined Prediction -> {team1_name}: {combined_prob_team1:.2f}, {team2_name}: {combined_prob_team2:.2f}\n" \
                  f"  Predicted Winner: {predicted_winner} | Actual Winner: {actual_winner}\n" \
                  f"  {'Correct' if is_correct else 'Incorrect'} Prediction"
    color_code = "\033[92m" if is_correct else "\033[91m"
    reset_code = "\033[0m"
    print(f"{color_code}{result_text}{reset_code}")
    print('-' * 80)

def display_ladder(round_num):
    sorted_teams = sorted(
        team_stats.items(),
        key=lambda item: (item[1]["points"], item[1]["elo"]),
        reverse=True
    )
    print(f"\nLadder Standings After Round {round_num}")
    print(f"{'Position':<10}{'Team':<30}{'Points':<10}{'ELO':<10}")
    print("=" * 50)
    for position, (team, stats) in enumerate(sorted_teams, start=1):
        print(f"{position:<10}{team:<30}{stats['points']:<10}{stats['elo']:<10.2f}")
    print("\n" + "=" * 50 + "\n")

# prediction script
def main():
    ensure_data_availability(current_year)
    base_elo_ratings = load_base_elo_ratings(current_year)
    elo_file = f'../match_stats_{current_year}/running_elo_ratings_{current_year}.yaml'
    running_elo_ratings = load_saved_state(elo_file) or base_elo_ratings.copy()
    model_path = 'trained_model.pth'
    scaler_path = 'scaler.pkl'
    yaml_file = f'../match_stats_{current_year}/clean_match_data_{current_year}.yaml'
    data = load_yaml(yaml_file)

    df = preprocess_data(data, selected_features)
    df['Round'] = df['MatchId'].apply(lambda x: int(str(x)[4:6]))

    input_size = len(selected_features)
    model = load_model(model_path, input_size)
    scaler = load_scaler(scaler_path)

    correct_predictions = 0
    total_predictions = 0
    round_accuracies = {}

    if mode == "out-of-season":
        avg_stats_data = load_average_stats(current_year - 1)
    else:
        avg_stats_data = None

    for round_num in sorted(df['Round'].unique()):
        if round_num > target_round:
            break  # Stop if we reach the target round

        round_matches = df[df['Round'] == round_num]
        round_correct = 0
        round_total = 0

        for match_id in round_matches['MatchId'].unique():
            match_df = round_matches[round_matches['MatchId'] == match_id]
            team1_name = match_df.iloc[0]['Team']
            team2_name = match_df.iloc[1]['Team']
            initialize_team_stats(team1_name, running_elo_ratings.get(team1_name, 1500))
            initialize_team_stats(team2_name, running_elo_ratings.get(team2_name, 1500))

            # Get MLP data
            if mode == "in-season" and round_num > 1:
                X_team1 = get_historic_stats(df, team1_name, round_num, selected_features)
                X_team2 = get_historic_stats(df, team2_name, round_num, selected_features)
            elif mode == "out-of-season":
                X_team1 = np.array([avg_stats_data[team1_name][feature] for feature in selected_features]).reshape(1,
                                                                                                                   -1)
                X_team2 = np.array([avg_stats_data[team2_name][feature] for feature in selected_features]).reshape(1,
                                                                                                                   -1)
            else:
                X_team1 = X_team2 = None  # Round 1 in in-season mode we have no data yet so efault 0.5

            if X_team1 is not None and X_team2 is not None:
                X_team1 = scaler.transform(X_team1)
                X_team2 = scaler.transform(X_team2)
                team1_prob = model(torch.tensor(X_team1, dtype=torch.float32)).item()
                team2_prob = model(torch.tensor(X_team2, dtype=torch.float32)).item()
            else:
                team1_prob = team2_prob = 0.5
                X_team1_tensor = X_team2_tensor = None

            # Get current Elo
            elo1 = float(running_elo_ratings.get(team1_name, 1500))
            elo2 = float(running_elo_ratings.get(team2_name, 1500))
            elo1_prob = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
            elo2_prob = 1 - elo1_prob

            # Adjust ELO weight
            w_elo = 0.85 if mode == "out-of-season" else 0.70

            # Calculate combined probabilities
            combined_prob_team1 = w_elo * elo1_prob + (1 - w_elo) * team1_prob
            combined_prob_team2 = w_elo * elo2_prob + (1 - w_elo) * team2_prob

            # predicted winner based on the highest combined probability
            predicted_winner = team1_name if combined_prob_team1 > combined_prob_team2 else team2_name

            actual_winner = team1_name if match_df.iloc[0]['Result'] == 1 else team2_name
            is_correct = predicted_winner == actual_winner
            correct_predictions += 1 if is_correct else 0
            total_predictions += 1
            round_correct += 1 if is_correct else 0
            round_total += 1

            if actual_winner == team1_name:
                update_team_points(team1_name, 4)
                update_team_points(team2_name, 0)
            elif actual_winner == team2_name:
                update_team_points(team1_name, 0)
                update_team_points(team2_name, 4)
            else:
                # Draw case
                update_team_points(team1_name, 2)
                update_team_points(team2_name, 2)

            match_result = {
                "MatchID": match_id,
                "Team1": team1_name,
                "Team2": team2_name,
                "Team1_Prob": combined_prob_team1,
                "Team2_Prob": combined_prob_team2
            }
            match_results.append(match_result)

            # online learning for mlp for in-season only
            if mode == "in-season" and X_team1_tensor is not None and X_team2_tensor is not None:
                actual_result = torch.tensor([1 if actual_winner == team1_name else 0], dtype=torch.float32).unsqueeze(
                    0)
                output = model(X_team1_tensor if predicted_winner == team1_name else X_team2_tensor)

                alpha = 0.95
                actual_result = 1 if actual_winner == team1_name else 0
                model_prediction = team1_prob if predicted_winner == team1_name else team2_prob
                blended_target = alpha * actual_result + (1 - alpha) * model_prediction
                target_tensor = torch.tensor([blended_target], dtype=torch.float32).unsqueeze(0)
                output = model(X_team1_tensor if predicted_winner == team1_name else X_team2_tensor)
                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.11)
                loss = criterion(output, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            team1_score = match_df.iloc[0]['Score']
            team2_score = match_df.iloc[1]['Score']

            # determine the weight to apply to losing/winning when upodate elo (trial and error for 20)
            k = max(20, abs(team1_score - team2_score))

            result = 1 if actual_winner == team1_name else 0
            new_elo1, new_elo2 = update_elo(elo1, elo2, result, k)
            running_elo_ratings[team1_name] = new_elo1
            running_elo_ratings[team2_name] = new_elo2

            display_prediction(
                match_id=match_id,
                team1_name=team1_name,
                team2_name=team2_name,
                elo1_prob=elo1_prob,
                elo2_prob=elo2_prob,
                team1_prob=team1_prob,
                team2_prob=team2_prob,
                combined_prob_team1=combined_prob_team1,
                combined_prob_team2=combined_prob_team2,
                predicted_winner=predicted_winner,
                actual_winner=actual_winner
            )

            round_accuracy = round(round_correct / round_total, 2) if round_total > 0 else 0
            round_accuracies[round_num] = round_accuracy

        save_state(elo_file, running_elo_ratings)
        overall_accuracy = round(correct_predictions / total_predictions, 2) if total_predictions > 0 else 0
        print(f"Overall accuracy: {overall_accuracy}")
        results = {
            "overall_accuracy": float(f"{overall_accuracy:.2f}"),
            "round_accuracies": {int(round_num): float(f"{acc:.2f}") for round_num, acc in round_accuracies.items()}
        }

        save_state(elo_file, running_elo_ratings)
        save_model(model, filename='mlp_model_adjusted.pth')

        with open(f"../Simulation/results_summary_{current_year}.yaml", "w") as f:
            yaml.dump(results, f)
            sorted_teams = sorted(
                team_stats.items(),
                key=lambda item: (item[1]["points"], item[1]["elo"]),
                reverse=True
            )
            results = {
                "year": current_year,
                "teams": [
                    {
                        "position": i + 1,
                        "team": team,
                        "points": stats["points"],
                        "elo": stats["elo"]
                    }
                    for i, (team, stats) in enumerate(sorted_teams)
                ]
            }
            with open(f"../Simulation/league_standings_{current_year}.json", "w") as f:
                json.dump(results, f, indent=4)
            with open(f"../Simulation/match_results_{current_year}.json", "w") as f:
                json.dump(match_results, f, indent=4)
            print(f"League standings for {current_year} have been saved to /Simulation/league_standings_{current_year}.json")

            display_ladder(round_num)

if __name__ == '__main__':
    main()