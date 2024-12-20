import torch
from NeuralNetwork.MLP import preprocess_data, load_model, load_scaler, selected_features, load_yaml
import pandas as pd
import yaml
import os
import pickle  # To save Elo ratings
import matplotlib.pyplot as plt
import numpy as np

year = 2024


# Function to get stats from the last N matches for a given team up to a certain round
def get_historic_stats(df, team_name, current_round, selected_features, n=3):
    team_df = df[(df['Team'] == team_name) & (df['Round'] < current_round)].sort_values('Round', ascending=False)
    team_df = team_df.head(n)
    if not team_df.empty:
        avg_stats = team_df[selected_features].mean()
        return avg_stats.values.reshape(1, -1)
    else:
        return None


# Function to calculate the Elo probability for a team
def elo_probability(elo1, elo2):
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))


# Function to update Elo ratings after a match with dynamic K factor based on score difference
def update_rating(rating, expected, actual, score_diff):
    k = max(30, abs(score_diff))  # Dynamic K based on score difference, minimum of 30
    return rating + k * (actual - expected)


# Combine MLP and Elo probabilities by taking the maximum of the two
def combine_probabilities(mlp_prob, elo_prob):
    return max(mlp_prob, elo_prob)


# Function to create Heatmap Plot of probabilities
def create_heatmap_plot(round_numbers, match_numbers, combined_probs_matrix):
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(combined_probs_matrix, cmap="coolwarm", aspect="auto", vmin=0.5, vmax=1)

    plt.colorbar(heatmap, label="Combined Probability")
    plt.xticks(ticks=np.arange(len(round_numbers)), labels=round_numbers, rotation=45)
    plt.yticks(ticks=np.arange(len(match_numbers)), labels=match_numbers)

    plt.xlabel("Rounds")
    plt.ylabel("Matches")
    plt.title("Combined Probabilities Heatmap")

    plt.show()


# Main prediction script
def main():
    model_path = '../../NeuralNetwork/trained_model.pth'
    scaler_path = '../../NeuralNetwork/scaler.pkl'
    yaml_file = f'../match_stats_{2024}/clean_match_data_{2024}.yaml'
    elo_ratings_file = f'../EloRating/elo_ratings_ongoing_{2024}.pkl'
    base_elo_file = '../EloRating/base_elo_ratings.yaml'

    # Load data from YAML
    data = load_yaml(yaml_file)

    if not os.path.exists(elo_ratings_file):
        print(f"Elo ratings file '{elo_ratings_file}' not found. Initializing with base Elo ratings.")
        if os.path.exists(base_elo_file):
            with open(base_elo_file, 'r') as base_file:
                base_elo_data = yaml.safe_load(base_file)
            elo_ratings = base_elo_data
        else:
            print(f"Error: Base Elo file '{base_elo_file}' not found. Cannot initialize Elo ratings.")
            return
    else:
        with open(elo_ratings_file, 'rb') as file:
            elo_ratings_data = pickle.load(file)
            elo_ratings = elo_ratings_data['current_ratings']

    df = preprocess_data(data, selected_features)
    df['Round'] = df['MatchId'].apply(lambda x: int(str(x)[4:6]))

    input_size = len(selected_features)
    model = load_model(model_path, input_size)
    scaler = load_scaler(scaler_path)

    correct_predictions = 0
    total_predictions = 0
    round_accuracy = {}
    ratings_by_round = {}

    # Collect data for heatmap
    combined_probs = []
    rounds = []
    matches_per_round = 8

    # Dictionary to store probabilities for each round
    round_probabilities = {}

    for match_id in df['MatchId'].unique():
        match_df = df[df['MatchId'] == match_id]
        current_round = match_df['Round'].iloc[0]

        team1_name = match_df.iloc[0]['Team']
        team2_name = match_df.iloc[1]['Team']
        team1_score = match_df.iloc[0]['Score']
        team2_score = match_df.iloc[1]['Score']

        # Get historical stats for each team
        X_team1 = get_historic_stats(df, team1_name, current_round, selected_features)
        X_team2 = get_historic_stats(df, team2_name, current_round, selected_features)

        if X_team1 is None or X_team2 is None:
            print(f"\033[93mInsufficient data for match {match_id}. Reverting to Elo only prediction.\033[0m")
            team1_elo = elo_ratings.get(team1_name, 1500)
            team2_elo = elo_ratings.get(team2_name, 1500)

            team1_elo_prob = elo_probability(team1_elo, team2_elo)
            team2_elo_prob = 1 - team1_elo_prob
            predicted_winner = team1_name if team1_elo_prob > team2_elo_prob else team2_name
            team1_combined_prob = team1_elo_prob
            team2_combined_prob = team2_elo_prob

        else:
            X_team1 = scaler.transform(X_team1)
            X_team2 = scaler.transform(X_team2)

            X_team1_tensor = torch.tensor(X_team1, dtype=torch.float32)
            X_team2_tensor = torch.tensor(X_team2, dtype=torch.float32)

            team1_mlp_prob = model(X_team1_tensor).item()
            team2_mlp_prob = model(X_team2_tensor).item()

            team1_elo = elo_ratings.get(team1_name, 1500)
            team2_elo = elo_ratings.get(team2_name, 1500)

            team1_elo_prob = elo_probability(team1_elo, team2_elo)
            team2_elo_prob = 1 - team1_elo_prob

            team1_combined_prob = combine_probabilities(team1_mlp_prob, team1_elo_prob)
            team2_combined_prob = combine_probabilities(team2_mlp_prob, team2_elo_prob)

            predicted_winner = team1_name if team1_combined_prob > team2_combined_prob else team2_name

        # Get the actual result from the match
        actual_winner = team1_name if match_df.iloc[0]['Result'] == 1 else team2_name
        actual_score_team1 = 1 if actual_winner == team1_name else 0
        actual_score_team2 = 1 - actual_score_team1

        # Update Elo ratings after the match using dynamic k-factor
        score_diff = abs(team1_score - team2_score)
        expected_score_team1 = team1_elo_prob
        expected_score_team2 = 1 - expected_score_team1

        new_team1_elo = update_rating(team1_elo, expected_score_team1, actual_score_team1, score_diff)
        new_team2_elo = update_rating(team2_elo, expected_score_team2, actual_score_team2, score_diff)

        # Update Elo ratings
        elo_ratings[team1_name] = new_team1_elo
        elo_ratings[team2_name] = new_team2_elo

        # Store the new Elo ratings for the round
        if current_round not in ratings_by_round:
            ratings_by_round[current_round] = {}
        ratings_by_round[current_round][team1_name] = new_team1_elo
        ratings_by_round[current_round][team2_name] = new_team2_elo

        # Store probabilities for heatmap, grouped by round
        if current_round not in round_probabilities:
            round_probabilities[current_round] = []
        round_probabilities[current_round].append(
            team1_combined_prob if predicted_winner == team1_name else team2_combined_prob
        )

        # Determine if the prediction was correct
        is_correct = predicted_winner == actual_winner
        correct_predictions += 1 if is_correct else 0
        total_predictions += 1

    # Prepare matrix for heatmap with padding if rounds have fewer than 8 matches
    combined_probs_matrix = []
    for round_num in sorted(round_probabilities.keys()):
        round_probs = round_probabilities[round_num]
        # Pad if fewer than 8 matches
        if len(round_probs) < matches_per_round:
            round_probs += [0] * (matches_per_round - len(round_probs))  # Fill missing with zeros
        combined_probs_matrix.append(round_probs)

    combined_probs_matrix = np.array(combined_probs_matrix).T  # Transpose for correct orientation

    # Create labels for rounds and matches
    round_labels = [f"Round {i}" for i in sorted(round_probabilities.keys())]
    match_labels = [f"Match {i + 1}" for i in range(matches_per_round)]

    # Call the function to create the heatmap plot
    create_heatmap_plot(round_labels, match_labels, combined_probs_matrix)


if __name__ == '__main__':
    main()
