# predict_model.py

import torch
from MLP import preprocess_data, load_model, load_scaler, selected_features, load_yaml
import pandas as pd
import yaml


# Function to get stats from the last N matches for a given team up to a certain round
def get_historic_stats(df, team_name, current_round, selected_features, n=3):
    # Filter for the team's games before the current round
    team_df = df[(df['Team'] == team_name) & (df['Round'] < current_round)].sort_values('Round', ascending=False)

    # If there are fewer than N games, use all available games
    team_df = team_df.head(n)

    # Calculate rolling averages over the last N games (or use raw data, depending on your preference)
    if not team_df.empty:
        avg_stats = team_df[selected_features].mean()
        return avg_stats.values.reshape(1, -1)
    else:
        return None


# Main prediction script
def main():
    # Load the saved model and scaler
    model_path = 'trained_model.pth'
    scaler_path = 'scaler.pkl'
    yaml_file = '../match_stats_2024/clean_match_data_2024.yaml'

    # Load data from YAML
    data = load_yaml(yaml_file)

    # Preprocess the data and add a 'Round' column
    df = preprocess_data(data, selected_features)
    df['Round'] = df['MatchId'].apply(lambda x: int(str(x)[4:6]))

    # Load the model and scaler
    input_size = len(selected_features)
    model = load_model(model_path, input_size)
    scaler = load_scaler(scaler_path)

    # Track accuracy
    correct_predictions = 0
    total_predictions = 0
    round_accuracy = {}

    # Iterate over each match and predict the outcome
    for match_id in df['MatchId'].unique():
        match_df = df[df['MatchId'] == match_id]
        current_round = match_df['Round'].iloc[0]

        # Extract team names
        team1_name = match_df.iloc[0]['Team']
        team2_name = match_df.iloc[1]['Team']

        # Get historical stats for each team
        X_team1 = get_historic_stats(df, team1_name, current_round, selected_features)
        X_team2 = get_historic_stats(df, team2_name, current_round, selected_features)

        if X_team1 is None or X_team2 is None:
            print(f"Insufficient data for match {match_id}. Skipping prediction.\n")
            continue

        # Standardize the features
        X_team1 = scaler.transform(X_team1)
        X_team2 = scaler.transform(X_team2)

        # Convert to PyTorch tensors
        X_team1_tensor = torch.tensor(X_team1, dtype=torch.float32)
        X_team2_tensor = torch.tensor(X_team2, dtype=torch.float32)

        # Predict the outcome
        team1_prob = model(X_team1_tensor).item()
        team2_prob = model(X_team2_tensor).item()

        predicted_winner = team1_name if team1_prob > team2_prob else team2_name

        # Get the actual result from the YAML data
        actual_winner = team1_name if match_df.iloc[0]['Result'] == 1 else team2_name

        # Compare prediction to actual result
        is_correct = predicted_winner == actual_winner
        correct_predictions += 1 if is_correct else 0
        total_predictions += 1

        # Track accuracy per round
        if current_round not in round_accuracy:
            round_accuracy[current_round] = {'correct': 0, 'total': 0}
        round_accuracy[current_round]['correct'] += 1 if is_correct else 0
        round_accuracy[current_round]['total'] += 1

        # Output prediction with team names and whether it was correct
        if is_correct:
            print(f"\033[92mMatch {match_id}: {predicted_winner} is predicted to win.\033[0m")  # Green for correct
        else:
            print(f"\033[91mMatch {match_id}: {predicted_winner} is predicted to win.\033[0m")  # Red for incorrect

        print(f"Actual Winner: {actual_winner}")
        print(f"Correct Prediction: {'\033[92mTrue\033[0m' if is_correct else '\033[91mFalse\033[0m'}")
        print(f"{team1_name} Probability: {team1_prob:.4f}, {team2_name} Probability: {team2_prob:.4f}\n")

    # Calculate overall accuracy
    overall_accuracy = round(correct_predictions / total_predictions, 2) if total_predictions > 0 else 0

    # Calculate round-wise accuracy
    round_accuracies = {}
    for rnd, stats in round_accuracy.items():
        round_accuracies[rnd] = round(stats['correct'] / stats['total'], 2) if stats['total'] > 0 else 0

    # Output YAML file with the accuracy results
    accuracy_data = {
        'overall_accuracy': overall_accuracy,
        'round_accuracies': {int(k): v for k, v in round_accuracies.items()}  # Convert to native types
    }

    # Write accuracy data to a YAML file
    output_yaml = 'model_accuracy.yaml'
    with open(output_yaml, 'w') as outfile:
        yaml.dump(accuracy_data, outfile)

    print(f"Accuracy results saved to {output_yaml}")


if __name__ == '__main__':
    main()
