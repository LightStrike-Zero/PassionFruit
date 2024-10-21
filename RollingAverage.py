import yaml
import pandas as pd


# Step 1: Load the YAML file containing historical match data
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# Step 2: Preprocess the data to create a DataFrame and calculate the new fields
def preprocess_data(data):
    rows = []
    for match in data:
        # Team1 vs Team2 data
        team1_data = match['Team1']
        team2_data = match['Team2']

        # Add new fields for Team1 (conceded stats from Team2)
        row_team1 = {
            'MatchId': match['MatchId'],
            'Team': team1_data['Name'],
            'Round': int(str(match['MatchId'])[4:6]),  # Extract round number from 'MatchId'
            'OppositionScore': team2_data['Score'],
            'OppositionInside50s': team2_data['Inside50s'],
            'Turnovers': team2_data['Intercepts'],  # Opposition intercepts
            'OppositionTackles': team2_data['Tackles'],  # Opposition tackles
            'MetersLost': team2_data['MetresGained'],  # Opposition meters gained
            'OppositionContestedPossessions': team2_data['ContestedPossessions'],
            'OppositionShotsAtGoal': team2_data['ShotsAtGoal'],
            'OppositionClearances': team2_data['TotalClearances']
        }
        row_team1.update({key: value for key, value in team1_data.items() if key not in ['Name', 'Abbreviation', 'Result']})

        # Add new fields for Team2 (conceded stats from Team1)
        row_team2 = {
            'MatchId': match['MatchId'],
            'Team': team2_data['Name'],
            'Round': int(str(match['MatchId'])[4:6]),  # Extract round number from 'MatchId'
            'OppositionScore': team1_data['Score'],
            'OppositionInside50s': team1_data['Inside50s'],
            'Turnovers': team1_data['Intercepts'],
            'OppositionTackles': team1_data['Tackles'],
            'MetersLost': team1_data['MetresGained'],
            'OppositionContestedPossessions': team1_data['ContestedPossessions'],
            'OppositionShotsAtGoal': team1_data['ShotsAtGoal'],
            'OppositionClearances': team1_data['TotalClearances']
        }
        row_team2.update({key: value for key, value in team2_data.items() if key not in ['Name', 'Abbreviation', 'Result']})

        rows.append(row_team1)
        rows.append(row_team2)

    df = pd.DataFrame(rows)

    # Ensure all numeric data is properly converted
    numeric_columns = df.columns.difference(['Team', 'MatchId'])  # Only numeric columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return df


# Step 3: Calculate rolling averages up to the current round (window of 3 rounds)
def calculate_weighted_rolling_averages(df, current_round):
    periods = [3]  # Rolling window of 3 rounds
    weighted_avgs = {}

    for team in df['Team'].unique():
        # Filter data up to the current round
        team_df = df[(df['Team'] == team) & (df['Round'] <= current_round)].sort_values('Round')

        numeric_columns = team_df.columns.difference(['MatchId', 'Team', 'Round'])

        # Calculate rolling averages with a window of 3, and minimum periods as per the round
        rolling_means = team_df[numeric_columns].rolling(window=3, min_periods=1).mean()

        # Round all values to 1 decimal place
        rolling_means = rolling_means.round(1)

        # Get the last row, which represents the rolling average up to the current round
        weighted_avgs[team] = rolling_means.iloc[-1].to_dict()

    return weighted_avgs


# Step 4: Save the rolling averages to a YAML file
def save_rolling_averages_to_yaml(rolling_averages, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(rolling_averages, file)


# Main function to load, process, and save rolling averages up to current round
def main(input_yaml, output_yaml, current_round):
    # Load the YAML match data
    data = load_yaml(input_yaml)

    # Preprocess the data into a DataFrame
    df = preprocess_data(data)

    # Calculate rolling averages for each team up to the current round
    rolling_averages = calculate_weighted_rolling_averages(df, current_round)

    # Save the rolling averages to a new YAML file
    save_rolling_averages_to_yaml(rolling_averages, output_yaml)

    print(f"Rolling averages up to round {current_round} saved to: {output_yaml}")

year = '2024'
current_round = 5

# Replace with the actual path to your input and output YAML files
input_yaml = f'match_stats_{year}/clean_match_data_{year}.yaml'
output_yaml = f'match_stats_{year}/rolling_averages_{year}.yaml'
main(input_yaml, output_yaml, current_round)
