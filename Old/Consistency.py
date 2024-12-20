import yaml
import pandas as pd

# Step 1: Load the YAML file containing historical match data
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Step 2: Preprocess the data to create a DataFrame and calculate the new fields
def preprocess_data(data, excluded_stats=None):
    rows = []
    for match in data:
        # Team1 vs Team2 data
        team1_data = match['Team1']
        team2_data = match['Team2']

        # Add new fields for Team1 (conceded stats from Team2)
        row_team1 = {
            'MatchId': match['MatchId'],
            'Team': team1_data['Name'],
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

    # Exclude unwanted stats
    if excluded_stats:
        df.drop(columns=excluded_stats, inplace=True)

    return df

# Step 3: Normalize stats to a range (e.g., 0-1 or 0-10)
def normalize_stats(df, new_min=0, new_max=1):
    numeric_columns = df.columns.difference(['Team', 'MatchId'])  # Only numeric columns
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:  # Avoid division by zero
            df[col] = ((df[col] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        else:
            df[col] = new_min  # If all values are the same, assign the minimum value of the new range
    return df

# Step 4: Calculate consistency for each team over 2, 4, 6, and 8 games
def calculate_consistency(df):
    periods = [2, 4, 6, 8]
    consistency = {}

    for team in df['Team'].unique():
        team_df = df[df['Team'] == team].sort_values('MatchId')

        consistency[team] = {}
        numeric_columns = team_df.columns.difference(['MatchId', 'Team'])

        for period in periods:
            rolling_means = team_df[numeric_columns].rolling(window=period, min_periods=1).mean()
            absolute_variations = (team_df[numeric_columns] - rolling_means).abs()
            total_variation = absolute_variations.sum()

            # Round the total variation to 3 decimal places
            total_variation = total_variation.round(3)

            # Store the total variation for the current period
            consistency[team][f'Consistency_{period}_games'] = total_variation.to_dict()

    return consistency

# Step 5: Save the consistency data to a YAML file
def save_consistency_to_yaml(consistency_data, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(consistency_data, file, default_flow_style=False)

# Main function to load, process, and save consistency
def main(input_yaml, output_yaml, excluded_stats=None, new_min=0, new_max=1):
    # Load the YAML match data
    data = load_yaml(input_yaml)

    # Preprocess the data into a DataFrame
    df = preprocess_data(data, excluded_stats=excluded_stats)

    # Normalize the stats if needed
    # df_normalized = normalize_stats(df, new_min=new_min, new_max=new_max)

    # Calculate consistency for each team
    consistency_data = calculate_consistency(df)

    # Save the consistency data to a new YAML file
    save_consistency_to_yaml(consistency_data, output_yaml)

    # Debugging: Print out the consistency data
    print("Consistency data saved to:", output_yaml)

# Replace with the actual path to your input and output YAML files
input_yaml = 'match_stats_2024/clean_match_data_2024.yaml'
output_yaml = 'match_stats_2024/team_consistency_2024.yaml'

# Specify which stats to exclude from normalization (if any)
excluded_stats = [
                  'Age',
                  'Experience',
                  'CoachesVotes',
                  'RatingPoints',
                  'xScore',
                  'xScoreRating']  # Example: You can add any stats you don't want to include

main(input_yaml, output_yaml, excluded_stats, new_min=0, new_max=1)
