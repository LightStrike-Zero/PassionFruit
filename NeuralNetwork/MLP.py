import torch
import torch.nn as nn
import pandas as pd
import yaml
import pickle  # Use pickle for scaler, not torch
from sklearn.preprocessing import StandardScaler

# Define the selected features once, to be shared across both training and prediction scripts
selected_features = [
    # 'Age',
    # 'Experience',
    # 'CoachesVotes',
    # 'RatingPoints',
    'Kicks',
    'Handballs',
    'Disposals',
    'DisposalEfficiency',
    'MetresGained',
    'Inside50s',
    'ContestedPossessions',
    'GroundBallGets',
    'Intercepts',
    'TotalClearances',
    'Marks',
    'ContestedMarks',
    'InterceptMarks',
    'ShotsAtGoal',
    'Goals',
    'Behinds',
    'Score',
    # 'xScore',
    # 'xScoreRating',
    'GoalAssists',
    'Tackles',
    'Hitouts',
    # 'GoalsFromKickIn',
    # 'BehindsFromKickIn',
    # 'PointsFromKickIn',
    # 'GoalsFromStoppage',
    # 'BehindsFromStoppage',
    # 'PointsFromStoppage',
    # 'GoalsFromTurnover',
    # 'BehindsFromTurnover',
    # 'PointsFromTurnover',
    # 'GoalsFromDefensiveHalf',
    # 'BehindsFromDefensiveHalf',
    # 'PointsFromDefensiveHalf',
    # 'GoalsFromForwardHalf',
    # 'BehindsFromForwardHalf',
    # 'PointsFromForwardHalf',
    # 'GoalsFromCentreBounce',
    # 'BehindsFromCentreBounce',
    # 'PointsFromCentreBounce'
]  # Example set of features for both training and prediction

# Define the neural network model
class WinPredictorMLP(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(WinPredictorMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Define custom feature engineering
def add_custom_features(data):
    rows = []
    for match in data:
        match_id = match['MatchId']
        team1_data = match['Team1']
        team2_data = match['Team2']

        # Custom features
        team1_score_conceded = team2_data.get('Score', 0)  # Score conceded by team1
        team2_score_conceded = team1_data.get('Score', 0)  # Score conceded by team2

        # For Team1
        team1_row = {
            'MatchId': match_id,
            'Team': team1_data['Name'],
            'Round': int(str(match_id)[4:6]),  # Extract round number
            'Result': 1 if team1_data.get('Result', '') == 'W' else 0,  # Result for team1
            'OppositionScore': team2_data['Score'],
            'OppositionInside50s': team2_data['Inside50s'],
            'Turnovers': team2_data['Intercepts'],  # Opposition intercepts
            'OppositionTackles': team2_data['Tackles'],  # Opposition tackles
            'MetersLost': team2_data['MetresGained'],  # Opposition meters gained
            'OppositionContestedPossessions': team2_data['ContestedPossessions'],
            'OppositionShotsAtGoal': team2_data['ShotsAtGoal'],
            'OppositionClearances': team2_data['TotalClearances']
        }
        team1_row.update({key: team1_data.get(key, 0) for key in selected_features})  # Default features

        # For Team2
        team2_row = {
            'MatchId': match_id,
            'Team': team2_data['Name'],
            'Round': int(str(match_id)[4:6]),  # Extract round number
            'Result': 1 if team2_data.get('Result', '') == 'W' else 0,  # Result for team2
            'OppositionScore': team1_data['Score'],
            'OppositionInside50s': team1_data['Inside50s'],
            'Turnovers': team1_data['Intercepts'],
            'OppositionTackles': team1_data['Tackles'],
            'MetersLost': team1_data['MetresGained'],
            'OppositionContestedPossessions': team1_data['ContestedPossessions'],
            'OppositionShotsAtGoal': team1_data['ShotsAtGoal'],
            'OppositionClearances': team1_data['TotalClearances']
        }
        team2_row.update({key: team2_data.get(key, 0) for key in selected_features})  # Default features

        rows.append(team1_row)
        rows.append(team2_row)

    df = pd.DataFrame(rows)

    # Ensure only numeric columns are processed
    numeric_columns = df.columns.difference(['Team', 'MatchId'])
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing data
    df.dropna(inplace=True)

    return df

def preprocess_data(data, selected_features=None):
    rows = []
    for match in data:
        match_id = match['MatchId']

        # Access Team1 and Team2 data for this match
        team1_data = match['Team1']
        team2_data = match['Team2']

        # Calculate scoreConceded for each team
        team1_score_conceded = team2_data.get('Score', 0)  # Team1 conceded Team2's score
        team2_score_conceded = team1_data.get('Score', 0)  # Team2 conceded Team1's score

        # Handle 'Result' safely for Team1
        team1_result = team1_data.get('Result', None)
        if isinstance(team1_result, list):
            team1_result = team1_result[0]
        team1_win_result = 1 if team1_result == 'W' else 0

        # Handle 'Result' safely for Team2
        team2_result = team2_data.get('Result', None)
        if isinstance(team2_result, list):
            team2_result = team2_result[0]
        team2_win_result = 1 if team2_result == 'W' else 0

        # Create a row for Team1
        team1_row = {
            'MatchId': match_id,
            'Team': team1_data['Name'],
            'Result': team1_win_result,
            'ScoreConceded': team1_score_conceded  # Engineered stat
        }

        # Create a row for Team2
        team2_row = {
            'MatchId': match_id,
            'Team': team2_data['Name'],
            'Result': team2_win_result,
            'ScoreConceded': team2_score_conceded  # Engineered stat
        }

        # Add selected features for Team1
        if selected_features:
            team1_row.update({key: team1_data.get(key, 0) for key in selected_features})
        else:
            team1_row.update({key: team1_data.get(key, 0) for key in team1_data if key not in ['Name', 'Abbreviation', 'Result']})

        # Add selected features for Team2
        if selected_features:
            team2_row.update({key: team2_data.get(key, 0) for key in selected_features})
        else:
            team2_row.update({key: team2_data.get(key, 0) for key in team2_data if key not in ['Name', 'Abbreviation', 'Result']})

        # Append rows for both teams
        rows.append(team1_row)
        rows.append(team2_row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Ensure only numeric columns are processed
    numeric_columns = df.columns.difference(['Team', 'MatchId'])
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing data
    df.dropna(inplace=True)

    return df




# Load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


# Load the trained model
def load_model(model_path, input_size):
    model = WinPredictorMLP(input_size, Dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode for inference
    return model


# Load the scaler used during training
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)  # Corrected: Use pickle.load instead of torch.load for the scaler


# Save the trained model (for training script)
def save_model(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# Save the scaler used during training (for training script)
def save_scaler(scaler, scaler_save_path):
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_save_path}")

Dropout_rate = 0.35
